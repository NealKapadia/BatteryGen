"""Continued pre-training: append NEW molecule sources as fresh shards.

Ingests extra SMILES (ZINC tranches, solvent / ionic-liquid sets, electrolyte
candidate lists, your own files) through the SAME canonicalize -> filter ->
descriptors -> SELFIES pipeline, writes them as new shards (shard_013, ...), and
re-finalizes membership + meta over old+new shards. Then keep training with
``train.py --resume`` on the expanded dataset.

By default the descriptor normalization is KEPT (so an already-trained model's
conditioning stays calibrated); pass --recompute-stats only for a fresh model.

  python -m batterygen.generative.add_data --input zinc_tranche1.smi zinc_tranche2.smi --tag zinc
  python -m batterygen.generative.add_data --input solvents.csv --tag solvent --delim , --smiles-col 0 --header
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

from batterygen.core import config

from batterygen.core import data

from batterygen.generative import preprocess



def _iter_sdf(path: Path, id_prefix: str):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rb") as fh:
        n = 0
        for mol in Chem.ForwardSDMolSupplier(fh, sanitize=True):
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                continue
            if smi:
                yield smi, f"{id_prefix}-{n}"
                n += 1


def _iter_csv(path: Path, smiles_col: int, id_prefix: str, header: bool):
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore", newline="") as fh:
        r = csv.reader(fh)
        if header:
            next(r, None)
        n = 0
        for parts in r:
            if smiles_col < len(parts):
                smi = parts[smiles_col].strip().strip('"').strip()
                if smi:
                    yield smi, f"{id_prefix}-{n}"
                    n += 1


def _iter_text(path: Path, smiles_col: int, delim, id_prefix: str, header: bool):
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        if header:
            f.readline()
        n = 0
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(delim) if delim else line.split()
            if not parts:
                continue
            smi = parts[smiles_col] if smiles_col < len(parts) else parts[0]
            yield smi, f"{id_prefix}-{n}"
            n += 1


def _iter_records(path: Path, smiles_col: int, delim, id_prefix: str, header: bool):
    name = path.name.lower()
    if name.endswith((".sdf", ".sdf.gz", ".mol")):
        yield from _iter_sdf(path, id_prefix)
    elif name.endswith((".csv", ".csv.gz")) or delim == ",":
        yield from _iter_csv(path, smiles_col, id_prefix, header)
    else:
        yield from _iter_text(path, smiles_col, delim, id_prefix, header)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="SMILES files (.smi/.txt/.csv[.gz])")
    ap.add_argument("--tag", default="ext", help="source label / id prefix (e.g. zinc, solvent)")
    ap.add_argument("--smiles-col", type=int, default=0)
    ap.add_argument("--delim", default=None, help="field delimiter (default: any whitespace)")
    ap.add_argument("--header", action="store_true", help="input files have a header row")
    ap.add_argument("--workers", type=int, default=config.N_WORKERS)
    ap.add_argument("--chunksize", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="max molecules to KEEP per input file (0=all)")
    ap.add_argument("--dedup", action="store_true",
                    help="skip molecules already in the existing training set (avoid leakage/duplication)")
    ap.add_argument("--recompute-stats", action="store_true",
                    help="recompute descriptor normalization over ALL data (changes "
                         "conditioning calibration; only for a model you'll train from scratch)")
    ap.add_argument("--bloom-capacity", type=int, default=10_000_000)
    ap.add_argument("--bloom-fp", type=float, default=1e-4)
    args = ap.parse_args()

    config.ensure_dirs()
    if not config.VOCAB_PATH.exists():
        data.Vocab.build().save()

    meta = json.load(open(config.META_PATH, encoding="utf-8")) if config.META_PATH.exists() else {"shards": []}
    existing = list(meta.get("shards", []))
    next_idx = len(existing)
    print(f"Existing shards: {len(existing)}. Appending {len(args.input)} source(s) ...")

    dedup = None
    if args.dedup:
        from batterygen.core.membership import MolportIndex
        _idx = MolportIndex()
        if _idx.available:
            dedup = lambda c: _idx.contains(c, already_canonical=True)
            print("  dedup ON: skipping molecules already in the training set")

    new_shards = []
    for k, inp in enumerate(args.input):
        inp = Path(inp)
        if not inp.exists():
            print(f"  ! {inp} not found, skipping")
            continue
        shard_name = f"shard_{next_idx + len(new_shards):03d}"
        records = _iter_records(inp, args.smiles_col, args.delim, args.tag, args.header)
        kept = preprocess.process_file(records, shard_name, f"{args.tag}:{inp.name}",
                                       workers=args.workers, chunksize=args.chunksize,
                                       remaining=args.limit, dedup=dedup)
        if kept > 0:
            new_shards.append(shard_name)

    # release the read-only dedup handle BEFORE finalize() deletes/rebuilds the DB
    # (Windows can't unlink a file with an open handle).
    if dedup is not None:
        _idx.close()

    if not new_shards:
        print("No new molecules added.")
        return

    all_shards = existing + new_shards
    preprocess.finalize(all_shards, args.bloom_capacity, args.bloom_fp,
                        recompute_stats=args.recompute_stats)

    # the scaffold split no longer aligns (length changed) -> drop it
    sp = config.PROC_DIR / "split_val.npy"
    if sp.exists():
        sp.unlink()
        print("Removed stale split_val.npy — re-run make_split.py if you want a scaffold split.")

    print(f"\nAdded {len(new_shards)} shard(s). Continue pre-training with:")
    print("  python -m batterygen.generative.train --resume --batch 320")


if __name__ == "__main__":
    main()
