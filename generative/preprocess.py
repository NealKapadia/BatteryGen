"""Streaming preprocessor: Molport SMILES(.gz) -> token shards + descriptor shards
+ Molport membership index.

Single pass, multiprocessing across CPU cores. The SELFIES vocab is the fixed
robust alphabet so no vocab-building pass is needed. Output is written per input
chunk (resumable: existing shards are skipped unless --force).

Usage:
    python -m batterygen.generative.preprocess --limit 50000      # quick subset
    python -m batterygen.generative.preprocess                     # full catalog
    python -m batterygen.generative.preprocess --shards 0 1        # only the first two chunks
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from batterygen.core import config

from batterygen.core import data

from batterygen.core import membership


# Worker-global vocab (loaded once per process by the Pool initializer).
_VOCAB = None


def _init_worker():
    global _VOCAB
    from batterygen.core import data as _d  # re-import in spawned process

    _VOCAB = _d.Vocab.load()


def _worker(args):
    smiles, mid = args
    from batterygen.core import data as _d


    return _d.process_record(smiles, mid, _VOCAB)


def _iter_lines(path: Path):
    """Yield (smiles_for_parsing, molport_id) from a Molport SMILES .gz file."""
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        header = f.readline()  # SMILES \t SMILES_CANONICAL \t MOLPORTID
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            # prefer the pre-canonicalized column, fall back to raw SMILES
            smi = parts[1] if parts[1] else parts[0]
            yield smi, parts[2]


def _shard_done(base: Path) -> bool:
    return all(
        (base.parent / f"{base.name}_{suf}").exists()
        for suf in ("tokens.npy", "desc.npy", "len.npy", "ids.txt", "meta.json")
    )


def process_file(records, shard_name: str, source: str, *, workers: int, chunksize: int,
                 remaining: int, dedup=None) -> int:
    """Process an iterable of (smiles, id) records into shard_<name>_*. Returns #kept.

    dedup: optional callable(canon)->bool; when it returns True the molecule is skipped
    (used to drop molecules already present in the existing training set)."""
    import multiprocessing as mp

    base = config.PROC_DIR / shard_name
    ids_buf: List[List[int]] = []
    len_buf: List[int] = []
    desc_buf: List[List[float]] = []
    canon_ids: List[str] = []

    seen = kept = 0
    t0 = time.time()
    with mp.Pool(workers, initializer=_init_worker) as pool:
        it = pool.imap_unordered(_worker, records, chunksize=chunksize)
        bar = tqdm(it, desc=f"{shard_name}", unit="mol", dynamic_ncols=True)
        for rec in bar:
            seen += 1
            if rec is not None:
                canon, mid, ids, length, desc = rec
                if dedup is not None and dedup(canon):
                    continue
                ids_buf.append(ids)
                len_buf.append(length)
                desc_buf.append(desc)
                canon_ids.append(f"{canon}\t{mid}")
                kept += 1
                if kept % 5000 == 0:
                    bar.set_postfix(kept=kept, keep_rate=f"{kept/seen:.1%}")
                if remaining and kept >= remaining:
                    break
        bar.close()
        pool.terminate()  # stop feeding once we have enough / file is done

    if kept == 0:
        return 0

    np.save(f"{base}_tokens.npy", np.asarray(ids_buf, dtype=np.int16))
    np.save(f"{base}_desc.npy", np.asarray(desc_buf, dtype=np.float32))
    np.save(f"{base}_len.npy", np.asarray(len_buf, dtype=np.int16))
    with open(f"{base}_ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(canon_ids))
    with open(f"{base}_meta.json", "w", encoding="utf-8") as f:
        json.dump({"source": source, "seen": seen, "kept": kept}, f)
    dt = time.time() - t0
    print(f"  {shard_name}: kept {kept:,}/{seen:,} ({kept/max(seen,1):.1%}) in {dt/60:.1f} min")
    return kept


def finalize(shard_names: List[str], bloom_capacity: int, fp_rate: float,
             recompute_stats: bool = True) -> None:
    print("Finalizing: descriptor stats + membership index ...")

    # total molecule count (cheap: just shard shapes)
    n = sum(int(np.load(config.PROC_DIR / f"{name}_desc.npy", mmap_mode="r").shape[0])
            for name in shard_names)

    # --- descriptor normalization stats (streaming mean/std) ---------------
    # Skipped for continued pre-training (recompute_stats=False) so an already
    # trained model's conditioning calibration stays consistent.
    if recompute_stats or not config.DESC_STATS_PATH.exists():
        s = np.zeros(config.N_PROPS, np.float64)
        ss = np.zeros(config.N_PROPS, np.float64)
        for name in shard_names:
            d = np.load(config.PROC_DIR / f"{name}_desc.npy").astype(np.float64)
            s += d.sum(0)
            ss += (d * d).sum(0)
        mean = (s / max(n, 1)).astype(np.float32)
        var = np.maximum(ss / max(n, 1) - (s / max(n, 1)) ** 2, 1e-8)
        std = np.sqrt(var).astype(np.float32)
        data.save_stats(mean, std)
    else:
        print("  (keeping existing descriptor_stats.json)")

    # --- membership: SQLite (exact) + Bloom (fast pre-filter) --------------
    if config.MOLPORT_DB.exists():
        config.MOLPORT_DB.unlink()
    conn = membership.open_db(write=True)
    bloom = membership.BloomFilter(max(bloom_capacity, n), fp_rate)
    for name in shard_names:
        rows = []
        with open(config.PROC_DIR / f"{name}_ids.txt", "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                canon, _, mid = line.rstrip("\n").partition("\t")
                bloom.add(canon)
                rows.append((canon, mid))
        membership.add_many(conn, rows)
        conn.commit()
    conn.close()
    bloom.save(config.MOLPORT_BLOOM)

    # --- global meta --------------------------------------------------------
    meta = {
        "shards": shard_names,
        "total": int(n),
        "max_len": config.MAX_SELFIES_LEN,
        "vocab_size": len(data.Vocab.load()),
        "properties": config.PROPERTIES,
        "constraints": data.CONSTRAINTS,
    }
    with open(config.META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. {n:,} molecules across {len(shard_names)} shards.")
    print(f"  vocab={meta['vocab_size']}  bloom~{bloom.m_bits//8/1e6:.1f}MB  db={config.MOLPORT_DB}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max molecules to KEEP (0 = all)")
    ap.add_argument("--shards", type=int, nargs="*", help="input-file indices to process")
    ap.add_argument("--workers", type=int, default=config.N_WORKERS)
    ap.add_argument("--chunksize", type=int, default=200)
    ap.add_argument("--force", action="store_true", help="reprocess shards even if present")
    ap.add_argument("--bloom-capacity", type=int, default=7_000_000)
    ap.add_argument("--bloom-fp", type=float, default=1e-4)
    args = ap.parse_args()

    config.ensure_dirs()

    # Fixed vocab from the SELFIES robust alphabet.
    vocab = data.Vocab.build()
    vocab.save()
    print(f"Vocab: {len(vocab)} tokens ({data.CONSTRAINTS} constraints) -> {config.VOCAB_PATH}")

    in_files = sorted(config.SMILES_DIR.glob(config.SMILES_GLOB))
    if not in_files:
        raise SystemExit(f"No SMILES files found in {config.SMILES_DIR}")
    if args.shards is not None:
        in_files = [in_files[i] for i in args.shards]

    shard_names: List[str] = []
    kept_total = 0
    for idx, in_path in enumerate(in_files):
        shard_name = f"shard_{idx:03d}"
        base = config.PROC_DIR / shard_name
        if _shard_done(base) and not args.force:
            n = int(json.load(open(f"{base}_meta.json"))["kept"])
            print(f"{shard_name}: already done ({n:,} molecules) — skipping")
            shard_names.append(shard_name)
            kept_total += n
            continue
        remaining = (args.limit - kept_total) if args.limit else 0
        if args.limit and remaining <= 0:
            break
        kept = process_file(_iter_lines(in_path), shard_name, in_path.name,
                            workers=args.workers, chunksize=args.chunksize, remaining=remaining)
        if kept > 0:
            shard_names.append(shard_name)
            kept_total += kept
        if args.limit and kept_total >= args.limit:
            break

    finalize(shard_names, args.bloom_capacity, args.bloom_fp)


if __name__ == "__main__":
    main()
