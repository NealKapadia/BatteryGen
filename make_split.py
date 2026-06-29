"""Build a scaffold-based train/val split.

Random splits let a model look good by *memorizing* close analogues. A scaffold
split holds out whole Bemis-Murcko scaffolds, so the validation molecules share no
ring core with anything in training — the honest test of whether the model learned
chemistry or just memorized. Run this BEFORE training so the 18 h model is validated
on genuinely unseen scaffolds.

Writes split_val.npy (bool, one per molecule, in shard order) which MolDataset uses
automatically when present.

  python molvae/make_split.py --val-frac 0.02
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
from tqdm import tqdm

import config


def _init():
    global _Chem, _Murcko
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    _Chem, _Murcko = Chem, MurckoScaffold


def _scaffold_key(smi: str):
    m = _Chem.MolFromSmiles(smi)
    if m is None:
        return None
    scaf = _Murcko.GetScaffoldForMol(m)
    # acyclic molecules have no Murcko scaffold -> use the molecule itself as its key
    return _Chem.MolToSmiles(scaf if scaf.GetNumAtoms() > 0 else m)


def _bucket(smi: str, nbuckets: int = 1000) -> int:
    key = _scaffold_key(smi)
    if key is None:
        return nbuckets - 1  # unparseable -> a non-val bucket (train)
    h = int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little")
    return h % nbuckets


_NBUCKETS = 1000
_VAL_BUCKETS = 20  # set in main


def _worker(smi: str) -> bool:
    return _bucket(smi, _NBUCKETS) < _VAL_BUCKETS


def main():
    import multiprocessing as mp

    global _VAL_BUCKETS
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.02, help="fraction of SCAFFOLDS held out")
    ap.add_argument("--workers", type=int, default=config.N_WORKERS)
    ap.add_argument("--chunksize", type=int, default=500)
    args = ap.parse_args()
    _VAL_BUCKETS = max(1, round(_NBUCKETS * args.val_frac))

    meta = json.load(open(config.META_PATH, encoding="utf-8"))
    masks = []
    n_val = n_tot = 0
    for shard in meta["shards"]:
        with open(config.PROC_DIR / f"{shard}_ids.txt", encoding="utf-8") as f:
            smiles = [ln.partition("\t")[0] for ln in f.read().splitlines() if ln.strip()]
        shard_mask = np.zeros(len(smiles), dtype=bool)
        with mp.Pool(args.workers, initializer=_init) as pool:
            for i, is_val in enumerate(tqdm(
                pool.imap(_worker, smiles, chunksize=args.chunksize),
                total=len(smiles), desc=shard, unit="mol", dynamic_ncols=True)):
                shard_mask[i] = is_val
        masks.append(shard_mask)
        n_val += int(shard_mask.sum())
        n_tot += len(shard_mask)

    split = np.concatenate(masks)
    np.save(config.PROC_DIR / "split_val.npy", split)
    with open(config.PROC_DIR / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump({"type": "scaffold", "val_frac": args.val_frac,
                   "n_val": n_val, "n_total": n_tot}, f, indent=2)
    print(f"\nScaffold split: {n_val:,} val / {n_tot:,} total ({n_val/max(n_tot,1):.2%}) "
          f"-> {config.PROC_DIR / 'split_val.npy'}")
    print("MolDataset will now use this held-out-scaffold split automatically.")


if __name__ == "__main__":
    main()
