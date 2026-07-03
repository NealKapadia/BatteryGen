"""HuggingFace PubChem-124M loader (streaming).

Dataset: hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC  (124M molecules).
Two uses:
  * --mode sample : write N SMILES for continued pre-training (add_data.py).
  * --mode bloom  : build a PubChem Bloom filter (canonicalized) for the
                    novelty-vs-PubChem benchmark metric (evaluate.py).

Requires `pip install datasets` (safe; pure-python + pyarrow). Streaming avoids
downloading all 124M at once. Canonicalization is parallelized across CPU cores.

  pip install datasets
  python molvae/hf_data.py --mode sample --max 2000000
  python molvae/hf_data.py --mode bloom  --max 20000000
"""
from __future__ import annotations

import argparse

from tqdm import tqdm

from molforge.core import config


HF_NAME = "hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC"


def _stream_smiles(max_n: int):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install first:  pip install datasets")
    ds = load_dataset(HF_NAME, split="train", streaming=True)
    n = 0
    for row in ds:
        smi = next((row[k] for k in row if "smiles" in k.lower() and row[k]), None)
        if smi:
            yield smi
            n += 1
            if max_n and n >= max_n:
                break


def _canon(smi):
    from molforge.core import data  # rdkit loaded on import

    return data.canonical_smiles(smi)


def sample(max_n: int, out: str):
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for smi in tqdm(_stream_smiles(max_n), total=max_n or None, desc="PubChem", unit="mol"):
            f.write(smi + "\n"); n += 1
    print(f"Wrote {n:,} SMILES -> {out}\nAdd to training:  python molvae/add_data.py "
          f"--input {out} --tag pubchem --dedup")


def build_bloom(max_n: int, out: str, workers: int, capacity: int):
    import multiprocessing as mp
    from molforge.core.membership import BloomFilter

    bloom = BloomFilter(max(capacity, max_n or capacity), 1e-4)
    n = 0
    with mp.Pool(workers) as pool:
        for canon in tqdm(pool.imap(_canon, _stream_smiles(max_n), chunksize=500),
                          total=max_n or None, desc="PubChem bloom", unit="mol"):
            if canon:
                bloom.add(canon); n += 1
    bloom.save(out)
    print(f"Built PubChem Bloom ({n:,} molecules, ~{bloom.m_bits//8/1e6:.0f} MB) -> {out}")
    print("evaluate.py auto-detects it for novelty-vs-PubChem.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sample", "bloom"], default="sample")
    ap.add_argument("--max", type=int, default=2_000_000, help="molecules to read (0 = all 124M)")
    ap.add_argument("--workers", type=int, default=config.N_WORKERS)
    ap.add_argument("--capacity", type=int, default=30_000_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    config.ensure_dirs()

    if args.mode == "sample":
        sample(args.max, args.out or str(config.DFT_DIR / "pubchem_smiles.txt"))
    else:
        build_bloom(args.max, args.out or str(config.MOLPORT_BLOOM.parent / "pubchem.bloom"),
                    args.workers, args.capacity)


if __name__ == "__main__":
    main()
