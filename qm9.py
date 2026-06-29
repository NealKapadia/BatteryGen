"""Fetch QM9 (134k small molecules with real DFT labels) and emit a labels CSV
that finetune_dft.py consumes directly.

QM9 is the gold-standard quantum-chemistry benchmark: B3LYP/6-31G(2df,p) DFT
properties for ~134k molecules. We export the chemically useful ones so the VAE's
latent gets grounded in *real* DFT, far better than xTB-on-a-subset.

  python molvae/qm9.py                      # download via PyG, write qm9_labels.csv
  python molvae/qm9.py --max 50000          # subset
then:
  python molvae/finetune_dft.py --target homo,lumo,gap,dipole --labels molvae_artifacts/dft/qm9_labels.csv

Units follow PyG's QM9 (orbital energies in eV, dipole in Debye). If the PyG
download fails, point --raw-csv at a gdb9 properties CSV (with a 'smiles' column).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import config

# PyG QM9 y-column order -> our names (first 12 are the chemically relevant ones).
_PYG_COLS = {0: "dipole", 1: "alpha", 2: "homo", 3: "lumo", 4: "gap",
             5: "r2", 6: "zpve", 7: "u0", 8: "u", 9: "h", 10: "g", 11: "cv"}
_EXPORT = ["dipole", "homo", "lumo", "gap", "zpve", "cv", "u0"]


def from_pyg(max_n: int):
    from torch_geometric.datasets import QM9

    root = str(config.ART_DIR / "qm9_raw")
    print(f"Loading QM9 via PyG into {root} (downloads ~ once) ...")
    ds = QM9(root=root)
    rows = []
    for i, d in enumerate(ds):
        if max_n and i >= max_n:
            break
        smi = getattr(d, "smiles", None)
        if not smi:
            continue
        y = d.y.view(-1).tolist()
        rec = {"smiles": smi}
        for col, name in _PYG_COLS.items():
            if name in _EXPORT and col < len(y):
                rec[name] = y[col]
        rows.append(rec)
    return rows


def from_raw_csv(path: Path, max_n: int):
    """Fallback: a CSV with a 'smiles' column + property columns named like ours."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if max_n and i >= max_n:
                break
            smi = r.get("smiles") or r.get("SMILES")
            if not smi:
                continue
            rec = {"smiles": smi}
            for name in _EXPORT:
                if name in r:
                    try:
                        rec[name] = float(r[name])
                    except ValueError:
                        pass
            rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0, help="cap number of molecules (0 = all)")
    ap.add_argument("--raw-csv", type=str, default=None, help="use this CSV instead of PyG download")
    ap.add_argument("--out", type=str, default=str(config.DFT_DIR / "qm9_labels.csv"))
    args = ap.parse_args()
    config.ensure_dirs()

    if args.raw_csv:
        rows = from_raw_csv(Path(args.raw_csv), args.max)
    else:
        try:
            rows = from_pyg(args.max)
        except Exception as e:
            raise SystemExit(
                f"PyG QM9 load failed ({e}).\nDownload a gdb9 properties CSV (with a "
                f"'smiles' column) and rerun with --raw-csv <path>.")

    if not rows:
        raise SystemExit("No QM9 rows extracted.")
    fields = ["smiles"] + [c for c in _EXPORT if any(c in r for r in rows)]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows):,} QM9 labels -> {args.out}")
    print(f"Columns: {fields}")
    print("Next: python molvae/finetune_dft.py --target homo,lumo,gap,dipole "
          f"--labels {args.out}")


if __name__ == "__main__":
    main()
