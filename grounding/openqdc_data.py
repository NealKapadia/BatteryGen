"""Optional 3D / quantum dataset loaders (OpenQDC GEOM, atomic-datasets).

These pull large 3D molecular datasets with quantum labels. They are OPTIONAL and
heavy; install separately (may change torch/numpy, so do it when NOT mid-training):

    pip install openqdc atomic-datasets

This writes:
  * a SMILES file (for continued pre-training via add_data.py), and
  * a labels CSV (SMILES + energy) for grounding via finetune_dft.py, when energies
    are available.

  python -m batterygen.grounding.openqdc_data --dataset GEOM --max 200000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from batterygen.core import config



def _smiles_of(entry):
    """Best-effort SMILES extraction across openqdc versions."""
    for attr in ("smiles", "name", "canonical_smiles"):
        v = entry.get(attr) if isinstance(entry, dict) else getattr(entry, attr, None)
        if isinstance(v, str) and any(c.isalpha() for c in v) and "/" not in v[:2]:
            return v
    return None


def load_openqdc(dataset: str, max_n: int):
    import importlib

    mod = importlib.import_module("openqdc.datasets")
    cls = getattr(mod, dataset)
    print(f"Loading OpenQDC {dataset} (downloads once; large) ...")
    ds = cls()
    smis, labels = [], []
    for i in range(len(ds)):
        if max_n and i >= max_n:
            break
        e = ds[i]
        smi = _smiles_of(e)
        if not smi:
            continue
        smis.append(smi)
        en = e.get("energies") if isinstance(e, dict) else getattr(e, "energies", None)
        try:
            labels.append((smi, float(en[0]) if hasattr(en, "__len__") else float(en)))
        except (TypeError, ValueError, IndexError):
            pass
    return smis, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GEOM", help="OpenQDC dataset class, e.g. GEOM")
    ap.add_argument("--max", type=int, default=200000)
    ap.add_argument("--smiles-out", default=str(config.DFT_DIR / "openqdc_smiles.txt"))
    ap.add_argument("--labels-out", default=str(config.DFT_DIR / "openqdc_labels.csv"))
    args = ap.parse_args()
    config.ensure_dirs()

    try:
        smis, labels = load_openqdc(args.dataset, args.max)
    except ImportError:
        raise SystemExit("openqdc not installed. Run: pip install openqdc  (then retry). "
                         "Skipping is fine — QM9 already grounds the property head.")
    except Exception as e:
        raise SystemExit(f"OpenQDC load failed ({e}). API may differ by version; "
                         "skip this step (QM9 grounding is sufficient).")

    Path(args.smiles_out).write_text("\n".join(smis), encoding="utf-8")
    print(f"Wrote {len(smis):,} SMILES -> {args.smiles_out} (add via add_data.py --tag openqdc)")
    if labels:
        with open(args.labels_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["smiles", "energy"]); w.writerows(labels)
        print(f"Wrote {len(labels):,} energy labels -> {args.labels_out}")


if __name__ == "__main__":
    main()
