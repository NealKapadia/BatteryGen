"""Normalize the electrolyte datasets into ONE training table the property model uses.

* CALiSol-23: wide format — salt name + one fraction column per solvent. We map salt
  -> (cation, anion SMILES) and each nonzero solvent column -> SMILES (solvent_lib).
* OEDB-electrolytes: already has Cation, Anion SMILES, Solvent SMILES, concentration,
  ionic conductivity, viscosity, density, and Cation←Anion / Cation←Solvent COORDINATION
  numbers (the "how they coordinate" signal).

Output: electrolyte_train.csv with columns
  mix (smi:frac;smi:frac), cation, anion_smiles, conc, temp,
  conductivity, viscosity, coord_cat_anion, coord_cat_solvent, density, source

  python molvae/electrolyte_data.py        # uses the files in the data dir
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import config
import solvent_lib as S

OUT_DEFAULT = config.ART_DIR / "electrolyte_train.csv"
FIELDS = ["mix", "cation", "anion_smiles", "conc", "temp", "conductivity",
          "viscosity", "coord_cat_anion", "coord_cat_solvent", "density", "source"]

_CALISOL_META = {"", "doi", "k", "T", "c", "salt", "c units", "solvent ratio type"}


def _find_key(keys, *tokens) -> Optional[str]:
    """First key containing all lowercase tokens (robust to spacing/unicode)."""
    for k in keys:
        kl = k.lower()
        if all(t in kl for t in tokens):
            return k
    return None


def _fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return ""


def convert_calisol(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {h: i for i, h in enumerate(header)}
        solvent_cols = [h for h in header if h not in _CALISOL_META and h in S.SOLVENT_SMILES]
        for p in r:
            if len(p) < len(header):
                continue
            try:
                k, T, c = float(p[idx["k"]]), float(p[idx["T"]]), float(p[idx["c"]])
            except (ValueError, KeyError):
                continue
            cation, anion = S.parse_salt(p[idx["salt"]])
            mix = []
            for col in solvent_cols:
                frac = _fnum(p[idx[col]])
                if frac and frac > 0:
                    mix.append(f"{S.SOLVENT_SMILES[col]}:{frac}")
            if not mix:
                continue
            rows.append({"mix": ";".join(mix), "cation": cation or "Li",
                         "anion_smiles": anion or "", "conc": c, "temp": T,
                         "conductivity": k, "source": "calisol23"})
    return rows


def convert_oedb(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        keys = reader.fieldnames or []
        k_solv = _find_key(keys, "solvent", "smiles")
        k_anion = _find_key(keys, "anion", "smiles")
        k_cat = _find_key(keys, "cation") or "Cation"
        k_conc = _find_key(keys, "concentration")
        k_cond = _find_key(keys, "conductivity")
        k_visc = _find_key(keys, "viscosity")
        k_ca = _find_key(keys, "coordination", "anion")
        k_cs = _find_key(keys, "coordination", "solvent")
        k_dens = _find_key(keys, "density")
        for d in reader:
            solv = (d.get(k_solv) or "").strip() if k_solv else ""
            if not solv:
                continue
            mix = ";".join(f"{s}:1" for s in solv.split(".") if s)
            rows.append({
                "mix": mix, "cation": (d.get(k_cat) or "Li").strip(),
                "anion_smiles": (d.get(k_anion) or "").strip() if k_anion else "",
                "conc": _fnum(d.get(k_conc)) if k_conc else "", "temp": 298.15,
                "conductivity": _fnum(d.get(k_cond)) if k_cond else "",
                "viscosity": _fnum(d.get(k_visc)) if k_visc else "",
                "coord_cat_anion": _fnum(d.get(k_ca)) if k_ca else "",
                "coord_cat_solvent": _fnum(d.get(k_cs)) if k_cs else "",
                "density": _fnum(d.get(k_dens)) if k_dens else "", "source": "oedb"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calisol", default=str(config.DATA_DIR / "calisol23_DOI_10.11583DTU.c.6929599.csv"))
    ap.add_argument("--oedb", default=str(config.DATA_DIR / "oedb-electrolytes-v2026-05-11.csv"))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    config.ensure_dirs()

    rows = []
    for name, path, fn in [("CALiSol-23", args.calisol, convert_calisol),
                           ("OEDB-electrolytes", args.oedb, convert_oedb)]:
        p = Path(path)
        if p.exists():
            r = fn(p)
            rows += r
            print(f"  {name}: {len(r)} formulations")
        else:
            print(f"  {name}: not found at {p} (skipped)")

    if not rows:
        raise SystemExit("No electrolyte rows — check the input paths.")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    cations = sorted({r["cation"] for r in rows})
    print(f"\nWrote {len(rows):,} formulations -> {args.out}")
    print(f"Cations present: {cations}")
    print("Note: conductivity assumed mS/cm for both sources; verify CALiSol-23 units if needed.")
    print("Next: python molvae/electrolyte.py --mode train --csv " + args.out +
          " --mix-col mix --cation-col cation --anion-smiles-col anion_smiles "
          "--conc-col conc --temp-col temp --target-cols conductivity --log-target")


if __name__ == "__main__":
    main()
