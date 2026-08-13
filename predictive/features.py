"""
predictive/features.py  (pipeline step 1)
=========================================
Builds the feature cache the predictive model trains on, from your dataset CSV
(located via config.resolve_ce_csv: --csv, $BATTERYGEN_CE_CSV, or a single file in data/).
The schema is read entirely from ``TARGET`` (predictive/target.py) - no column name is
hardcoded here, so the same code serves any chemistry / any measured target.

Cache contents:
  X            : DataFrame [RDKit(19) + xTB(7, optional) + TARGET.context_cols]
  y            : the TARGET.target_col value
  cov          : per-row coefficient of variation of the replicate columns (%), for the
                 noise filter (0 for every row when TARGET.replicate_cols is empty)
  groups       : Bemis-Murcko scaffold (acyclic -> own SMILES) for scaffold GroupKFold
  feat_med     : per-feature median (impute missing xTB at inference)
  train_fps    : Morgan fingerprints for the domain-similarity readout
  feature_order: the exact column order of X

xTB (GFN2 single-point) gives HOMO/LUMO/gap/dipole per UNIQUE molecule; chi/eta/omega are
derived. Results are cached to predictive/xtb_cache.csv so re-runs are instant. Charged
fragments are run at their RDKit formal charge.

Run:  python -m batterygen.predictive.features [--no-xtb]
Output: <artifacts>/predictive/feature_cache.pkl
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from batterygen.core import config
from batterygen.core import data
from batterygen.predictive.target import TARGET
from batterygen.predictive import paths

RDKIT_COLS = ["MolLogP", "TPSA", "HDonor", "HAccept", "RotB", "FracCSP3", "MolWt",
              "NHOH", "NOCount", "BertzCT", "MolMR", "MaxAbsQ", "MinQ", "ArRings",
              "AliRings", "LabuteASA", "QED", "nN", "nO"]
XTB_COLS = ["xtb_HOMO", "xtb_LUMO", "xtb_gap", "xtb_chi", "xtb_eta", "xtb_omega", "xtb_dipole"]


def feature_order(use_xtb: bool) -> list:
    """The full candidate-feature column order for the current TARGET."""
    cols = list(RDKIT_COLS)
    if use_xtb:
        cols += XTB_COLS
    cols += list(TARGET.context_cols)
    return cols


# --------------------------------------------------------------------------- #
def rdkit_features(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import (AllChem, Crippen, Descriptors, Lipinski, QED,
                            rdMolDescriptors, GraphDescriptors)

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        AllChem.ComputeGasteigerCharges(m)
        q = [a.GetDoubleProp("_GasteigerCharge") for a in m.GetAtoms()]
        q = [v for v in q if np.isfinite(v)]
        max_abs_q = float(max(abs(v) for v in q)) if q else 0.0
        min_q = float(min(q)) if q else 0.0
    except Exception:
        max_abs_q = min_q = 0.0
    f = {
        "MolLogP": Crippen.MolLogP(m), "TPSA": rdMolDescriptors.CalcTPSA(m),
        "HDonor": Lipinski.NumHDonors(m), "HAccept": Lipinski.NumHAcceptors(m),
        "RotB": Descriptors.NumRotatableBonds(m), "FracCSP3": rdMolDescriptors.CalcFractionCSP3(m),
        "MolWt": Descriptors.MolWt(m), "NHOH": Lipinski.NHOHCount(m), "NOCount": Lipinski.NOCount(m),
        "BertzCT": GraphDescriptors.BertzCT(m), "MolMR": Crippen.MolMR(m),
        "MaxAbsQ": max_abs_q, "MinQ": min_q,
        "ArRings": rdMolDescriptors.CalcNumAromaticRings(m),
        "AliRings": rdMolDescriptors.CalcNumAliphaticRings(m),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(m), "QED": QED.qed(m),
        "nN": sum(a.GetSymbol() == "N" for a in m.GetAtoms()),
        "nO": sum(a.GetSymbol() == "O" for a in m.GetAtoms()),
    }
    return {k: float(v) for k, v in f.items()}


def scaffold_of(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(smiles)
        return s if s else smiles  # acyclic -> group by the molecule itself
    except Exception:
        return smiles


def load_xtb_cache() -> dict:
    cache = {}
    if paths.XTB_CACHE.exists():
        with open(paths.XTB_CACHE, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cache[r["smiles"]] = {k: (float(r[k]) if r[k] not in ("", "None") else None)
                                      for k in ("homo", "lumo", "gap", "dipole")}
    return cache


def compute_xtb(smiles_list, cache: dict):
    """Fill xTB HOMO/LUMO/gap/dipole for any uncached molecule; persist incrementally."""
    from rdkit import Chem
    from batterygen.grounding import xtb_label

    todo = [s for s in smiles_list if s not in cache]
    if not todo:
        return cache
    print(f"  xTB: {len(todo)} new molecules ({len(cache)} cached) ...")
    env = dict(os.environ); env.setdefault("OMP_NUM_THREADS", "4")
    scratch = Path(tempfile.mkdtemp(prefix="xtb_pred_"))
    write_header = not paths.XTB_CACHE.exists()
    paths.PRED_DIR.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm
    with open(paths.XTB_CACHE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["smiles", "homo", "lumo", "gap", "dipole"])
        if write_header:
            w.writeheader()
        for smi in tqdm(todo, unit="mol", dynamic_ncols=True):
            chrg = 0
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                chrg = Chem.GetFormalCharge(m)
            res = xtb_label.run_xtb(smi, scratch, opt=False, charge=chrg, timeout=120, env=env)
            rec = {"homo": None, "lumo": None, "gap": None, "dipole": None}
            if res is not None:
                rec = {k: res.get(k) for k in ("homo", "lumo", "gap", "dipole")}
            cache[smi] = rec
            w.writerow({"smiles": smi, **rec}); f.flush()
    return cache


def xtb_derived(rec: dict) -> dict:
    """HOMO/LUMO/gap/dipole -> the 7 xTB feature columns (chi/eta/omega derived; eV)."""
    homo, lumo = rec.get("homo"), rec.get("lumo")
    gap, dip = rec.get("gap"), rec.get("dipole")
    out = {c: np.nan for c in XTB_COLS}
    if homo is not None and lumo is not None:
        chi = -(homo + lumo) / 2.0          # electronegativity
        eta = (lumo - homo) / 2.0           # chemical hardness
        omega = (chi * chi) / (2.0 * eta) if eta and abs(eta) > 1e-6 else np.nan  # electrophilicity
        out.update(xtb_HOMO=homo, xtb_LUMO=lumo, xtb_chi=chi, xtb_eta=eta, xtb_omega=omega)
    if gap is not None:
        out["xtb_gap"] = gap
    if dip is not None:
        out["xtb_dipole"] = dip
    return out


def _row_cov(row: dict) -> float:
    """Coefficient of variation (%) of the replicate columns; 0 when no replicates configured."""
    if not TARGET.replicate_cols:
        return 0.0
    try:
        trip = [float(row[c]) for c in TARGET.replicate_cols]
    except (KeyError, ValueError):
        return 999.0
    mean = np.mean(trip)
    return float(np.std(trip) / mean * 100.0) if mean else 999.0


def _append_measurement(csv_path: Path, smiles, value, context_overrides: dict, name):
    """Append one measured row (active learning). Replicate columns are set to `value`;
    context columns use the provided override or the column median."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    row = {TARGET.smiles_col: smiles, "IUPAC_NAME": name, TARGET.target_col: value}
    for c in TARGET.replicate_cols:
        row[c] = value
    for c in TARGET.context_cols:
        row[c] = float(context_overrides[c]) if c in context_overrides else float(df[c].median())
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[active-learning] appended {smiles} ({TARGET.target_name}={value}) -> {csv_path} "
          f"(now {len(df)} rows). Rebuilding cache; then run select + train.")


def main():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="dataset CSV (default: auto-detect a single CSV in data/, or $BATTERYGEN_CE_CSV)")
    ap.add_argument("--no-xtb", dest="xtb", action="store_false", default=TARGET.use_xtb,
                    help="skip xTB (xtb cols dropped from the feature set)")
    ap.add_argument("--xtb", dest="xtb", action="store_true", help="force xTB on")
    # active learning: append a measured (SMILES, value) row, then rebuild the cache
    ap.add_argument("--append-smiles", help="active learning: add this measured molecule")
    ap.add_argument("--append-value", type=float, help=f"its measured {TARGET.target_name}")
    ap.add_argument("--append-context", default="",
                    help="context overrides as 'col=val,col=val' (default: column medians)")
    ap.add_argument("--append-name", default="user_measured")
    args = ap.parse_args()
    args.csv = config.resolve_ce_csv(args.csv)
    use_xtb = args.xtb
    paths.PRED_DIR.mkdir(parents=True, exist_ok=True)

    if args.append_smiles:
        if args.append_value is None:
            raise SystemExit("--append-smiles requires --append-value")
        overrides = {}
        for kv in args.append_context.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1); overrides[k.strip()] = v.strip()
        _append_measurement(Path(args.csv), args.append_smiles, args.append_value, overrides, args.append_name)
        # fall through: rebuild the cache so the new row is included

    order = feature_order(use_xtb)
    rows = []
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            canon = data.canonical_smiles((r.get(TARGET.smiles_col) or "").strip())
            if not canon:
                continue
            try:
                y = float(r[TARGET.target_col])
                context = {c: float(r[c]) for c in TARGET.context_cols}
            except (KeyError, ValueError):
                continue
            rows.append({"smiles": canon, "y": y, "context": context, "cov": _row_cov(r)})
    print(f"Loaded {len(rows)} rows | {len(set(r['smiles'] for r in rows))} unique molecules "
          f"| target='{TARGET.target_col}' | xTB={'on' if use_xtb else 'off'}")

    uniq = list(dict.fromkeys(r["smiles"] for r in rows))
    rd = {s: rdkit_features(s) for s in uniq}
    rd = {s: v for s, v in rd.items() if v is not None}

    xd = {}
    if use_xtb:
        xc = compute_xtb(uniq, load_xtb_cache())
        xd = {s: xtb_derived(xc.get(s, {})) for s in uniq}

    X, y, cov, groups, kept_smiles = [], [], [], [], []
    for r in rows:
        s = r["smiles"]
        if s not in rd:
            continue
        feat = dict(rd[s])
        if use_xtb:
            feat.update(xd.get(s, {c: np.nan for c in XTB_COLS}))
        feat.update(r["context"])
        X.append([feat.get(c, np.nan) for c in order])
        y.append(r["y"]); cov.append(r["cov"]); groups.append(scaffold_of(s)); kept_smiles.append(s)

    Xdf = pd.DataFrame(X, columns=order)
    feat_med = Xdf.median(numeric_only=True)
    Xdf = Xdf.fillna(feat_med)                       # impute failed xTB with medians
    if use_xtb:
        n_xtb_ok = int(np.isfinite([xd[s]["xtb_HOMO"] for s in uniq]).sum())
        print(f"Feature matrix {Xdf.shape} | xTB ok for {n_xtb_ok}/{len(uniq)} unique mols")
    else:
        print(f"Feature matrix {Xdf.shape}")

    # Morgan fps for the domain-similarity readout
    fps = []
    for s in kept_smiles:
        m = Chem.MolFromSmiles(s)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None)

    joblib.dump(dict(X=Xdf, y=np.asarray(y, float), cov=np.asarray(cov, float),
                     groups=np.asarray(groups, object), feat_med=feat_med,
                     train_fps=fps, train_smiles=kept_smiles, feature_order=order,
                     use_xtb=use_xtb),
                paths.FEATURE_CACHE)
    keep = np.asarray(cov) <= TARGET.max_cov
    print(f"CoV<={TARGET.max_cov} keeps {keep.sum()}/{len(cov)} rows")
    print(f"Saved -> {paths.FEATURE_CACHE}")


if __name__ == "__main__":
    main()
