"""
ce_features.py  (step 1/2 of the CE workflow)
=============================================
Builds the feature cache the CE model trains on, from your CE dataset CSV
(located via config.resolve_ce_csv: --csv, $MOLVAE_CE_CSV, or a single file in data/):

  X        : DataFrame [RDKit(19) + xTB(7) + LMR]  (FEATURE_ORDER)
  y        : CE_aver (%)
  cov      : per-row coefficient of variation of the CE_1/2/3 triplicate (%) -> noise filter
  groups   : Bemis-Murcko scaffold (acyclic -> own SMILES) for scaffold GroupKFold
  feat_med : per-feature median (impute missing xTB at inference)
  train_fps/train_smiles : Morgan fps for the domain-similarity readout

xTB (GFN2 single-point) gives HOMO/LUMO/gap/dipole per UNIQUE additive; chi/eta/omega are
derived. Results are cached to ce/xtb_cache.csv so re-runs are instant. Charged fragments
(e.g. acetate from a Na-salt) are run at their RDKit formal charge.

Run:  python molvae/ce_features.py [--no-xtb]   (--no-xtb fills xTB cols with NaN -> imputed)
Output: molvae_artifacts/ce/feature_cache.pkl
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

import config
import data

CE_DIR = config.ARTIFACTS / "ce" if hasattr(config, "ARTIFACTS") else config.CKPT_DIR.parent / "ce"
CACHE = CE_DIR / "feature_cache.pkl"
XTB_CACHE = CE_DIR / "xtb_cache.csv"

RDKIT_COLS = ["MolLogP", "TPSA", "HDonor", "HAccept", "RotB", "FracCSP3", "MolWt",
              "NHOH", "NOCount", "BertzCT", "MolMR", "MaxAbsQ", "MinQ", "ArRings",
              "AliRings", "LabuteASA", "QED", "nN", "nO"]
XTB_COLS = ["xtb_HOMO", "xtb_LUMO", "xtb_gap", "xtb_chi", "xtb_eta", "xtb_omega", "xtb_dipole"]
FEATURE_ORDER = RDKIT_COLS + XTB_COLS + ["LMR"]

SMILES_COL, TARGET_COL = "Additive_SMILES", "CE_aver. (%)"
TRIP_COLS = ["CE_1 (%)", "CE_2 (%)", "CE_3 (%)"]


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
    if XTB_CACHE.exists():
        with open(XTB_CACHE, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cache[r["smiles"]] = {k: (float(r[k]) if r[k] not in ("", "None") else None)
                                      for k in ("homo", "lumo", "gap", "dipole")}
    return cache


def compute_xtb(smiles_list, cache: dict):
    """Fill xTB HOMO/LUMO/gap/dipole for any uncached molecule; persist incrementally."""
    from rdkit import Chem
    import xtb_label

    todo = [s for s in smiles_list if s not in cache]
    if not todo:
        return cache
    print(f"  xTB: {len(todo)} new molecules ({len(cache)} cached) ...")
    env = dict(os.environ); env.setdefault("OMP_NUM_THREADS", "4")
    scratch = Path(tempfile.mkdtemp(prefix="xtb_ce_"))
    write_header = not XTB_CACHE.exists()
    from tqdm import tqdm
    with open(XTB_CACHE, "a", newline="", encoding="utf-8") as f:
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


def _append_measurement(csv_path: Path, smiles, ce, lmr, zn, addmole, name):
    """Append one measured row to the CE CSV (active learning). Triplicate cols = CE."""
    import pandas as pd
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    def med(col, override):
        return float(override) if override is not None else float(df[col].median())
    row = {
        "#": len(df) + 1,
        "Zn_mole (mmol)": med("Zn_mole (mmol)", zn),
        "Additive_mole (%)": med("Additive_mole (%)", addmole),
        "LogMolarRatio": med("LogMolarRatio", lmr),
        "Additive_SMILES": smiles, "IUPAC_NAME": name,
        "CE_1 (%)": ce, "CE_2 (%)": ce, "CE_3 (%)": ce, "CE_aver. (%)": ce,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[active-learning] appended {smiles} (CE {ce}%) -> {csv_path} "
          f"(now {len(df)} rows). Rebuilding cache; then run ce_train.")


def main():
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="CE dataset CSV (default: auto-detect a single CSV in data/, "
                         "or set $MOLVAE_CE_CSV)")
    ap.add_argument("--no-xtb", action="store_true", help="skip xTB (xtb cols = NaN -> imputed)")
    # active learning: append a measured (SMILES, CE) row, then rebuild the cache
    ap.add_argument("--append-smiles", help="active learning: add this measured additive")
    ap.add_argument("--append-ce", type=float, help="its measured CE_aver (%%)")
    ap.add_argument("--append-lmr", type=float, default=None, help="its LogMolarRatio (default median)")
    ap.add_argument("--append-zn", type=float, default=None)
    ap.add_argument("--append-addmole", type=float, default=None)
    ap.add_argument("--append-name", default="user_measured")
    args = ap.parse_args()
    args.csv = config.resolve_ce_csv(args.csv)
    CE_DIR.mkdir(parents=True, exist_ok=True)

    if args.append_smiles:
        if args.append_ce is None:
            raise SystemExit("--append-smiles requires --append-ce")
        _append_measurement(Path(args.csv), args.append_smiles, args.append_ce,
                            args.append_lmr, args.append_zn, args.append_addmole, args.append_name)
        # fall through: rebuild the cache so the new row is included (xTB computed + cached)

    rows = []
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            canon = data.canonical_smiles((r.get(SMILES_COL) or "").strip())
            if not canon:
                continue
            try:
                trip = [float(r[c]) for c in TRIP_COLS]
                y = float(r[TARGET_COL])
                lmr = float(r["LogMolarRatio"])
            except (KeyError, ValueError):
                continue
            mean = np.mean(trip)
            cov = float(np.std(trip) / mean * 100.0) if mean else 999.0
            rows.append({"smiles": canon, "y": y, "lmr": lmr, "cov": cov})
    print(f"Loaded {len(rows)} rows | {len(set(r['smiles'] for r in rows))} unique additives")

    uniq = list(dict.fromkeys(r["smiles"] for r in rows))
    rd = {s: rdkit_features(s) for s in uniq}
    rd = {s: v for s, v in rd.items() if v is not None}

    xc = {} if args.no_xtb else load_xtb_cache()
    if not args.no_xtb:
        xc = compute_xtb(uniq, xc)
    xd = {s: xtb_derived(xc.get(s, {})) for s in uniq}

    X, y, cov, groups, kept_smiles = [], [], [], [], []
    for r in rows:
        s = r["smiles"]
        if s not in rd:
            continue
        feat = dict(rd[s]); feat.update(xd.get(s, {c: np.nan for c in XTB_COLS}))
        feat["LMR"] = r["lmr"]
        X.append([feat[c] for c in FEATURE_ORDER])
        y.append(r["y"]); cov.append(r["cov"]); groups.append(scaffold_of(s)); kept_smiles.append(s)

    Xdf = pd.DataFrame(X, columns=FEATURE_ORDER)
    feat_med = Xdf.median(numeric_only=True)
    Xdf = Xdf.fillna(feat_med)                       # impute failed xTB with medians
    n_xtb_ok = int(np.isfinite([xd[s]["xtb_HOMO"] for s in uniq]).sum())
    print(f"Feature matrix {Xdf.shape} | xTB ok for {n_xtb_ok}/{len(uniq)} unique mols")

    # Morgan fps for domain-similarity readout
    fps = []
    for s in kept_smiles:
        m = Chem.MolFromSmiles(s)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None)

    joblib.dump(dict(X=Xdf, y=np.asarray(y, float), cov=np.asarray(cov, float),
                     groups=np.asarray(groups, object), feat_med=feat_med,
                     train_fps=fps, train_smiles=kept_smiles, feature_order=FEATURE_ORDER),
                CACHE)
    keep = np.asarray(cov) <= 3.0
    print(f"COV<=3.0 keeps {keep.sum()}/{len(cov)} rows")
    print(f"Saved -> {CACHE}")


if __name__ == "__main__":
    main()
