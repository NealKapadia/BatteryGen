"""
predictive/train.py  (pipeline step 4)  - fit & lock the production model.
==========================================================================
Ensemble = XGBoost + ExtraTrees (+ RandomForest for the uncertainty readout), trained on
the SHORTLISTED features from select.py, in the target space chosen by TARGET.target_transform.
Point prediction = et_w * ExtraTrees + (1-et_w) * XGBoost; the RF/ET/XGB spread gives the
ensemble-std uncertainty. Reports honest out-of-fold random-split R^2 and scaffold R^2
(novel-molecule generalization).

`predict(bundle, smiles, context, compute_xtb=...)` is the scoring backbone the generator
loop calls: returns per molecule -> {pred, unc, domain_sim, xtb_homo}.

Run:  python -m molforge.predictive.train [--gpu]
Output: <artifacts>/predictive/production_model.pkl (+ a feature-importance figure)
"""
import warnings; warnings.filterwarnings("ignore")
import argparse
import json

import numpy as np
import joblib

from molforge.predictive.target import TARGET
from molforge.predictive import paths, transform
from molforge.predictive import features as F

RS = 42


def gpu_available(force_cpu):
    if force_cpu:
        return False
    try:
        import xgboost as xgb
        xgb.XGBRegressor(device="cuda", n_estimators=2).fit(np.zeros((4, 3)), np.zeros(4))
        return True
    except Exception:
        return False


def _oof_blend_r2(X, y, g, xkw, etkw, et_w, splitter, use_groups):
    """Out-of-fold R^2 (reported in the ORIGINAL target space) of the ExtraTrees+XGBoost blend."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import r2_score
    import xgboost as xgb

    yt = transform.forward(y)
    pred = np.zeros(len(y))
    it = splitter.split(X, y, g) if use_groups else splitter.split(X)
    for tr, te in it:
        sc = StandardScaler().fit(X[tr]); a, b = sc.transform(X[tr]), sc.transform(X[te])
        xm = xgb.XGBRegressor(**xkw).fit(a, yt[tr])
        et = ExtraTreesRegressor(**etkw).fit(a, yt[tr])
        p = et_w * et.predict(b) + (1 - et_w) * xm.predict(b)
        pred[te] = transform.inverse(p)
    return r2_score(y, pred)


def _load_selected(C):
    names = list(C["feature_order"])
    if not paths.SELECTED.exists():
        raise SystemExit("No feature shortlist. Run: python -m molforge.predictive.select")
    with open(paths.SELECTED, encoding="utf-8") as f:
        sel = json.load(f)["features"]
    sel = [n for n in names if n in sel]        # canonical order
    idx = [names.index(n) for n in sel]
    return sel, idx


def main():
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.model_selection import GroupKFold, KFold
    import xgboost as xgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="allow XGBoost CUDA (default CPU)")
    ap.add_argument("--cov", type=float, default=TARGET.max_cov)
    args = ap.parse_args()

    print("=" * 70 + "\nSTEP 4: TRAIN & LOCK PRODUCTION MODEL\n" + "=" * 70)
    C = joblib.load(paths.FEATURE_CACHE)
    sel, idx = _load_selected(C)
    keep = C["cov"] <= args.cov
    X = C["X"].values[keep][:, idx]; y = C["y"][keep]; g = C["groups"][keep]
    use_gpu = gpu_available(not args.gpu)
    print(f"  rows={keep.sum()} | features={len(sel)} ({', '.join(sel)}) | "
          f"XGB {'GPU' if use_gpu else 'CPU'} | transform={TARGET.target_transform}")

    et_w = 0.65
    etkw = dict(n_estimators=800, max_features=0.5, min_samples_leaf=1, random_state=RS, n_jobs=-1)
    if paths.HPARAMS.exists():
        hp = joblib.load(paths.HPARAMS)
        xkw = dict(**hp["xgb_params"], random_state=RS, tree_method="hist")
        etkw.update(hp.get("et_params", {})); etkw["random_state"] = RS; etkw["n_jobs"] = -1
        et_w = hp.get("et_w", et_w)
        print(f"  Optuna hparams loaded (tuned scaffold R^2={hp['scaffold_R2']:.3f}, "
              f"random R^2={hp['random_R2']:.3f}, et_w={et_w:.2f})")
    else:
        xkw = dict(n_estimators=500, max_depth=3, learning_rate=0.04, subsample=0.8,
                   colsample_bytree=0.7, reg_lambda=3.0, random_state=RS, tree_method="hist")
        print("  Default hparams (run tune.py to tune).")
    xkw.update(device="cuda") if use_gpu else xkw.update(n_jobs=-1)

    # honest OOF R^2 (original target space) of the deployed ExtraTrees+XGBoost blend
    gkf, kf = GroupKFold(5), KFold(5, shuffle=True, random_state=RS)
    random_r2 = _oof_blend_r2(X, y, g, xkw, etkw, et_w, kf, False)
    scaffold_r2 = _oof_blend_r2(X, y, g, xkw, etkw, et_w, gkf, True)
    print(f"  OOF random-split R^2 = {random_r2:.3f}")
    print(f"  OOF scaffold    R^2 = {scaffold_r2:.3f}  (novel-molecule generalization)")

    # fit production models on ALL kept data (in the chosen target space)
    yt = transform.forward(y)
    scaler = StandardScaler().fit(X); Xs = scaler.transform(X)
    xg = xgb.XGBRegressor(**xkw).fit(Xs, yt)
    et = ExtraTreesRegressor(**etkw).fit(Xs, yt)
    rf = RandomForestRegressor(n_estimators=600, random_state=RS, n_jobs=-1).fit(Xs, yt)
    print(f"  Fitted ExtraTrees + XGBoost (+ RandomForest for uncertainty) | et_w={et_w}")

    bundle = dict(scaler=scaler, xgb=xg, et=et, rf=rf, et_w=et_w,
                  feature_order=sel, feat_med=C["feat_med"], use_xtb=C.get("use_xtb", True),
                  context_cols=list(TARGET.context_cols),
                  train_fps=C["train_fps"], train_smiles=C["train_smiles"],
                  scaffold_R2=round(float(scaffold_r2), 4), random_R2=round(float(random_r2), 4),
                  y_range=[float(y.min()), float(y.max())],
                  target_col=TARGET.target_col, target_name=TARGET.target_name)
    paths.PRED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, paths.MODEL)
    print(f"  Saved -> {paths.MODEL}")
    _plot_importance(xg, sel)
    print("\nSTEP 4 complete.\n")


def _plot_importance(xg, names):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        paths.FIG_DIR.mkdir(parents=True, exist_ok=True)
        imp = xg.feature_importances_; order = np.argsort(imp)[::-1]
        lbl = [names[i] for i in order]
        colors = ["#C44E52" if names[i].startswith("xtb") else "#4C72B0" for i in order]
        plt.figure(figsize=(9, 6))
        plt.barh(range(len(imp)), imp[order][::-1], color=colors[::-1])
        plt.yticks(range(len(imp)), lbl[::-1], fontsize=8)
        plt.xlabel("XGBoost importance (gain)")
        plt.title(f"{TARGET.target_name} feature importance (red = xTB, blue = RDKit/context)")
        fig = paths.FIG_DIR / "04_feature_importance.png"
        plt.tight_layout(); plt.savefig(fig, dpi=130); plt.close()
        print(f"  Figure -> {fig} | top: {', '.join(lbl[:5])}")
    except Exception as e:
        print(f"  (figure skipped: {e})")


# --------------------------------------------------------------------------- #
# Inference backbone (used by the design loop)
# --------------------------------------------------------------------------- #
def load_production():
    if not paths.MODEL.exists():
        raise SystemExit("No production model. Run: "
                         "python -m molforge.predictive.features && "
                         "python -m molforge.predictive.select && "
                         "python -m molforge.predictive.train")
    return joblib.load(paths.MODEL)


def _feature_rows(bundle, smiles_list, context, compute_xtb, xtb_cache):
    """Build the selected-feature matrix for a list of SMILES. Missing values -> imputed medians;
    xTB columns computed with GFN2-xTB only when compute_xtb and an xTB feature was selected."""
    from rdkit import Chem
    import os, tempfile
    from pathlib import Path

    sel = bundle["feature_order"]
    med = bundle["feat_med"]
    need_xtb = compute_xtb and any(c in F.XTB_COLS for c in sel)
    if need_xtb:
        todo = [s for s in dict.fromkeys(smiles_list) if s not in (xtb_cache or {})]
        if todo:
            from molforge.grounding import xtb_label
            env = dict(os.environ); env.setdefault("OMP_NUM_THREADS", "4")
            scratch = Path(tempfile.mkdtemp(prefix="xtb_pred_"))
            for s in todo:
                m = Chem.MolFromSmiles(s); chrg = Chem.GetFormalCharge(m) if m else 0
                r = xtb_label.run_xtb(s, scratch, opt=False, charge=chrg, timeout=120, env=env)
                xtb_cache[s] = {k: (r.get(k) if r else None) for k in ("homo", "lumo", "gap", "dipole")}

    rows, keep = [], []
    for s in smiles_list:
        rd = F.rdkit_features(s)
        if rd is None:
            continue
        feat = dict(rd)
        if need_xtb and (xtb_cache or {}).get(s):
            feat.update(F.xtb_derived(xtb_cache[s]))
        feat.update(context or {})
        vec = [feat[c] if (c in feat and feat[c] == feat[c]) else float(med[c]) for c in sel]
        rows.append(vec); keep.append(s)
    return np.asarray(rows, float), keep


def predict(bundle, smiles_list, context=None, compute_xtb=False, xtb_cache=None):
    """{smiles: {pred, unc, domain_sim, xtb_homo}} for each encodable SMILES.
    `context` is a dict of TARGET.context_col -> value (defaults to training medians)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    if xtb_cache is None:
        xtb_cache = {}
    sel = bundle["feature_order"]
    context = context or {c: float(bundle["feat_med"][c]) for c in bundle.get("context_cols", [])}
    X, keep = _feature_rows(bundle, smiles_list, context, compute_xtb, xtb_cache)
    if not keep:
        return {}
    Xs = bundle["scaler"].transform(X)
    p_xgb = bundle["xgb"].predict(Xs)
    p_et = bundle["et"].predict(Xs)
    p_rf = bundle["rf"].predict(Xs)
    ew = bundle.get("et_w", 0.65)
    blend = ew * p_et + (1 - ew) * p_xgb
    pred = transform.inverse(blend)
    unc = np.std(np.vstack([transform.inverse(p_xgb), transform.inverse(p_et),
                            transform.inverse(p_rf)]), axis=0)
    has_homo = "xtb_HOMO" in sel
    homo_i = sel.index("xtb_HOMO") if has_homo else None
    out = {}
    fps = [fp for fp in bundle["train_fps"] if fp is not None]
    for i, s in enumerate(keep):
        m = Chem.MolFromSmiles(s)
        sim = 0.0
        if m is not None and fps:
            q = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
            sim = float(max(DataStructs.BulkTanimotoSimilarity(q, fps)))
        out[s] = {"pred": float(pred[i]), "unc": float(unc[i]), "domain_sim": round(sim, 3),
                  "xtb_homo": (float(X[i, homo_i]) if has_homo else float("nan"))}
    return out


if __name__ == "__main__":
    main()
