"""
ce_train.py  (step 3)  — fit & lock the production CE model.
===========================================================
Ensemble = XGBoost + ExtraTrees + RandomForest + SVR(yeo-johnson).
Point prediction = blend_w * XGB + (1-blend_w) * SVR (the blend tuned in step2b);
ExtraTrees/RandomForest give the ensemble-std uncertainty. Reports HONEST out-of-fold
random-split R^2 (the 'test R^2' target) and scaffold R^2 (novel-molecule generalization).

`predict(bundle, smiles, lmr, compute_xtb=...)` is the scoring backbone the generator calls:
returns per molecule  -> {ce, lce, uncertainty, domain_sim, xtb_homo}.
compute_xtb=False imputes xTB from training medians (fast, for screening thousands);
True runs GFN2-xTB per molecule (accurate, for a final shortlist).

Run:  python molvae/ce_train.py [--gpu]
Output: molvae_artifacts/ce/production_model.pkl, .../figures/03_feature_importance.png
"""
import warnings; warnings.filterwarnings("ignore")
import argparse
import os
import numpy as np
import joblib

import config
import ce_features as F

CE_DIR = config.CKPT_DIR.parent / "ce"
CACHE = CE_DIR / "feature_cache.pkl"
MODEL = CE_DIR / "production_model.pkl"
HPARAMS = CE_DIR / "best_hparams.pkl"
FIG = CE_DIR / "figures" / "03_feature_importance.png"
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


def lce_fwd(ce):
    """CE % -> LCE = -log10(1 - CE/100). Stretches the high-CE tail (98-99.9%)."""
    return -np.log10(np.clip(1.0 - np.asarray(ce, float) / 100.0, 1e-3, 1.0))


def lce_inv(lce):
    """LCE -> CE %."""
    return 100.0 * (1.0 - np.power(10.0, -np.asarray(lce, float)))


def _oof_blend_r2(X, y, g, xkw, etkw, et_w, splitter, use_groups, target="ce"):
    """Out-of-fold R^2 (always reported in CE space) of the ExtraTrees+XGBoost blend.
    target='lce' fits on the log-transformed target then inverts for scoring."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import r2_score
    import xgboost as xgb

    yt = lce_fwd(y) if target == "lce" else y
    pred = np.zeros(len(y))
    it = splitter.split(X, y, g) if use_groups else splitter.split(X)
    for tr, te in it:
        sc = StandardScaler().fit(X[tr]); a, b = sc.transform(X[tr]), sc.transform(X[te])
        xm = xgb.XGBRegressor(**xkw).fit(a, yt[tr])
        et = ExtraTreesRegressor(**etkw).fit(a, yt[tr])
        p = et_w * et.predict(b) + (1 - et_w) * xm.predict(b)
        pred[te] = lce_inv(p) if target == "lce" else p
    return r2_score(y, pred)


def main():
    from sklearn.preprocessing import StandardScaler, PowerTransformer
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.model_selection import GroupKFold, KFold
    import xgboost as xgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="allow XGBoost CUDA (default CPU)")
    ap.add_argument("--cov", type=float, default=3.0)
    ap.add_argument("--target", choices=["ce", "lce"], default="lce",
                    help="fit on raw CE or LCE=-log10(1-CE/100) (lce resolves the 98-99%% tail)")
    args = ap.parse_args()

    print("=" * 70 + "\nSTEP 3: TRAIN & LOCK PRODUCTION CE MODEL\n" + "=" * 70)
    C = joblib.load(CACHE)
    cov = C["cov"] if args.cov is None else args.cov
    keep = C["cov"] <= args.cov
    X = C["X"].values[keep]; y = C["y"][keep]; g = C["groups"][keep]
    use_gpu = gpu_available(not args.gpu)
    print(f"  rows={keep.sum()} | features={X.shape[1]} | XGB {'GPU' if use_gpu else 'CPU'}")

    et_w = 0.65
    etkw = dict(n_estimators=800, max_features=0.5, min_samples_leaf=1, random_state=RS, n_jobs=-1)
    if HPARAMS.exists():
        hp = joblib.load(HPARAMS)
        xkw = dict(**hp["xgb_params"], random_state=RS, tree_method="hist")
        etkw.update(hp.get("et_params", {})); etkw["random_state"] = RS; etkw["n_jobs"] = -1
        et_w = hp.get("et_w", et_w)
        print(f"  Optuna hparams loaded (tuned scaffold R2={hp['scaffold_R2']:.3f}, "
              f"random R2={hp['random_R2']:.3f}, et_w={et_w:.2f})")
    else:
        xkw = dict(n_estimators=500, max_depth=3, learning_rate=0.04, subsample=0.8,
                   colsample_bytree=0.7, reg_lambda=3.0, random_state=RS, tree_method="hist")
        print("  Default hparams (run ce_tune.py to tune).")
    xkw.update(device="cuda") if use_gpu else xkw.update(n_jobs=-1)

    tgt = args.target
    print(f"  target = {tgt.upper()}")
    # honest OOF R^2 (always in CE space) of the deployed ExtraTrees+XGBoost blend
    gkf, kf = GroupKFold(5), KFold(5, shuffle=True, random_state=RS)
    random_r2 = _oof_blend_r2(X, y, g, xkw, etkw, et_w, kf, False, target=tgt)
    scaffold_r2 = _oof_blend_r2(X, y, g, xkw, etkw, et_w, gkf, True, target=tgt)
    print(f"  OOF random-split R^2 = {random_r2:.3f}  (target >0.75: "
          f"{'MET' if random_r2 > 0.75 else 'NOT met'})")
    print(f"  OOF scaffold    R^2 = {scaffold_r2:.3f}  (novel-molecule generalization)")

    # fit production models on ALL kept data (in the chosen target space)
    yt = lce_fwd(y) if tgt == "lce" else y
    scaler = StandardScaler().fit(X); Xs = scaler.transform(X)
    xg = xgb.XGBRegressor(**xkw).fit(Xs, yt)
    et = ExtraTreesRegressor(**etkw).fit(Xs, yt)
    rf = RandomForestRegressor(n_estimators=600, random_state=RS, n_jobs=-1).fit(Xs, yt)
    print(f"  Fitted ExtraTrees + XGBoost (+ RandomForest for uncertainty) | et_w={et_w}")

    bundle = dict(scaler=scaler, xgb=xg, et=et, rf=rf, et_w=et_w, target=tgt,
                  feature_order=F.FEATURE_ORDER, feat_med=C["feat_med"],
                  train_fps=C["train_fps"], train_smiles=C["train_smiles"],
                  scaffold_R2=round(float(scaffold_r2), 4), random_R2=round(float(random_r2), 4),
                  y_range=[float(y.min()), float(y.max())])
    CE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL)
    print(f"  Saved -> {MODEL}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG.parent.mkdir(parents=True, exist_ok=True)
        imp = xg.feature_importances_; order = np.argsort(imp)[::-1]
        names = [F.FEATURE_ORDER[i] for i in order]
        colors = ["#C44E52" if F.FEATURE_ORDER[i].startswith("xtb") else "#4C72B0" for i in order]
        plt.figure(figsize=(9, 6))
        plt.barh(range(len(imp)), imp[order][::-1], color=colors[::-1])
        plt.yticks(range(len(imp)), names[::-1], fontsize=8)
        plt.xlabel("XGBoost importance (gain)")
        plt.title("CE feature importance (red = xTB electronic, blue = RDKit)")
        plt.tight_layout(); plt.savefig(FIG, dpi=130); plt.close()
        print(f"  Figure -> {FIG} | top: {', '.join(names[:5])}")
    except Exception as e:
        print(f"  (figure skipped: {e})")
    print("\nSTEP 3 complete.\n")


# --------------------------------------------------------------------------- #
# Inference backbone (used by the generator / ce_model design loop)
# --------------------------------------------------------------------------- #
def load_production():
    if not MODEL.exists():
        raise SystemExit("No production CE model. Run: python molvae/ce_features.py && "
                         "python molvae/ce_tune.py && python molvae/ce_train.py")
    return joblib.load(MODEL)


def _feature_rows(bundle, smiles_list, lmr, compute_xtb, xtb_cache):
    """Build the FEATURE_ORDER matrix for a list of SMILES. Missing xTB -> imputed medians
    (compute_xtb=False) or GFN2-xTB (True, cached in xtb_cache dict)."""
    from rdkit import Chem
    med = bundle["feat_med"]
    rows, keep = [], []
    todo_xtb = []
    if compute_xtb:
        todo_xtb = [s for s in dict.fromkeys(smiles_list) if s not in (xtb_cache or {})]
    if compute_xtb and todo_xtb:
        import tempfile, xtb_label
        env = dict(os.environ); env.setdefault("OMP_NUM_THREADS", "4")
        scratch = __import__("pathlib").Path(tempfile.mkdtemp(prefix="xtb_ce_"))
        for s in todo_xtb:
            m = Chem.MolFromSmiles(s); chrg = Chem.GetFormalCharge(m) if m else 0
            r = xtb_label.run_xtb(s, scratch, opt=False, charge=chrg, timeout=120, env=env)
            xtb_cache[s] = {k: (r.get(k) if r else None) for k in ("homo", "lumo", "gap", "dipole")}
    for s in smiles_list:
        rd = F.rdkit_features(s)
        if rd is None:
            continue
        feat = dict(rd)
        if compute_xtb and xtb_cache.get(s):
            feat.update(F.xtb_derived(xtb_cache[s]))
        else:
            feat.update({c: np.nan for c in F.XTB_COLS})
        feat["LMR"] = lmr
        vec = [feat[c] if (c in feat and feat[c] == feat[c]) else float(med[c]) for c in F.FEATURE_ORDER]
        rows.append(vec); keep.append(s)
    return np.asarray(rows, float), keep


def predict(bundle, smiles_list, lmr, compute_xtb=False, xtb_cache=None):
    """{smiles: {ce, lce, uncertainty, domain_sim, xtb_homo}} for each encodable SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    if xtb_cache is None:
        xtb_cache = {}
    X, keep = _feature_rows(bundle, smiles_list, lmr, compute_xtb, xtb_cache)
    if not keep:
        return {}
    Xs = bundle["scaler"].transform(X)
    p_xgb = bundle["xgb"].predict(Xs)
    p_et = bundle["et"].predict(Xs)
    p_rf = bundle["rf"].predict(Xs)
    ew = bundle.get("et_w", 0.65)
    blend = ew * p_et + (1 - ew) * p_xgb
    if bundle.get("target") == "lce":   # models predict LCE -> convert to CE %
        ce = lce_inv(blend)
        unc = np.std(np.vstack([lce_inv(p_xgb), lce_inv(p_et), lce_inv(p_rf)]), axis=0)
    else:
        ce = blend
        unc = np.std(np.vstack([p_xgb, p_et, p_rf]), axis=0)
    out = {}
    fps = [fp for fp in bundle["train_fps"] if fp is not None]
    for i, s in enumerate(keep):
        m = Chem.MolFromSmiles(s)
        sim = 0.0
        if m is not None and fps:
            q = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
            sim = float(max(DataStructs.BulkTanimotoSimilarity(q, fps)))
        c = float(ce[i]); frac = min(max(c / 100.0, 0.0), 0.9999)
        out[s] = {"ce": c, "lce": float(-np.log10(max(1e-3, 1 - frac))),
                  "uncertainty": float(unc[i]), "domain_sim": round(sim, 3),
                  "xtb_homo": float(X[i, F.FEATURE_ORDER.index("xtb_HOMO")])}
    return out


if __name__ == "__main__":
    main()
