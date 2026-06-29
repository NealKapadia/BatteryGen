"""
ce_tune.py  (step 2b)  — Optuna hyper-parameter search, dual objective.
======================================================================
Optimizes a weighted sum of scaffold-split and random-split out-of-fold R^2 for an
XGBoost + SVR(yeo-johnson) blend:
    combined = w_scaffold * scaffold_R2 + w_random * random_R2
Scaffold R2 = honest generalization to novel chemistry; random R2 = robustness / the
'test R^2' target. Default weights favor random (we're chasing test R^2 > 0.75 here).

Run:  python molvae/ce_tune.py --trials 200 --scaffold_weight 0.3 --random_weight 0.7
Output: molvae_artifacts/ce/best_hparams.pkl, .../figures/02b_optuna_history.png
"""
import warnings; warnings.filterwarnings("ignore")
import argparse
import numpy as np
import joblib

import config

CE_DIR = config.CKPT_DIR.parent / "ce"
CACHE = CE_DIR / "feature_cache.pkl"
OUT = CE_DIR / "best_hparams.pkl"
FIG = CE_DIR / "figures" / "02b_optuna_history.png"
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


def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import r2_score
    import xgboost as xgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--scaffold_weight", type=float, default=0.3)
    ap.add_argument("--random_weight", type=float, default=0.7)
    ap.add_argument("--cov", type=float, default=3.0, help="keep rows with CE triplicate COV <= this")
    ap.add_argument("--gpu", action="store_true", help="allow XGBoost CUDA (default CPU)")
    args = ap.parse_args()

    tw = args.scaffold_weight + args.random_weight
    w_scaf, w_rand = args.scaffold_weight / tw, args.random_weight / tw
    print("=" * 70)
    print(f"STEP 2b: OPTUNA (scaffold w={w_scaf:.2f}, random w={w_rand:.2f})")
    print("=" * 70)
    C = joblib.load(CACHE)
    keep = C["cov"] <= args.cov
    X = C["X"].values[keep]; y = C["y"][keep]; g = C["groups"][keep]
    use_gpu = gpu_available(not args.gpu)
    print(f"  {keep.sum()} rows | unique groups={len(set(g))} | XGB {'GPU' if use_gpu else 'CPU'} | trials {args.trials}")

    gkf = GroupKFold(5); kf = KFold(5, shuffle=True, random_state=RS)

    def blend_oof(xp, etp, et_w, splitter, use_groups):
        """OOF R^2 of ExtraTrees+XGBoost blend (SVR dropped: it degraded R^2 here)."""
        pred = np.zeros(len(y))
        it = splitter.split(X, y, g) if use_groups else splitter.split(X)
        for tr, te in it:
            sc = StandardScaler().fit(X[tr]); a, b = sc.transform(X[tr]), sc.transform(X[te])
            kw = dict(tree_method="hist", random_state=RS, **xp)
            kw.update(device="cuda") if use_gpu else kw.update(n_jobs=-1)
            xm = xgb.XGBRegressor(**kw).fit(a, y[tr])
            et = ExtraTreesRegressor(random_state=RS, n_jobs=-1, **etp).fit(a, y[tr])
            pred[te] = et_w * et.predict(b) + (1 - et_w) * xm.predict(b)
        return r2_score(y, pred)

    def objective(t):
        xp = dict(
            n_estimators=t.suggest_int("n_estimators", 200, 800, step=100),
            max_depth=t.suggest_int("max_depth", 2, 6),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            min_child_weight=t.suggest_int("min_child_weight", 1, 8),
        )
        etp = dict(
            n_estimators=t.suggest_int("et_n_estimators", 400, 1400, step=200),
            max_features=t.suggest_float("et_max_features", 0.2, 0.9),
            min_samples_leaf=t.suggest_int("et_min_samples_leaf", 1, 4),
        )
        et_w = t.suggest_float("et_w", 0.3, 0.85)
        s = blend_oof(xp, etp, et_w, gkf, True)
        r = blend_oof(xp, etp, et_w, kf, False)
        t.set_user_attr("scaffold_r2", s); t.set_user_attr("random_r2", r)
        return w_scaf * s + w_rand * r

    base = dict(n_estimators=500, max_depth=3, learning_rate=0.04, subsample=0.8,
                colsample_bytree=0.7, reg_lambda=3.0, min_child_weight=1)
    baseet = dict(n_estimators=800, max_features=0.5, min_samples_leaf=1)
    bs, br = blend_oof(base, baseet, 0.65, gkf, True), blend_oof(base, baseet, 0.65, kf, False)
    print(f"  baseline: scaffold R2={bs:.3f}  random R2={br:.3f}")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    bp = study.best_params
    xp = {k: bp[k] for k in ["n_estimators", "max_depth", "learning_rate", "subsample",
                             "colsample_bytree", "reg_lambda", "min_child_weight"]}
    etp = {"n_estimators": bp["et_n_estimators"], "max_features": bp["et_max_features"],
           "min_samples_leaf": bp["et_min_samples_leaf"]}
    bt = study.best_trial
    ts, tr = bt.user_attrs["scaffold_r2"], bt.user_attrs["random_r2"]
    print(f"\n  {'':10s}{'scaffold':>12s}{'random':>12s}")
    print(f"  {'baseline':10s}{bs:12.3f}{br:12.3f}")
    print(f"  {'tuned':10s}{ts:12.3f}{tr:12.3f}")

    CE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(xgb_params=xp, et_params=etp, et_w=bp["et_w"],
                     scaffold_R2=ts, random_R2=tr, cov=args.cov), OUT)
    print(f"  Saved -> {OUT}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG.parent.mkdir(parents=True, exist_ok=True)
        rv = [t.user_attrs.get("random_r2") for t in study.trials if t.value is not None]
        sv = [t.user_attrs.get("scaffold_r2") for t in study.trials if t.value is not None]
        plt.figure(figsize=(8, 5))
        plt.plot(np.maximum.accumulate(rv), "-", color="#55A630", lw=2, label="best random R2")
        plt.plot(np.maximum.accumulate(sv), "-", color="#C44E52", lw=2, label="best scaffold R2")
        plt.axhline(0.75, ls=":", color="k", label="target 0.75")
        plt.xlabel("trial"); plt.ylabel("R2"); plt.legend(); plt.grid(alpha=0.3)
        plt.title("Optuna history (CE model)"); plt.tight_layout(); plt.savefig(FIG, dpi=130); plt.close()
        print(f"  Figure -> {FIG}")
    except Exception as e:
        print(f"  (figure skipped: {e})")


if __name__ == "__main__":
    main()
