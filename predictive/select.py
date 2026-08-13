"""
predictive/select.py  (pipeline step 2)
=======================================
Shortlist the feature set down to the columns that actually move the target, using
Recursive Feature Elimination with scaffold-grouped cross-validation (RFE-CV).

RFE-CV repeatedly refits an ExtraTrees model, drops the least-important feature, and keeps
the subset whose grouped-CV R^2 is best - grouped by Bemis-Murcko scaffold so the score
reflects generalization to novel chemistry, not memorized scaffolds. The surviving columns
are written to selected_features.json and consumed by tune / train / design.

TARGET controls it:
  n_features   None -> RFE-CV picks the count automatically; an int forces the top-K.
  pin_context  True -> the context columns are always kept, regardless of RFE.

Run:  python -m batterygen.predictive.select
Output: <artifacts>/predictive/selected_features.json  (+ a CV-curve figure)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import joblib

from batterygen.predictive.target import TARGET
from batterygen.predictive import paths, transform


def _importance_topk(X, y, names, k):
    from sklearn.ensemble import ExtraTreesRegressor
    et = ExtraTreesRegressor(n_estimators=600, random_state=42, n_jobs=-1).fit(X, y)
    order = np.argsort(et.feature_importances_)[::-1]
    return [names[i] for i in order[:k]]


def main():
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import GroupKFold

    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", type=float, default=TARGET.max_cov,
                    help="keep rows with replicate CoV%% <= this")
    ap.add_argument("--step", type=int, default=1, help="features dropped per RFE iteration")
    args = ap.parse_args()

    print("=" * 70 + "\nSTEP 2: FEATURE SELECTION (RFE-CV)\n" + "=" * 70)
    C = joblib.load(paths.FEATURE_CACHE)
    names = list(C["feature_order"])
    keep = C["cov"] <= args.cov
    X = C["X"].values[keep]
    y = transform.forward(C["y"][keep])
    g = C["groups"][keep]
    n_groups = len(set(g))
    print(f"  rows={keep.sum()} | candidate features={len(names)} | scaffolds={n_groups}")

    context = [c for c in TARGET.context_cols if c in names]
    pinned = context if TARGET.pin_context else []

    cv_r2 = None
    if TARGET.n_features is not None:
        selected = _importance_topk(X, y, names, int(TARGET.n_features))
        method = f"importance-top{int(TARGET.n_features)}"
        print(f"  forced top-{TARGET.n_features} by ExtraTrees importance")
    else:
        n_splits = max(2, min(5, n_groups))
        est = ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        rfecv = RFECV(est, step=args.step, cv=GroupKFold(n_splits), scoring="r2",
                      min_features_to_select=max(1, len(pinned)), n_jobs=-1)
        rfecv.fit(X, y, groups=g)
        selected = [n for n, keep_f in zip(names, rfecv.support_) if keep_f]
        cv_r2 = float(np.max(rfecv.cv_results_["mean_test_score"]))
        method = "rfe-cv"
        print(f"  RFE-CV kept {rfecv.n_features_}/{len(names)} features "
              f"| best CV R^2={cv_r2:.3f}")
        _plot(rfecv, len(names))

    # honor pinned context columns even if RFE dropped them
    for c in pinned:
        if c not in selected:
            selected.append(c)
            print(f"  pinned context feature kept: {c}")

    # preserve the canonical feature order
    selected = [n for n in names if n in selected]
    paths.PRED_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths.SELECTED, "w", encoding="utf-8") as f:
        json.dump({"features": selected, "method": method, "cv_r2": cv_r2,
                   "n_candidates": len(names)}, f, indent=2)
    print(f"\n  shortlist ({len(selected)}): {', '.join(selected)}")
    print(f"  Saved -> {paths.SELECTED}\n")


def _plot(rfecv, n_total):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        paths.FIG_DIR.mkdir(parents=True, exist_ok=True)
        scores = rfecv.cv_results_["mean_test_score"]
        xs = range(max(1, len(rfecv.support_) and 1), len(scores) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(list(xs), scores, "-o", color="#4C72B0", ms=3)
        plt.axvline(rfecv.n_features_, ls=":", color="#C44E52",
                    label=f"selected {rfecv.n_features_}")
        plt.xlabel("number of features"); plt.ylabel("grouped-CV R^2")
        plt.title("RFE-CV feature selection"); plt.legend(); plt.grid(alpha=0.3)
        fig = paths.FIG_DIR / "02_rfecv.png"
        plt.tight_layout(); plt.savefig(fig, dpi=130); plt.close()
        print(f"  Figure -> {fig}")
    except Exception as e:
        print(f"  (figure skipped: {e})")


if __name__ == "__main__":
    main()
