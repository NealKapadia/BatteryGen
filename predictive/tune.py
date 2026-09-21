"""
BatteryGen predictive hyperparameter tuning:
LightGBM + RBF-SVR + CatBoost

Missing-value policy:
- Do NOT drop rows or features because of NaN/inf descriptor values.
- Convert non-finite values to NaN.
- Fit a median SimpleImputer ONLY on each training fold.
- keep_empty_features=True preserves feature dimensionality even if a feature is
  entirely missing within a particular training fold.
- Fit StandardScaler only after imputation, again using the training fold only.

The original TARGET.max_cov replicate-quality filter is preserved.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import optuna

from optuna.samplers import TPESampler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ============================================================================
# PROJECT-LOCAL BATTERYGEN STORAGE
# ============================================================================
# This tune.py lives in:
#   <BatteryGen-main>/predictive/tune.py
#
# All artifacts are kept under:
#   <BatteryGen-main>/batterygen_artifacts/
#
# No notebook setup cell is required.
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "batterygen_artifacts"
os.environ["BATTERYGEN_ART_DIR"] = str(ARTIFACT_ROOT)
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

from . import paths, transform
from .target import TARGET


RS = 42
N_SPLITS = 5
N_TRIALS = int(os.getenv("BATTERYGEN_TUNE_TRIALS", "75"))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def finite_to_nan(X):
    """Preserve every row/column; represent +/-inf as NaN for imputation."""
    X = np.asarray(X, dtype=float).copy()
    X[~np.isfinite(X)] = np.nan
    return X


def make_imputer():
    """Median imputation without dropping all-missing features."""
    return SimpleImputer(strategy="median", keep_empty_features=True)


def selected_columns(cache):
    names = list(cache["feature_order"])
    with open(paths.SELECTED, encoding="utf-8") as f:
        selected = json.load(f)["features"]

    selected = [name for name in names if name in selected]
    idx = [names.index(name) for name in selected]
    return idx, selected



def _recover_existing_artifact(target_path):
    """Recover an already-created feature/selection artifact into aligned storage."""
    target_path = Path(target_path)

    if target_path.exists():
        return

    candidates = []

    for root in (PROJECT_ROOT, PROJECT_ROOT.parent):
        if not root.exists():
            continue

        try:
            for candidate in root.rglob(target_path.name):
                try:
                    if (
                        candidate.is_file()
                        and candidate.resolve() != target_path.resolve()
                    ):
                        candidates.append(candidate)
                except OSError:
                    pass
        except (OSError, PermissionError):
            pass

    if not candidates:
        return

    source = max(candidates, key=lambda p: p.stat().st_mtime)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_path)

    print(f"[artifact alignment] recovered {target_path.name}")
    print(f"    from: {source}")
    print(f"      to: {target_path}")



def load_data():
    _recover_existing_artifact(paths.FEATURE_CACHE)
    _recover_existing_artifact(paths.SELECTED)

    if not paths.FEATURE_CACHE.exists():
        raise FileNotFoundError(f"Missing feature cache: {paths.FEATURE_CACHE}")
    if not paths.SELECTED.exists():
        raise FileNotFoundError(f"Missing selected features: {paths.SELECTED}")

    cache = joblib.load(paths.FEATURE_CACHE)
    idx, selected = selected_columns(cache)

    # Preserve the replicate-quality rule from the original workflow.
    keep = np.asarray(cache["cov"], dtype=float) <= float(TARGET.max_cov)

    X_full = cache["X"].values if hasattr(cache["X"], "values") else np.asarray(cache["X"])
    X = finite_to_nan(X_full[keep][:, idx])

    y = np.asarray(cache["y"], dtype=float)[keep]
    y_model = np.asarray(transform.forward(y), dtype=float)
    groups = np.asarray(cache["groups"], dtype=object)[keep]

    return X, y, y_model, groups, selected


def scaffold_cv_rmse(builder, X, y, y_model, groups):
    splitter = GroupKFold(n_splits=N_SPLITS)
    scores = []

    for tr, te in splitter.split(X, y_model, groups):
        # Leakage-safe preprocessing: fit only on the training fold.
        imputer = make_imputer()
        X_tr = imputer.fit_transform(X[tr])
        X_te = imputer.transform(X[te])

        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_te = scaler.transform(X_te)

        model = builder()
        model.fit(X_tr, y_model[tr])

        pred_model = np.asarray(model.predict(X_te), dtype=float).reshape(-1)
        pred_ce = np.asarray(transform.inverse(pred_model), dtype=float)
        scores.append(rmse(y[te], pred_ce))

    return float(np.mean(scores))


def main():
    X, y, y_model, groups, selected = load_data()

    print("=" * 80)
    print("BATTERYGEN — THREE-MODEL HYPERPARAMETER TUNING")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Artifact root: {ARTIFACT_ROOT}")
    print(f"Output dir   : {paths.PRED_DIR}")
    print("=" * 80)
    print(f"Rows             : {len(y)}")
    print(f"Selected features: {len(selected)}")
    print(f"Scaffold groups  : {len(set(groups))}")
    print(f"Trials/model     : {N_TRIALS}")
    print(f"Missing cells    : {int(np.isnan(X).sum())}")

    if len(set(groups)) < N_SPLITS:
        raise RuntimeError(f"Need at least {N_SPLITS} scaffold groups.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    def objective_lgbm(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        return scaffold_cv_rmse(
            lambda: LGBMRegressor(
                objective="regression",
                random_state=RS,
                n_jobs=-1,
                verbosity=-1,
                subsample_freq=1,
                **p,
            ),
            X, y, y_model, groups,
        )

    print("\nTuning LightGBM...")
    study_lgbm = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RS),
    )
    study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS)
    print(f"Best LightGBM scaffold RMSE: {study_lgbm.best_value:.6f}")

    # ------------------------------------------------------------------
    # RBF-SVR
    # ------------------------------------------------------------------
    def objective_svr(trial):
        p = {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-5, 1.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-4, 0.5, log=True),
        }
        return scaffold_cv_rmse(
            lambda: SVR(kernel="rbf", cache_size=1000, **p),
            X, y, y_model, groups,
        )

    print("\nTuning RBF-SVR...")
    study_svr = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RS + 1),
    )
    study_svr.optimize(objective_svr, n_trials=N_TRIALS)
    print(f"Best RBF-SVR scaffold RMSE: {study_svr.best_value:.6f}")

    # ------------------------------------------------------------------
    # CatBoost
    # ------------------------------------------------------------------
    def objective_catboost(trial):
        p = {
            "iterations": trial.suggest_int("iterations", 200, 1500),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        }
        return scaffold_cv_rmse(
            lambda: CatBoostRegressor(
                loss_function="RMSE",
                random_seed=RS,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
                **p,
            ),
            X, y, y_model, groups,
        )

    print("\nTuning CatBoost...")
    study_cat = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RS + 2),
    )
    study_cat.optimize(objective_catboost, n_trials=N_TRIALS)
    print(f"Best CatBoost scaffold RMSE: {study_cat.best_value:.6f}")

    result = {
        "lgbm_params": dict(study_lgbm.best_params),
        "svr_params": dict(study_svr.best_params),
        "catboost_params": dict(study_cat.best_params),
        "lgbm_scaffold_rmse": float(study_lgbm.best_value),
        "svr_scaffold_rmse": float(study_svr.best_value),
        "catboost_scaffold_rmse": float(study_cat.best_value),
        "tuning_method": "5-fold scaffold GroupKFold",
        "objective": "CE-space RMSE",
        "preprocessing": "median imputation -> StandardScaler, fitted within each training fold",
        "keep_empty_features": True,
        "random_state": RS,
    }

    paths.PRED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, paths.HPARAMS)

    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print("=" * 80)
    print(f"Saved hyperparameters: {Path(paths.HPARAMS).resolve()}")
    print(f"Exists               : {Path(paths.HPARAMS).exists()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
