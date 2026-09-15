import os, json, warnings, joblib, numpy as np, optuna
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from . import paths, transform
from .target import TARGET

RS, N_SPLITS = 42, 5
N_TRIALS = int(os.getenv("BATTERYGEN_TUNE_TRIALS", "75"))

def rmse(y, pred): return float(np.sqrt(mean_squared_error(y, pred)))

def selected_columns(cache):
    names = list(cache["feature_order"])
    with open(paths.SELECTED, encoding="utf-8") as f: selected = json.load(f)["features"]
    selected = [x for x in names if x in selected]
    return [names.index(x) for x in selected], selected

def load_data():
    if not paths.FEATURE_CACHE.exists(): raise FileNotFoundError(f"Missing feature cache: {paths.FEATURE_CACHE}")
    if not paths.SELECTED.exists(): raise FileNotFoundError(f"Missing selected features: {paths.SELECTED}")
    cache = joblib.load(paths.FEATURE_CACHE)
    idx, selected = selected_columns(cache)
    keep = np.asarray(cache["cov"], dtype=float) <= float(TARGET.max_cov)
    X0 = cache["X"].values if hasattr(cache["X"], "values") else np.asarray(cache["X"])
    X = np.asarray(X0[keep][:, idx], dtype=float)
    y = np.asarray(cache["y"], dtype=float)[keep]
    y_model = np.asarray(transform.forward(y), dtype=float)
    groups = np.asarray(cache["groups"], dtype=object)[keep]
    return X, y, y_model, groups, selected

def scaffold_cv_rmse(builder, X, y, y_model, groups):
    scores = []
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y_model, groups):
        scaler = StandardScaler().fit(X[tr])
        model = builder()
        model.fit(scaler.transform(X[tr]), y_model[tr])
        pred = np.asarray(transform.inverse(np.asarray(model.predict(scaler.transform(X[te])), dtype=float).reshape(-1)), dtype=float)
        scores.append(rmse(y[te], pred))
    return float(np.mean(scores))

def main():
    X, y, y_model, groups, selected = load_data()
    print("="*80, "\nBATTERYGEN THREE-MODEL HYPERPARAMETER TUNING\n", "="*80, sep="")
    print(f"Rows             : {len(y)}\nSelected features: {len(selected)}\nScaffold groups  : {len(set(groups))}\nTrials/model     : {N_TRIALS}")
    if len(set(groups)) < N_SPLITS: raise RuntimeError(f"Need at least {N_SPLITS} scaffold groups.")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective_lgbm(t):
        p = {
            "n_estimators": t.suggest_int("n_estimators", 100, 1200),
            "learning_rate": t.suggest_float("learning_rate", 1e-3, 0.15, log=True),
            "num_leaves": t.suggest_int("num_leaves", 8, 128),
            "max_depth": t.suggest_int("max_depth", 3, 12),
            "min_child_samples": t.suggest_int("min_child_samples", 5, 50),
            "subsample": t.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.60, 1.00),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
        }
        return scaffold_cv_rmse(lambda: LGBMRegressor(objective="regression", random_state=RS, n_jobs=-1, verbosity=-1, subsample_freq=1, **p), X, y, y_model, groups)

    def objective_svr(t):
        p = {
            "C": t.suggest_float("C", 1e-2, 1e3, log=True),
            "gamma": t.suggest_float("gamma", 1e-5, 1.0, log=True),
            "epsilon": t.suggest_float("epsilon", 1e-4, 0.5, log=True)
        }
        return scaffold_cv_rmse(lambda: SVR(kernel="rbf", cache_size=1000, **p), X, y, y_model, groups)

    def objective_cat(t):
        p = {
            "iterations": t.suggest_int("iterations", 200, 1500),
            "depth": t.suggest_int("depth", 4, 10),
            "learning_rate": t.suggest_float("learning_rate", 1e-3, 0.15, log=True),
            "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1e-3, 30.0, log=True),
            "random_strength": t.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": t.suggest_float("bagging_temperature", 0.0, 10.0)
        }
        return scaffold_cv_rmse(lambda: CatBoostRegressor(loss_function="RMSE", random_seed=RS, verbose=False, allow_writing_files=False, thread_count=-1, **p), X, y, y_model, groups)

    studies = {}
    for name, objective in [("LightGBM", objective_lgbm), ("RBF-SVR", objective_svr), ("CatBoost", objective_cat)]:
        print(f"\nTuning {name}...")
        s = optuna.create_study(direction="minimize")
        s.optimize(objective, n_trials=N_TRIALS)
        studies[name] = s
        print(f"Best {name} scaffold RMSE: {s.best_value:.6f}")

    result = {
        "lgbm_params": dict(studies["LightGBM"].best_params),
        "svr_params": dict(studies["RBF-SVR"].best_params),
        "catboost_params": dict(studies["CatBoost"].best_params),
        "lgbm_scaffold_rmse": float(studies["LightGBM"].best_value),
        "svr_scaffold_rmse": float(studies["RBF-SVR"].best_value),
        "catboost_scaffold_rmse": float(studies["CatBoost"].best_value),
        "tuning_method": "5-fold scaffold GroupKFold",
        "objective": "CE-space RMSE",
        "random_state": RS
    }

    paths.PRED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, paths.HPARAMS)
    print("\nSaved hyperparameters to:", paths.HPARAMS)

if __name__ == "__main__": main()
