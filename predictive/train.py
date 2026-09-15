from itertools import combinations
from pathlib import Path
import json, warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from . import paths, transform
from .target import TARGET

RS = 42
SCAFFOLD_WEIGHT, RANDOM_WEIGHT, WEIGHT_STEP = 0.70, 0.30, 0.01
PRODUCTION_MODEL = paths.PRED_DIR / "production_model.pkl"
SCAN_DIR = paths.PRED_DIR / "three_model_scan"


# ============================================================
# METRICS
# ============================================================

def rmse(y, pred):
    return float(np.sqrt(mean_squared_error(y, pred)))

def spearman(a, b):
    v = spearmanr(a, b).statistic
    return float(v) if np.isfinite(v) else np.nan

def metrics(y, pred):
    return {
        "R2": float(r2_score(y, pred)),
        "RMSE": rmse(y, pred),
        "MAE": float(mean_absolute_error(y, pred)),
        "Spearman": spearman(y, pred)
    }


# ============================================================
# DATA
# ============================================================

def selected_columns(cache):
    names = list(cache["feature_order"])
    with open(paths.SELECTED, encoding="utf-8") as f: selected = json.load(f)["features"]
    selected = [x for x in names if x in selected]
    return [names.index(x) for x in selected], selected

def load_training_data():
    for p, msg in [
        (paths.FEATURE_CACHE, "feature cache"),
        (paths.HPARAMS, "tuned hyperparameters"),
        (paths.SELECTED, "selected features")
    ]:
        if not p.exists(): raise FileNotFoundError(f"Missing {msg}: {p}")

    cache, hp = joblib.load(paths.FEATURE_CACHE), joblib.load(paths.HPARAMS)
    required = ["lgbm_params", "svr_params", "catboost_params"]
    missing = [k for k in required if k not in hp]
    if missing: raise RuntimeError(f"Hyperparameter file missing: {missing}")

    idx, selected = selected_columns(cache)
    keep = np.asarray(cache["cov"], dtype=float) <= float(TARGET.max_cov)
    X0 = cache["X"].values if hasattr(cache["X"], "values") else np.asarray(cache["X"])
    X = np.asarray(X0[keep][:, idx], dtype=float)
    y = np.asarray(cache["y"], dtype=float)[keep]
    y_model = np.asarray(transform.forward(y), dtype=float)
    groups = np.asarray(cache["groups"], dtype=object)[keep]
    return cache, hp, X, y, y_model, groups, selected


# ============================================================
# MODELS
# ============================================================

def make_lgbm(p):
    return LGBMRegressor(objective="regression", random_state=RS, n_jobs=-1,
                         verbosity=-1, subsample_freq=1, **p)

def make_svr(p):
    return SVR(kernel="rbf", cache_size=1000, **p)

def make_catboost(p):
    return CatBoostRegressor(loss_function="RMSE", random_seed=RS, verbose=False,
                             allow_writing_files=False, thread_count=-1, **p)

def make_builders(hp):
    return {
        "LightGBM": lambda: make_lgbm(dict(hp["lgbm_params"])),
        "RBF-SVR": lambda: make_svr(dict(hp["svr_params"])),
        "CatBoost": lambda: make_catboost(dict(hp["catboost_params"]))
    }


# ============================================================
# OOF PREDICTIONS
# ============================================================

def get_oof_predictions(X, y_model, groups, builders, split_type):
    if split_type == "scaffold":
        splits = GroupKFold(n_splits=5).split(X, y_model, groups)
    elif split_type == "random":
        splits = KFold(n_splits=5, shuffle=True, random_state=RS).split(X)
    else:
        raise ValueError("split_type must be 'scaffold' or 'random'")

    preds = {name: np.full(len(y_model), np.nan) for name in builders}

    for fold, (tr, te) in enumerate(splits, 1):
        print(f"{split_type.capitalize()} fold {fold}/5")
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])

        for name, builder in builders.items():
            model = builder()
            model.fit(Xtr, y_model[tr])
            preds[name][te] = np.asarray(model.predict(Xte), dtype=float).reshape(-1)

    return preds


# ============================================================
# PAIR / WEIGHT SCAN
# ============================================================

def scan_pair(y, scaffold_preds, random_preds, model_a, model_b, all_models):
    weights = np.round(np.arange(0, 1 + WEIGHT_STEP / 2, WEIGHT_STEP), 10)

    scaffold_all_ce = [
        np.asarray(transform.inverse(scaffold_preds[m]), dtype=float) for m in all_models
    ]
    random_all_ce = [
        np.asarray(transform.inverse(random_preds[m]), dtype=float) for m in all_models
    ]

    scaffold_unc = np.std(np.vstack(scaffold_all_ce), axis=0)
    random_unc = np.std(np.vstack(random_all_ce), axis=0)
    rows = []

    for wa in weights:
        wb = 1.0 - wa

        s_model = wa * scaffold_preds[model_a] + wb * scaffold_preds[model_b]
        r_model = wa * random_preds[model_a] + wb * random_preds[model_b]

        s_ce = np.asarray(transform.inverse(s_model), dtype=float)
        r_ce = np.asarray(transform.inverse(r_model), dtype=float)

        sm, rm = metrics(y, s_ce), metrics(y, r_ce)
        combined = SCAFFOLD_WEIGHT * sm["R2"] + RANDOM_WEIGHT * rm["R2"]

        rows.append({
            "Model_A": model_a,
            "Model_B": model_b,
            "Weight_A": float(wa),
            "Weight_B": float(wb),

            "Scaffold_R2": sm["R2"],
            "Scaffold_RMSE": sm["RMSE"],
            "Scaffold_MAE": sm["MAE"],
            "Scaffold_Spearman": sm["Spearman"],

            "Random_R2": rm["R2"],
            "Random_RMSE": rm["RMSE"],
            "Random_MAE": rm["MAE"],
            "Random_Spearman": rm["Spearman"],

            "Combined_R2_Score": float(combined),
            "Scaffold_Uncertainty_Error_Spearman":
                spearman(scaffold_unc, np.abs(y - s_ce)),
            "Random_Uncertainty_Error_Spearman":
                spearman(random_unc, np.abs(y - r_ce))
        })

    return pd.DataFrame(rows)


# ============================================================
# PRODUCTION PREDICTION
# ============================================================

def predict_selected(bundle, X_selected):
    X = bundle["scaler"].transform(np.asarray(X_selected, dtype=float))

    pred_model = {
        name: np.asarray(model.predict(X), dtype=float).reshape(-1)
        for name, model in bundle["models"].items()
    }

    pred_ce = {
        name: np.asarray(transform.inverse(pred), dtype=float)
        for name, pred in pred_model.items()
    }

    role = bundle["role"]
    blend = (
        role["weight_a"] * pred_model[role["model_a"]] +
        role["weight_b"] * pred_model[role["model_b"]]
    )

    pred = np.asarray(transform.inverse(blend), dtype=float)
    unc = np.std(np.vstack([
        pred_ce["LightGBM"],
        pred_ce["RBF-SVR"],
        pred_ce["CatBoost"]
    ]), axis=0)

    return {
        "pred": pred,
        "unc": unc,
        "LightGBM": pred_ce["LightGBM"],
        "RBF-SVR": pred_ce["RBF-SVR"],
        "CatBoost": pred_ce["CatBoost"]
    }

def load_production(model_path=None):
    return joblib.load(Path(model_path) if model_path is not None else PRODUCTION_MODEL)


# ============================================================
# MAIN
# ============================================================

def main():
    cache, hp, X, y, y_model, groups, selected = load_training_data()
    builders = make_builders(hp)
    model_names = list(builders)
    SCAN_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("BATTERYGEN THREE-MODEL CE ENSEMBLE")
    print("=" * 90)
    print(f"Rows             : {len(y)}")
    print(f"Selected features: {len(selected)}")
    print(f"Scaffold groups  : {len(set(groups))}")

    print("\nGenerating scaffold OOF predictions...")
    scaffold_preds = get_oof_predictions(
        X, y_model, groups, builders, "scaffold"
    )

    print("\nGenerating random OOF predictions...")
    random_preds = get_oof_predictions(
        X, y_model, groups, builders, "random"
    )

    # Individual model metrics
    rows = []
    for name in model_names:
        s_ce = np.asarray(transform.inverse(scaffold_preds[name]), dtype=float)
        r_ce = np.asarray(transform.inverse(random_preds[name]), dtype=float)
        sm, rm = metrics(y, s_ce), metrics(y, r_ce)

        rows.append({
            "Model": name,
            "Scaffold_R2": sm["R2"],
            "Scaffold_RMSE": sm["RMSE"],
            "Scaffold_MAE": sm["MAE"],
            "Random_R2": rm["R2"],
            "Random_RMSE": rm["RMSE"],
            "Random_MAE": rm["MAE"]
        })

    individual_df = pd.DataFrame(rows)

    # Scan all 3 possible pairs
    scans = []
    for a, b in combinations(model_names, 2):
        print(f"\nScanning {a} + {b}...")
        scans.append(
            scan_pair(y, scaffold_preds, random_preds, a, b, model_names)
        )

    full_scan = pd.concat(scans, ignore_index=True)

    # Best weight for each pair
    best_rows = []
    for _, group in full_scan.groupby(["Model_A", "Model_B"], sort=False):
        best_rows.append(
            group.sort_values(
                ["Combined_R2_Score", "Scaffold_R2", "Scaffold_RMSE"],
                ascending=[False, False, True]
            ).iloc[0]
        )

    best_pairs = (
        pd.DataFrame(best_rows)
        .sort_values(["Combined_R2_Score", "Scaffold_R2"],
                     ascending=[False, False])
        .reset_index(drop=True)
    )

    winner = best_pairs.iloc[0]
    remaining = [
        m for m in model_names
        if m not in (winner["Model_A"], winner["Model_B"])
    ][0]

    role = {
        "model_a": str(winner["Model_A"]),
        "model_b": str(winner["Model_B"]),
        "model_c": str(remaining),
        "weight_a": float(winner["Weight_A"]),
        "weight_b": float(winner["Weight_B"])
    }

    # Save evaluation
    individual_df.to_csv(SCAN_DIR / "individual_model_metrics.csv", index=False)
    full_scan.to_csv(SCAN_DIR / "all_pair_weight_scans.csv", index=False)
    best_pairs.to_csv(SCAN_DIR / "best_pair_by_combined_objective.csv", index=False)

    with open(SCAN_DIR / "role_assignment.json", "w", encoding="utf-8") as f:
        json.dump(role, f, indent=2)

    # Final production models
    print("\nTraining final models on all retained data...")
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    final_models = {}
    for name, builder in builders.items():
        model = builder()
        model.fit(X_scaled, y_model)
        final_models[name] = model

    bundle = {
        "architecture": "LightGBM + RBF-SVR + CatBoost",
        "scaler": scaler,
        "models": final_models,
        "role": role,
        "selected_features": list(selected),
        "feature_order": list(cache["feature_order"]),
        "target_col": TARGET.target_col,
        "max_cov": float(TARGET.max_cov),
        "uncertainty": "std_of_three_model_CE_predictions",
        "blend_space": "transformed_target_space",
        "random_state": RS
    }

    paths.PRED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, PRODUCTION_MODEL)

    print("\n" + "=" * 90)
    print("FINAL ROLE ASSIGNMENT")
    print("=" * 90)
    print("Model A :", role["model_a"])
    print("Model B :", role["model_b"])
    print("Weight A:", role["weight_a"])
    print("Weight B:", role["weight_b"])
    print("Model C :", role["model_c"])
    print("Scaffold R2:", float(winner["Scaffold_R2"]))
    print("Scaffold RMSE:", float(winner["Scaffold_RMSE"]))
    print("Random R2:", float(winner["Random_R2"]))
    print("Combined score:", float(winner["Combined_R2_Score"]))
    print("Uncertainty rho:",
          float(winner["Scaffold_Uncertainty_Error_Spearman"]))
    print("\nProduction model:", PRODUCTION_MODEL)

if __name__ == "__main__":
    main()
