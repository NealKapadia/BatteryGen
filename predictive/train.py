"""
BatteryGen final CE predictor:
LightGBM + RBF-SVR + CatBoost

Workflow:
1. Load selected BatteryGen descriptors and tuned hyperparameters.
2. Preserve the original TARGET.max_cov replicate-quality filter.
3. Never drop rows/features because of NaN/inf descriptors:
   median-impute within each training fold, then scale.
4. Generate 5-fold scaffold OOF and 5-fold random OOF predictions.
5. Test all three two-model pairs and all weights from 0.00 to 1.00.
6. Select the pair maximizing:
       0.70 * scaffold R2 + 0.30 * random R2
7. Keep the remaining model as Model C.
8. Train all three models on all retained data.
9. Use scaffold-OOF predictions to select a disagreement penalty lambda for ranking.
10. Train all three models on all retained data.
11. Save one production_model.pkl containing imputer, scaler, models, roles, weights,
    and the ranking penalty.

Point prediction:
- Blend Model A and Model B in transformed-target space.
- Then inverse-transform to CE.

Uncertainty:
- Standard deviation across the three individual CE predictions.

Candidate ranking:
- RankingScore = Predicted_CE - lambda * disagreement_SD.
- lambda is selected from scaffold-OOF predictions to maximize recovery of
  experimental CE >= HIT_THRESHOLD within the top SCREEN_FRACTION of candidates.
- Average precision is used as a tie-breaker; if still tied, the smaller lambda wins.
- This SD is inter-model disagreement, not a calibrated probabilistic standard deviation.
"""

from itertools import combinations
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    average_precision_score,
)
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ============================================================================
# PROJECT-LOCAL BATTERYGEN STORAGE
# ============================================================================
# This train.py lives in:
#   <BatteryGen-main>/predictive/train.py
#
# Every BatteryGen artifact used/created by this file is kept under:
#   <BatteryGen-main>/batterygen_artifacts/
#
# No notebook setup cell or environment-variable command is required.
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "batterygen_artifacts"
os.environ["BATTERYGEN_ART_DIR"] = str(ARTIFACT_ROOT)

ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

from . import paths, transform, features
from .target import TARGET


RS = 42
SCAFFOLD_WEIGHT = 0.70
RANDOM_WEIGHT = 0.30
WEIGHT_STEP = 0.01

# Ranking settings. The point predictor is unchanged; these settings only determine
# how generated candidates are ordered after prediction.
HIT_THRESHOLD = 98.0
SCREEN_FRACTION = 0.20
RANK_LAMBDA_MIN = 0.0
RANK_LAMBDA_MAX = 2.0
RANK_LAMBDA_STEP = 0.05

PRODUCTION_MODEL = paths.PRED_DIR / "production_model.pkl"
SCAN_DIR = paths.PRED_DIR / "three_model_scan"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def spearman(a, b):
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else np.nan


def metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Spearman": spearman(y_true, y_pred),
    }


def finite_to_nan(X):
    """Preserve every row/column; represent +/-inf as NaN for imputation."""
    X = np.asarray(X, dtype=float).copy()
    X[~np.isfinite(X)] = np.nan
    return X


def make_imputer():
    """Median imputation while preserving all feature columns."""
    return SimpleImputer(strategy="median", keep_empty_features=True)


def selected_columns(cache):
    names = list(cache["feature_order"])
    with open(paths.SELECTED, encoding="utf-8") as f:
        selected = json.load(f)["features"]

    selected = [name for name in names if name in selected]
    idx = [names.index(name) for name in selected]
    return idx, selected



def _recover_existing_predictive_artifact(target_path):
    """
    If an already-computed feature/select/tune artifact was created before the
    project paths were standardized, copy the newest matching file into the
    aligned BatteryGen artifact folder.

    The search is restricted to this BatteryGen project and its immediate
    parent folder.
    """
    import shutil

    target_path = Path(target_path)
    if target_path.exists():
        return

    candidates = []

    for search_root in (PROJECT_ROOT, PROJECT_ROOT.parent):
        if not search_root.exists():
            continue

        try:
            for candidate in search_root.rglob(target_path.name):
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


def _prepare_predictive_inputs():
    """Put all inputs required by train.py in the aligned predictive folder."""
    required = [
        Path(paths.FEATURE_CACHE),
        Path(paths.SELECTED),
        Path(paths.HPARAMS),
    ]

    for p in required:
        _recover_existing_predictive_artifact(p)

    missing = [p for p in required if not p.exists()]

    if missing:
        lines = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "The following predictive artifacts are still missing:\n"
            + lines
            + "\nRun the corresponding feature/select/tune stage first."
        )



def load_training_data():
    _prepare_predictive_inputs()
    if not paths.FEATURE_CACHE.exists():
        raise FileNotFoundError(f"Missing feature cache: {paths.FEATURE_CACHE}")
    if not paths.HPARAMS.exists():
        raise FileNotFoundError(f"Missing tuned hyperparameters: {paths.HPARAMS}")
    if not paths.SELECTED.exists():
        raise FileNotFoundError(f"Missing selected features: {paths.SELECTED}")

    cache = joblib.load(paths.FEATURE_CACHE)
    hp = joblib.load(paths.HPARAMS)

    required = ["lgbm_params", "svr_params", "catboost_params"]
    missing = [key for key in required if key not in hp]
    if missing:
        raise RuntimeError(
            "best_hparams.pkl is missing required three-model parameters: "
            + ", ".join(missing)
        )

    idx, selected = selected_columns(cache)

    # Preserve the original replicate-quality rule.
    keep = np.asarray(cache["cov"], dtype=float) <= float(TARGET.max_cov)

    X_full = cache["X"].values if hasattr(cache["X"], "values") else np.asarray(cache["X"])
    X = finite_to_nan(X_full[keep][:, idx])

    y = np.asarray(cache["y"], dtype=float)[keep]
    y_model = np.asarray(transform.forward(y), dtype=float)
    groups = np.asarray(cache["groups"], dtype=object)[keep]

    return cache, hp, X, y, y_model, groups, selected


def make_lgbm(params):
    return LGBMRegressor(
        objective="regression",
        random_state=RS,
        n_jobs=-1,
        verbosity=-1,
        subsample_freq=1,
        **params,
    )


def make_svr(params):
    return SVR(kernel="rbf", cache_size=1000, **params)


def make_catboost(params):
    return CatBoostRegressor(
        loss_function="RMSE",
        random_seed=RS,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **params,
    )


def make_builders(hp):
    return {
        "LightGBM": lambda: make_lgbm(dict(hp["lgbm_params"])),
        "RBF-SVR": lambda: make_svr(dict(hp["svr_params"])),
        "CatBoost": lambda: make_catboost(dict(hp["catboost_params"])),
    }


def preprocess_fold(X_train_raw, X_test_raw):
    """
    Leakage-safe preprocessing.
    No row/feature dropping.
    """
    imputer = make_imputer()
    X_train = imputer.fit_transform(finite_to_nan(X_train_raw))
    X_test = imputer.transform(finite_to_nan(X_test_raw))

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def get_oof_predictions(X, y_model, groups, builders, split_type):
    if split_type == "scaffold":
        splitter = GroupKFold(n_splits=5)
        splits = list(splitter.split(X, y_model, groups))
    elif split_type == "random":
        splitter = KFold(n_splits=5, shuffle=True, random_state=RS)
        splits = list(splitter.split(X))
    else:
        raise ValueError("split_type must be 'scaffold' or 'random'")

    preds = {
        name: np.full(len(y_model), np.nan, dtype=float)
        for name in builders
    }

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"{split_type.capitalize()} fold {fold}/5")

        X_train, X_test = preprocess_fold(
            X[train_idx],
            X[test_idx],
        )

        for name, builder in builders.items():
            model = builder()
            model.fit(X_train, y_model[train_idx])
            preds[name][test_idx] = np.asarray(
                model.predict(X_test),
                dtype=float,
            ).reshape(-1)

    return preds


def scan_pair(y, scaffold_preds, random_preds, model_a, model_b, all_models):
    weights = np.round(
        np.arange(0.0, 1.0 + WEIGHT_STEP / 2.0, WEIGHT_STEP),
        10,
    )

    # Three-model disagreement uncertainty is independent of pair weight,
    # so compute it once.
    scaffold_all_ce = [
        np.asarray(transform.inverse(scaffold_preds[name]), dtype=float)
        for name in all_models
    ]
    random_all_ce = [
        np.asarray(transform.inverse(random_preds[name]), dtype=float)
        for name in all_models
    ]

    scaffold_unc = np.std(np.vstack(scaffold_all_ce), axis=0)
    random_unc = np.std(np.vstack(random_all_ce), axis=0)

    rows = []

    for weight_a in weights:
        weight_b = 1.0 - weight_a

        # Preserve the original workflow: blend BEFORE inverse transform.
        scaffold_blend_model = (
            weight_a * scaffold_preds[model_a]
            + weight_b * scaffold_preds[model_b]
        )
        random_blend_model = (
            weight_a * random_preds[model_a]
            + weight_b * random_preds[model_b]
        )

        scaffold_blend_ce = np.asarray(
            transform.inverse(scaffold_blend_model),
            dtype=float,
        )
        random_blend_ce = np.asarray(
            transform.inverse(random_blend_model),
            dtype=float,
        )

        scaffold_m = metrics(y, scaffold_blend_ce)
        random_m = metrics(y, random_blend_ce)

        combined = (
            SCAFFOLD_WEIGHT * scaffold_m["R2"]
            + RANDOM_WEIGHT * random_m["R2"]
        )

        scaffold_unc_rho = spearman(
            scaffold_unc,
            np.abs(y - scaffold_blend_ce),
        )
        random_unc_rho = spearman(
            random_unc,
            np.abs(y - random_blend_ce),
        )

        rows.append({
            "Model_A": model_a,
            "Model_B": model_b,
            "Weight_A": float(weight_a),
            "Weight_B": float(weight_b),
            "Scaffold_R2": scaffold_m["R2"],
            "Scaffold_RMSE": scaffold_m["RMSE"],
            "Scaffold_MAE": scaffold_m["MAE"],
            "Scaffold_Spearman": scaffold_m["Spearman"],
            "Random_R2": random_m["R2"],
            "Random_RMSE": random_m["RMSE"],
            "Random_MAE": random_m["MAE"],
            "Random_Spearman": random_m["Spearman"],
            "Combined_R2_Score": float(combined),
            "Scaffold_Uncertainty_Error_Spearman": scaffold_unc_rho,
            "Random_Uncertainty_Error_Spearman": random_unc_rho,
        })

    return pd.DataFrame(rows)



def ensemble_ce_from_oof(preds, role):
    """Return the selected weighted A+B OOF prediction on the original CE scale."""
    blend_model = (
        role["weight_a"] * preds[role["model_a"]]
        + role["weight_b"] * preds[role["model_b"]]
    )
    return np.asarray(transform.inverse(blend_model), dtype=float)


def disagreement_from_oof(preds, model_names):
    """Three-model disagreement SD on the original CE scale."""
    all_ce = [
        np.asarray(transform.inverse(preds[name]), dtype=float)
        for name in model_names
    ]
    return np.std(np.vstack(all_ce), axis=0)


def screening_metrics(y_true, ranking_score, threshold=HIT_THRESHOLD,
                      fraction=SCREEN_FRACTION):
    """Evaluate a ranking by high-CE hit recovery within a fixed screening budget."""
    y_true = np.asarray(y_true, dtype=float)
    ranking_score = np.asarray(ranking_score, dtype=float)
    hit = y_true >= float(threshold)
    n_hits = int(hit.sum())
    n = len(y_true)
    k = max(1, int(round(float(fraction) * n)))

    order = np.argsort(-ranking_score, kind="stable")
    found = int(hit[order[:k]].sum())
    recovery = float(found / n_hits) if n_hits else np.nan
    enrichment = float(recovery / fraction) if n_hits and fraction > 0 else np.nan
    ap = float(average_precision_score(hit.astype(int), ranking_score)) if n_hits else np.nan

    return {
        "N": int(n),
        "N_Hits": n_hits,
        "TopK": int(k),
        "Hits_Found": found,
        "TopK_Hit_Recovery": recovery,
        "TopK_Enrichment": enrichment,
        "Average_Precision": ap,
    }


def select_ranking_lambda(y, scaffold_pred_ce, scaffold_unc):
    """
    Select the disagreement penalty used for prospective ranking.

    Primary objective: maximize scaffold-OOF recovery of CE >= HIT_THRESHOLD hits
    within the top SCREEN_FRACTION of ranked molecules.
    Tie-breakers: higher average precision, then smaller lambda.

    Note: this chooses a screening/ranking hyperparameter from model-development OOF
    predictions. Any claimed improvement in prospective performance should still be
    confirmed on genuinely unseen generated candidates.
    """
    lambdas = np.round(
        np.arange(
            RANK_LAMBDA_MIN,
            RANK_LAMBDA_MAX + RANK_LAMBDA_STEP / 2.0,
            RANK_LAMBDA_STEP,
        ),
        10,
    )

    rows = []
    for lam in lambdas:
        score = scaffold_pred_ce - float(lam) * scaffold_unc
        m = screening_metrics(y, score)
        rows.append({
            "Lambda": float(lam),
            "Threshold_CE": float(HIT_THRESHOLD),
            "Screen_Fraction": float(SCREEN_FRACTION),
            **m,
        })

    scan = pd.DataFrame(rows)
    ranked = scan.sort_values(
        ["TopK_Hit_Recovery", "Average_Precision", "Lambda"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    if ranked.empty or not np.isfinite(ranked.iloc[0]["TopK_Hit_Recovery"]):
        best_lambda = 0.0
    else:
        best_lambda = float(ranked.iloc[0]["Lambda"])

    return best_lambda, scan


def fit_final_preprocessor(X):
    """
    Fit one production imputer and scaler on all retained training data.
    Keeps all feature columns.
    """
    imputer = make_imputer()
    X_imputed = imputer.fit_transform(finite_to_nan(X))

    scaler = StandardScaler().fit(X_imputed)
    X_scaled = scaler.transform(X_imputed)

    return imputer, scaler, X_scaled


def predict_selected(bundle, X_selected):
    """
    Predict from a matrix containing the exact selected features, in the same
    order stored in bundle['selected_features'].

    Returns:
        pred       : weighted A+B point prediction in CE space
        unc        : std across LightGBM/SVR/CatBoost CE predictions
        rank_score : disagreement-penalized score = pred - lambda * unc
    """
    X_selected = finite_to_nan(X_selected)

    X_imputed = bundle["imputer"].transform(X_selected)
    X_scaled = bundle["scaler"].transform(X_imputed)

    pred_model = {}
    pred_ce = {}

    for name, model in bundle["models"].items():
        p = np.asarray(model.predict(X_scaled), dtype=float).reshape(-1)
        pred_model[name] = p
        pred_ce[name] = np.asarray(transform.inverse(p), dtype=float)

    role = bundle["role"]

    blend_model = (
        role["weight_a"] * pred_model[role["model_a"]]
        + role["weight_b"] * pred_model[role["model_b"]]
    )
    pred = np.asarray(transform.inverse(blend_model), dtype=float)

    unc = np.std(
        np.vstack([
            pred_ce["LightGBM"],
            pred_ce["RBF-SVR"],
            pred_ce["CatBoost"],
        ]),
        axis=0,
    )

    ranking_cfg = bundle.get("ranking", {})
    rank_lambda = float(ranking_cfg.get("lambda", 0.0))
    rank_score = pred - rank_lambda * unc

    # Rank 1 = largest disagreement-penalized score. Stable sorting preserves
    # original input order for exact ties.
    order = np.argsort(-rank_score, kind="stable")
    rank = np.empty(len(rank_score), dtype=int)
    rank[order] = np.arange(1, len(rank_score) + 1)

    return {
        "pred": pred,
        "unc": unc,
        "rank_score": rank_score,
        "rank": rank,
        "rank_lambda": rank_lambda,
        "LightGBM": pred_ce["LightGBM"],
        "RBF-SVR": pred_ce["RBF-SVR"],
        "CatBoost": pred_ce["CatBoost"],
    }



def predict(bundle, smiles_list, context=None, compute_xtb=False, xtb_cache=None):
    """
    BatteryGen design.py-compatible prediction wrapper.

    Parameters
    ----------
    bundle : dict
        Production bundle returned by load_production().
    smiles_list : sequence[str]
        Candidate SMILES to score.
    context : dict or None
        Context values such as LogMolarRatio. Missing context values fall back
        to training medians stored in the production bundle.
    compute_xtb : bool
        False during the fast recursive screen: xTB-selected features are left
        missing and are handled by the production median imputer.
        True during final refinement: GFN2-xTB is calculated/cached and used.
    xtb_cache : dict or None
        Optional in-memory xTB cache supplied by design.py.

    Returns
    -------
    dict
        Mapping input SMILES -> {
            "pred": weighted Model-A/Model-B CE prediction,
            "unc": three-model disagreement SD in CE space,
            "domain_sim": max Morgan/Tanimoto similarity to retained training data,
            "xtb_homo": xTB HOMO when available,
            plus the individual model predictions and ranking fields.
        }

    Notes
    -----
    This function does not change the trained scientific model.  It only
    restores the public BatteryGen train.predict(...) interface expected by
    predictive/design.py.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    context = dict(context or {})
    smiles_list = list(smiles_list)

    selected = list(
        bundle.get("selected_features")
        or bundle.get("feature_order")
        or []
    )
    if not selected:
        raise RuntimeError(
            "Production bundle does not contain selected feature names."
        )

    feat_med = bundle.get("feat_med", {})
    if hasattr(feat_med, "to_dict"):
        feat_med = feat_med.to_dict()
    feat_med = dict(feat_med)

    # Context defaults are training medians. Explicit values passed by design.py
    # take precedence.
    context_cols = list(bundle.get("context_cols", []))
    resolved_context = {}
    for col in context_cols:
        value = context.get(col, feat_med.get(col, np.nan))
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = np.nan
        resolved_context[col] = value

    # RDKit descriptors for every valid candidate.
    rd = {}
    for smi in smiles_list:
        vals = features.rdkit_features(smi)
        if vals is not None:
            rd[smi] = vals

    # Optional real xTB refinement. During fast screening, xTB columns remain
    # NaN and the exact production imputer fitted in main() supplies training
    # medians. This preserves the two-stage BatteryGen design workflow.
    xtb_features = {}
    raw_xtb = {}
    if compute_xtb and bundle.get("use_xtb", TARGET.use_xtb):
        cache = xtb_cache if xtb_cache is not None else features.load_xtb_cache()
        cache = features.compute_xtb(list(rd.keys()), cache)
        for smi in rd:
            rec = cache.get(smi, {})
            raw_xtb[smi] = rec
            xtb_features[smi] = features.xtb_derived(rec)

    # Build the exact selected-feature matrix used by the fitted ensemble.
    valid_smiles = []
    rows = []
    for smi in smiles_list:
        if smi not in rd:
            continue

        feat = dict(rd[smi])

        if bundle.get("use_xtb", TARGET.use_xtb):
            if compute_xtb:
                feat.update(
                    xtb_features.get(
                        smi,
                        {c: np.nan for c in features.XTB_COLS},
                    )
                )
            else:
                # Fast screen: leave xTB unknown and let the saved production
                # SimpleImputer fill these from training data.
                feat.update({c: np.nan for c in features.XTB_COLS})

        feat.update(resolved_context)

        rows.append([feat.get(name, np.nan) for name in selected])
        valid_smiles.append(smi)

    if not rows:
        return {}

    pred_out = predict_selected(bundle, np.asarray(rows, dtype=float))

    # Domain similarity to the retained training set, matching BatteryGen's
    # Morgan radius=2, 2048-bit representation.
    train_fps = [fp for fp in bundle.get("train_fps", []) if fp is not None]

    results = {}
    for i, smi in enumerate(valid_smiles):
        mol = Chem.MolFromSmiles(smi)
        domain_sim = 0.0
        if mol is not None and train_fps:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            domain_sim = float(max(sims)) if sims else 0.0

        xtb_homo = np.nan
        if compute_xtb:
            rec = raw_xtb.get(smi, {})
            try:
                homo = rec.get("homo")
                if homo is not None and np.isfinite(float(homo)):
                    xtb_homo = float(homo)
            except (TypeError, ValueError):
                pass

        results[smi] = {
            "pred": float(pred_out["pred"][i]),
            "unc": float(pred_out["unc"][i]),
            "rank_score": float(pred_out["rank_score"][i]),
            "rank": int(pred_out["rank"][i]),
            "rank_lambda": float(pred_out["rank_lambda"]),
            "LightGBM": float(pred_out["LightGBM"][i]),
            "RBF-SVR": float(pred_out["RBF-SVR"][i]),
            "CatBoost": float(pred_out["CatBoost"][i]),
            "domain_sim": domain_sim,
            "xtb_homo": xtb_homo,
        }

    return results


def load_production(model_path=None):
    model_path = Path(model_path) if model_path is not None else PRODUCTION_MODEL
    model_path = model_path.expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            "BatteryGen production model was not found.\n"
            f"Expected location: {model_path}\n"
            "Run: python -m batterygen.predictive.train"
        )

    return joblib.load(model_path)


def main():
    cache, hp, X, y, y_model, groups, selected = load_training_data()
    builders = make_builders(hp)
    model_names = list(builders.keys())

    SCAN_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print("BATTERYGEN — THREE-MODEL CE ENSEMBLE")
    print("=" * 95)
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Artifact root       : {ARTIFACT_ROOT}")
    print(f"Predictive directory: {paths.PRED_DIR}")
    print("=" * 95)
    print(f"Rows              : {len(y)}")
    print(f"Selected features : {len(selected)}")
    print(f"Scaffold groups   : {len(set(groups))}")
    print(f"Missing cells     : {int(np.isnan(X).sum())}")
    print(f"Objective         : {SCAFFOLD_WEIGHT:.2f} scaffold R2 + "
          f"{RANDOM_WEIGHT:.2f} random R2")

    print("\nGenerating scaffold OOF predictions...")
    scaffold_preds = get_oof_predictions(
        X, y_model, groups, builders, "scaffold"
    )

    print("\nGenerating random OOF predictions...")
    random_preds = get_oof_predictions(
        X, y_model, groups, builders, "random"
    )

    # Individual tuned models
    individual_rows = []
    for name in model_names:
        scaffold_ce = np.asarray(
            transform.inverse(scaffold_preds[name]),
            dtype=float,
        )
        random_ce = np.asarray(
            transform.inverse(random_preds[name]),
            dtype=float,
        )

        sm = metrics(y, scaffold_ce)
        rm = metrics(y, random_ce)

        individual_rows.append({
            "Model": name,
            "Scaffold_R2": sm["R2"],
            "Scaffold_RMSE": sm["RMSE"],
            "Scaffold_MAE": sm["MAE"],
            "Random_R2": rm["R2"],
            "Random_RMSE": rm["RMSE"],
            "Random_MAE": rm["MAE"],
        })

    individual_df = pd.DataFrame(individual_rows)

    # Scan all three possible pairs.
    all_scans = []
    for model_a, model_b in combinations(model_names, 2):
        print(f"\nScanning {model_a} + {model_b}...")
        pair_df = scan_pair(
            y,
            scaffold_preds,
            random_preds,
            model_a,
            model_b,
            model_names,
        )
        all_scans.append(pair_df)

    full_scan = pd.concat(all_scans, ignore_index=True)

    # Best weight within each model pair.
    best_rows = []
    for (model_a, model_b), group in full_scan.groupby(
        ["Model_A", "Model_B"],
        sort=False,
    ):
        best_rows.append(
            group.sort_values(
                ["Combined_R2_Score", "Scaffold_R2", "Scaffold_RMSE"],
                ascending=[False, False, True],
            ).iloc[0]
        )

    best_pairs = (
        pd.DataFrame(best_rows)
        .sort_values(
            ["Combined_R2_Score", "Scaffold_R2"],
            ascending=[False, False],
        )
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
        "weight_b": float(winner["Weight_B"]),
    }

    # ---------------------------------------------------------------
    # Select a disagreement-aware ranking penalty using scaffold OOF.
    # This does NOT change the CE point prediction; it only changes the
    # order in which prospective candidates should be prioritized.
    # ---------------------------------------------------------------
    scaffold_ensemble_ce = ensemble_ce_from_oof(scaffold_preds, role)
    scaffold_unc = disagreement_from_oof(scaffold_preds, model_names)
    rank_lambda, rank_scan = select_ranking_lambda(
        y, scaffold_ensemble_ce, scaffold_unc
    )

    baseline_rank = screening_metrics(y, scaffold_ensemble_ce)
    penalized_rank = screening_metrics(
        y, scaffold_ensemble_ce - rank_lambda * scaffold_unc
    )

    ranking = {
        "method": "predicted_CE_minus_lambda_times_three_model_disagreement_SD",
        "lambda": float(rank_lambda),
        "hit_threshold_ce": float(HIT_THRESHOLD),
        "screen_fraction": float(SCREEN_FRACTION),
        "lambda_search_min": float(RANK_LAMBDA_MIN),
        "lambda_search_max": float(RANK_LAMBDA_MAX),
        "lambda_search_step": float(RANK_LAMBDA_STEP),
        "selection_primary": "scaffold_OOF_topK_hit_recovery",
        "selection_tiebreaker": "average_precision_then_smaller_lambda",
        "sd_interpretation": "inter_model_disagreement_not_calibrated_probability",
        "baseline_topK_hit_recovery": baseline_rank["TopK_Hit_Recovery"],
        "penalized_topK_hit_recovery": penalized_rank["TopK_Hit_Recovery"],
        "baseline_average_precision": baseline_rank["Average_Precision"],
        "penalized_average_precision": penalized_rank["Average_Precision"],
    }

    # Save all evaluation/selection outputs.
    individual_df.to_csv(
        SCAN_DIR / "01_individual_model_metrics.csv",
        index=False,
    )
    full_scan.to_csv(
        SCAN_DIR / "02_all_pair_weight_scans.csv",
        index=False,
    )
    best_pairs.to_csv(
        SCAN_DIR / "03_best_pair_by_combined_objective.csv",
        index=False,
    )

    with open(
        SCAN_DIR / "04_role_assignment.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(role, f, indent=2)

    rank_scan.to_csv(
        SCAN_DIR / "05_ranking_lambda_scan.csv",
        index=False,
    )
    with open(
        SCAN_DIR / "06_ranking_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ranking, f, indent=2)

    # Final production preprocessing + final model fits.
    print("\nTraining final models on all retained data...")
    imputer, scaler, X_scaled = fit_final_preprocessor(X)

    final_models = {}
    for name, builder in builders.items():
        model = builder()
        model.fit(X_scaled, y_model)
        final_models[name] = model

    # Production bundle: keep the complete three-model ensemble information
    # and restore the metadata/interface expected by BatteryGen design.py.
    keep_mask = np.asarray(cache["cov"], dtype=float) <= float(TARGET.max_cov)

    all_train_fps = list(cache.get("train_fps", []))
    retained_train_fps = [
        fp for fp, keep_row in zip(all_train_fps, keep_mask)
        if keep_row and fp is not None
    ]

    all_train_smiles = list(cache.get("train_smiles", []))
    retained_train_smiles = [
        smi for smi, keep_row in zip(all_train_smiles, keep_mask)
        if keep_row
    ]

    feat_med = cache.get("feat_med", {})
    if hasattr(feat_med, "to_dict"):
        feat_med = feat_med.to_dict()
    else:
        feat_med = dict(feat_med)

    bundle = {
        "architecture": "LightGBM + RBF-SVR + CatBoost",
        "imputer": imputer,
        "scaler": scaler,
        "models": final_models,
        "role": role,
        "ranking": ranking,

        # Exact model inputs.
        "selected_features": list(selected),
        # BatteryGen design.py historically calls this feature_order and uses
        # its length as the number of selected production features.
        "feature_order": list(selected),
        "full_feature_order": list(cache["feature_order"]),

        # Metadata required by BatteryGen design.py.
        "feat_med": feat_med,
        "context_cols": list(TARGET.context_cols),
        "target_name": TARGET.target_name,
        "target_col": TARGET.target_col,
        "use_xtb": bool(cache.get("use_xtb", TARGET.use_xtb)),
        "train_fps": retained_train_fps,
        "train_smiles": retained_train_smiles,
        "y_range": (float(np.min(y)), float(np.max(y))),
        "scaffold_R2": float(winner["Scaffold_R2"]),
        "random_R2": float(winner["Random_R2"]),

        # Reproducibility / interpretation.
        "max_cov": float(TARGET.max_cov),
        "uncertainty": "std_of_three_model_CE_predictions",
        "ranking_score": "predicted_CE - lambda * three_model_disagreement_SD",
        "blend_space": "transformed_target_space",
        "missing_value_policy": "median_imputation_keep_empty_features",
        "random_state": RS,
    }

    paths.PRED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, PRODUCTION_MODEL)

    print("\n" + "=" * 95)
    print("FINAL ROLE ASSIGNMENT")
    print("=" * 95)
    print(f"Model A             : {role['model_a']}")
    print(f"Model B             : {role['model_b']}")
    print(f"Weight A            : {role['weight_a']:.2f}")
    print(f"Weight B            : {role['weight_b']:.2f}")
    print(f"Model C             : {role['model_c']}")
    print(f"Scaffold R2         : {winner['Scaffold_R2']:.4f}")
    print(f"Scaffold RMSE       : {winner['Scaffold_RMSE']:.4f}")
    print(f"Scaffold MAE        : {winner['Scaffold_MAE']:.4f}")
    print(f"Random R2           : {winner['Random_R2']:.4f}")
    print(f"Combined R2 score   : {winner['Combined_R2_Score']:.4f}")
    print(f"Uncertainty rho     : "
          f"{winner['Scaffold_Uncertainty_Error_Spearman']:.4f}")
    print("\nRANKING CONFIGURATION")
    print(f"Lambda              : {rank_lambda:.2f}")
    print(f"Ranking score       : Predicted_CE - {rank_lambda:.2f} * SD")
    print(f"Top {SCREEN_FRACTION:.0%} recovery, CE-only : "
          f"{baseline_rank['TopK_Hit_Recovery']:.3f}")
    print(f"Top {SCREEN_FRACTION:.0%} recovery, penalized: "
          f"{penalized_rank['TopK_Hit_Recovery']:.3f}")
    print(f"Average precision, CE-only : "
          f"{baseline_rank['Average_Precision']:.3f}")
    print(f"Average precision, penalized: "
          f"{penalized_rank['Average_Precision']:.3f}")
    print("\n" + "=" * 95)
    print("TRAINING COMPLETE — FILES SAVED")
    print("=" * 95)
    print(f"Production model : {PRODUCTION_MODEL.resolve()}")
    print(f"Model exists     : {PRODUCTION_MODEL.exists()}")
    print(f"Scan outputs     : {SCAN_DIR.resolve()}")
    print("=" * 95)


if __name__ == "__main__":
    main()
