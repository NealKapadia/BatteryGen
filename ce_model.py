"""Coulombic-Efficiency (CE) predictor + inverse design for battery additives.

Pipeline:
  Additive_SMILES --(featurizer)--> molecular vector  ⊕  [Zn_mole, Additive_mole,
  LogMolarRatio]  -->  regressor  -->  CE_aver (%)

Featurizer is pluggable (--features):
  ecfp    Morgan/ECFP4 fingerprint (1024 bits)         [default block]
  rdkit   the 11 RDKit descriptors (MolWt, LogP, ...)   [default block]
  latent  the VAE 256-d latent (shared with the generator)
default = "ecfp+rdkit". ECFP/RDKit beat the drug-like-trained VAE latent on these
small OOD additives. The VAE is still the GENERATOR for inverse design; the predictor
just scores decoded candidates with its own features — they need not share a space.

Modes:
  train    fit + model-select (Ridge/RF/HGB), report random-split R^2 (the target) AND
           grouped-by-molecule R^2 (honest generalization). Auto-Optuna if below --target.
  tune     force Optuna on the best model family.
  screen   GLOBAL inverse design: sample a large diverse valid pool from the VAE prior,
           predict CE, return top-ranked novel additives.
  suggest  LOCAL inverse design: seed from best known performers, sample latent neighbors.
  predict  CE for explicit SMILES (+ optional concentration context).

Examples:
  python -m molforge.ce_model --mode train          # dataset auto-detected from data/
  python -m molforge.ce_model --mode train --csv path/to/your.csv
  python -m molforge.ce_model --mode screen --n 5000 --top 25 --out hits.csv
  python -m molforge.ce_model --mode suggest --top 25
  python molvae/ce_model.py --mode predict --smiles "OCCN(CCO)CCO"
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import config
import data
import infer

CE_MODEL_PATH = config.CKPT_DIR / "ce_model.pkl"
LATENT_DIM = config.LATENT_DIM
ECFP_BITS = 1024


def _resolve_ckpt(ckpt):
    """Default to the validated base (best.pt), not latest.pt (may be mid-continued-train)."""
    return ckpt or str(config.CKPT_DIR / "best.pt")
COND_COLS = ["Zn_mole (mmol)", "Additive_mole (%)", "LogMolarRatio"]
SMILES_COL = "Additive_SMILES"
TARGET_COL = "CE_aver. (%)"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_csv(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            canon = data.canonical_smiles((r.get(SMILES_COL) or "").strip())
            if not canon:
                continue
            try:
                cond = [float(r[c]) for c in COND_COLS]
                y = float(r[TARGET_COL])
            except (KeyError, ValueError):
                continue
            rows.append({"smiles": canon, "cond": cond, "ce": y,
                         "name": (r.get("IUPAC_NAME") or "").strip()})
    if not rows:
        raise SystemExit(f"No usable rows in {path} (need {SMILES_COL}, {COND_COLS}, {TARGET_COL}).")
    return rows


# --------------------------------------------------------------------------- #
# Featurizers (each returns {canonical_smiles: vector})
# --------------------------------------------------------------------------- #
def _feat_ecfp(smiles: List[str]) -> Dict[str, np.ndarray]:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    out = {}
    for s in dict.fromkeys(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(m, 2, ECFP_BITS)
        arr = np.zeros(ECFP_BITS, np.float32); DataStructs.ConvertToNumpyArray(bv, arr)
        out[s] = arr
    return out


def _feat_rdkit(smiles: List[str]) -> Dict[str, np.ndarray]:
    out = {}
    for s in dict.fromkeys(smiles):
        v = data.descriptors_for_smiles(s)
        if v is not None:
            out[s] = np.asarray(v, np.float32)
    return out


@torch.no_grad()
def _feat_latent(smiles: List[str], net, vocab, device, batch=256) -> Dict[str, np.ndarray]:
    import selfies as sf

    out, ids_buf, len_buf, smi_buf = {}, [], [], []

    def flush():
        if not ids_buf:
            return
        mu, _ = net.encode(torch.tensor(ids_buf, dtype=torch.long, device=device),
                           torch.tensor(len_buf))
        for s, m in zip(smi_buf, mu.cpu().numpy()):
            out[s] = m
        ids_buf.clear(); len_buf.clear(); smi_buf.clear()

    for s in dict.fromkeys(smiles):
        try:
            sel = sf.encoder(s)
        except Exception:
            continue
        if not sel or sf.len_selfies(sel) > config.MAX_SELFIES_LEN - 2:
            continue
        ids, length = vocab.encode(sel)
        ids_buf.append(ids); len_buf.append(length); smi_buf.append(s)
        if len(ids_buf) >= batch:
            flush()
    flush()
    return out


def featurize(smiles: List[str], kinds: List[str], net=None, vocab=None, device=None
              ) -> Dict[str, np.ndarray]:
    """Concatenate the requested feature blocks (fixed order) per molecule."""
    blocks = {}
    if "ecfp" in kinds:
        blocks["ecfp"] = _feat_ecfp(smiles)
    if "rdkit" in kinds:
        blocks["rdkit"] = _feat_rdkit(smiles)
    if "latent" in kinds:
        blocks["latent"] = _feat_latent(smiles, net, vocab, device)
    out = {}
    for s in dict.fromkeys(smiles):
        try:
            out[s] = np.concatenate([blocks[k][s] for k in kinds])
        except KeyError:
            continue  # a featurizer dropped this molecule
    return out


_BAD_SMARTS = None
_OK_ANION_SMARTS = None


def chem_ok(smi: str) -> bool:
    """Reject obviously unstable/reactive or non-viable candidates so the output is testable
    additive chemistry. Drops acyl halides, N-halamines, peroxides, hydrazines (N-N),
    allenes/ketenes/diazo, strained 3-rings, exotic atoms, bare cations, and charged species
    that are NOT a real anionic-additive class (carboxylate/sulfonate/sulfonimide) — e.g.
    bare alkoxide/amide dianions like [O-]C[O-] that the model over-scores but are just
    strong bases that protonate in water."""
    from rdkit import Chem

    global _BAD_SMARTS, _OK_ANION_SMARTS
    if _BAD_SMARTS is None:
        pats = ["[CX3](=O)[F,Cl,Br,I]", "[#7][F,Cl,Br,I]", "[OX2][OX2]", "[#7]-[#7]",
                "[#6]=[#6]=[#6]", "[#6]=[#6]=[#8]", "*=[N+]=[N-]", "[r3]", "[#6]=[#6]=[#7]"]
        _BAD_SMARTS = [Chem.MolFromSmarts(p) for p in pats]
        _OK_ANION_SMARTS = [Chem.MolFromSmarts(p) for p in
                            ("[CX3](=O)[O-]", "[SX4](=O)(=O)[O-]", "[#7-]S(=O)(=O)", "[n-]")]
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    if any(a.GetSymbol() not in config.ALLOWED_ATOMS for a in m.GetAtoms()):
        return False
    chg = Chem.GetFormalCharge(m)
    if chg > 0:
        return False  # bare cations aren't neutral/anionic additives
    if chg < 0 and not any(m.HasSubstructMatch(p) for p in _OK_ANION_SMARTS):
        return False  # anion but not a carboxylate/sulfonate/sulfonimide -> artifact
    return not any(p is not None and m.HasSubstructMatch(p) for p in _BAD_SMARTS)


def build_xy(rows, feats) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, groups = [], [], []
    for r in rows:
        if r["smiles"] not in feats:
            continue
        X.append(np.concatenate([feats[r["smiles"]], np.asarray(r["cond"], np.float32)]))
        y.append(r["ce"]); groups.append(r["smiles"])
    return np.asarray(X, np.float32), np.asarray(y, np.float32), groups


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _make_model(kind: str, params: dict):
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "ridge":
        return make_pipeline(StandardScaler(with_mean=False), Ridge(**params))
    if kind == "rf":
        return RandomForestRegressor(random_state=0, n_jobs=-1, **params)
    if kind == "hgb":
        return HistGradientBoostingRegressor(random_state=0, **params)
    raise ValueError(kind)


_DEFAULTS = {
    "ridge": {"alpha": 5.0},
    "rf": {"n_estimators": 600, "max_features": 0.2, "min_samples_leaf": 2},
    "hgb": {"learning_rate": 0.05, "max_iter": 600, "l2_regularization": 1.0, "max_leaf_nodes": 31},
}


def _metrics(yt, yp) -> dict:
    from sklearn.metrics import r2_score, mean_absolute_error
    return {"r2": round(float(r2_score(yt, yp)), 4),
            "mae": round(float(mean_absolute_error(yt, yp)), 3),
            "rmse": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 3)}


def grouped_cv_r2(kind, params, X, y, groups, n_splits=5) -> float:
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score

    n_splits = min(n_splits, len(set(groups)))
    if n_splits < 2:
        return float("nan")
    preds = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, np.array(groups)):
        m = _make_model(kind, params); m.fit(X[tr], y[tr]); preds[te] = m.predict(X[te])
    return round(float(r2_score(y, preds)), 4)


# --------------------------------------------------------------------------- #
# Optuna (random KFold objective = matches the random-split target)
# --------------------------------------------------------------------------- #
def _ensure_optuna():
    try:
        import optuna  # noqa
    except ImportError:
        print("[ce] installing optuna ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna"], check=True)


def tune(kind, X, y, n_trials=60) -> dict:
    _ensure_optuna()
    import optuna
    from sklearn.model_selection import KFold, cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    def objective(trial):
        if kind == "hgb":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_iter": trial.suggest_int("max_iter", 200, 1200),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128),
                "max_depth": trial.suggest_int("max_depth", 2, 16),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 30),
            }
        elif kind == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
                "max_depth": trial.suggest_int("max_depth", 4, 40),
                "max_features": trial.suggest_float("max_features", 0.05, 0.8),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            }
        else:
            params = {"alpha": trial.suggest_float("alpha", 1e-2, 100.0, log=True)}
        return float(np.mean(cross_val_score(_make_model(kind, params), X, y, cv=cv,
                                             scoring="r2", n_jobs=-1)))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[ce] optuna best 5-fold CV R^2 = {study.best_value:.4f}")
    return study.best_params


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def _need_vae(kinds, mode_generates):
    return ("latent" in kinds) or mode_generates


def train(args):
    import joblib
    from sklearn.model_selection import train_test_split

    kinds = args.features.split("+")
    rows = load_csv(Path(args.csv))
    net = vocab = device = None
    if _need_vae(kinds, False):
        net, vocab, _, device = infer.load_model(_resolve_ckpt(args.ckpt), device=args.device)
    feats = featurize([r["smiles"] for r in rows], kinds, net, vocab, device)
    X, y, groups = build_xy(rows, feats)
    print(f"rows={len(rows)} unique={len(set(groups))} | features={args.features} "
          f"-> X{X.shape}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    print("\n== model selection (random train/test split) ==")
    print(f"{'model':>8}{'test_R2':>10}{'test_MAE':>10}{'grouped_R2':>12}")
    results = {}
    for kind in ("ridge", "rf", "hgb"):
        m = _make_model(kind, _DEFAULTS[kind]); m.fit(Xtr, ytr)
        met = _metrics(yte, m.predict(Xte))
        gcv = grouped_cv_r2(kind, _DEFAULTS[kind], X, y, groups)
        results[kind] = (met, gcv)
        print(f"{kind:>8}{met['r2']:>10}{met['mae']:>10}{gcv:>12}")

    # Deploy by GROUPED R^2 — the inverse-design loop scores NOVEL molecules, so
    # generalization to unseen chemistry (not the leakage-inflated random split) is
    # what counts. (--select test to optimize the random-split number instead.)
    if args.select == "grouped":
        best_kind = max(results, key=lambda k: results[k][1])
    else:
        best_kind = max(results, key=lambda k: results[k][0]["r2"])
    best_params = dict(_DEFAULTS[best_kind])
    best_r2 = results[best_kind][0]["r2"]
    print(f"\nselected '{best_kind}' by {args.select} R^2 | test R^2={best_r2} "
          f"grouped R^2={results[best_kind][1]} | target={args.target}")

    if args.tune or best_r2 < args.target:
        print(f"[ce] Optuna on '{best_kind}' "
              f"({'forced' if args.tune else f'{best_r2} < {args.target}'}) ...")
        best_params = tune(best_kind, X, y, n_trials=args.trials)
        m = _make_model(best_kind, best_params); m.fit(Xtr, ytr)
        met = _metrics(yte, m.predict(Xte))
        gcv = grouped_cv_r2(best_kind, best_params, X, y, groups)
        print(f"[ce] tuned {best_kind}: test R^2={met['r2']} MAE={met['mae']} | grouped R^2={gcv}")
        if met["r2"] >= results[best_kind][0]["r2"]:
            results[best_kind] = (met, gcv)
        else:
            print("[ce] tuned worse than default; keeping defaults.")
            best_params = dict(_DEFAULTS[best_kind])
        best_r2 = results[best_kind][0]["r2"]

    final = _make_model(best_kind, best_params); final.fit(X, y)
    bundle = {
        "model": final, "kind": best_kind, "params": best_params, "features": kinds,
        "ckpt": _resolve_ckpt(args.ckpt),
        "cond_cols": COND_COLS, "ecfp_bits": ECFP_BITS,
        "cond_median": np.median(np.array([r["cond"] for r in rows], np.float32), axis=0).tolist(),
        "metrics": {"test": results[best_kind][0], "grouped_r2": results[best_kind][1]},
        "y_range": [float(np.min(y)), float(np.max(y))], "n_rows": len(rows), "n_unique": len(set(groups)),
    }
    joblib.dump(bundle, CE_MODEL_PATH)
    print(f"\nSaved CE model -> {CE_MODEL_PATH}")
    print(f"  test R^2 = {best_r2}  (target {args.target}: {'MET' if best_r2 >= args.target else 'NOT met'})"
          f"  | grouped (unseen-molecule) R^2 = {results[best_kind][1]}")
    return bundle


# --------------------------------------------------------------------------- #
# Inference / inverse design
# --------------------------------------------------------------------------- #
def _load_bundle():
    import joblib
    if not CE_MODEL_PATH.exists():
        raise SystemExit("No CE model. Train it first: python -m molforge.ce_model --mode train "
                         "(put your CE dataset CSV in data/, or pass --csv / set $MOLVAE_CE_CSV)")
    return joblib.load(CE_MODEL_PATH)


def _predict_ce(bundle, smiles, cond, net=None, vocab=None, device=None) -> Dict[str, float]:
    feats = featurize(smiles, bundle["features"], net, vocab, device)
    cond = np.asarray(cond, np.float32)
    keys = [s for s in dict.fromkeys(smiles) if s in feats]
    if not keys:
        return {}
    X = np.asarray([np.concatenate([feats[s], cond]) for s in keys], np.float32)
    preds = bundle["model"].predict(X)
    return {k: float(v) for k, v in zip(keys, preds)}


@torch.no_grad()
def _generate(net, vocab, device, n, z_scale, temperature, seed_latents=None, spread=0.6):
    # validity is ~1.0 (default SELFIES constraints), so generate only ~1.3x the shortfall
    # instead of the old 4x — the autoregressive decode is the cost, especially on CPU.
    smis, tries = [], 0
    while len(smis) < n and tries < 6:
        tries += 1
        b = min(512, max(8, int((n - len(smis)) * 1.3)))
        cond = torch.zeros(b, config.N_PROPS, device=device)
        if seed_latents is not None and len(seed_latents):
            idx = torch.randint(0, seed_latents.size(0), (b,), device=device)
            z = seed_latents[idx] + spread * torch.randn(b, net.latent_dim, device=device)
            seqs = net.sample(b, cond, vocab.bos, vocab.eos, z=z, temperature=temperature, device=device)
        else:
            seqs = net.sample(b, cond, vocab.bos, vocab.eos, temperature=temperature,
                              z_scale=z_scale, device=device)
        smis.extend(filter(None, (vocab.decode_to_smiles(s) for s in seqs.tolist())))
    return smis


def _cond_context(bundle, args) -> List[float]:
    cond = list(bundle["cond_median"])
    ov = {"Zn_mole (mmol)": args.zn_mole, "Additive_mole (%)": args.additive_mole,
          "LogMolarRatio": args.log_ratio}
    for i, c in enumerate(bundle["cond_cols"]):
        if ov.get(c) is not None:
            cond[i] = float(ov[c])
    return cond


def screen(args, seeded=False):
    from membership import MolportIndex

    bundle = _load_bundle()
    net, vocab, _, device = infer.load_model(_resolve_ckpt(args.ckpt or bundle.get("ckpt")),
                                             device=args.device)
    cond = _cond_context(bundle, args)
    tm = bundle["metrics"]
    print(f"CE model: {bundle['kind']} [{'+'.join(bundle['features'])}] "
          f"test R^2 {tm['test']['r2']}, grouped R^2 {tm['grouped_r2']} | "
          f"conc {dict(zip(bundle['cond_cols'], [round(c,4) for c in cond]))}")

    seed_latents, known = None, set()
    if seeded:
        rows = sorted(load_csv(Path(args.csv)), key=lambda r: r["ce"], reverse=True)
        known = {r["smiles"] for r in rows}
        top = rows[:args.seed_k]
        lat = _feat_latent([r["smiles"] for r in top], net, vocab, device)
        seed_latents = torch.tensor(np.array([lat[r["smiles"]] for r in top if r["smiles"] in lat]),
                                    device=device)
        print(f"Seeding from top {len(seed_latents)} performers "
              f"(CE {top[-1]['ce']:.1f}-{top[0]['ce']:.1f}%): e.g. {top[0]['name'] or top[0]['smiles']}")
    else:
        if args.csv and Path(args.csv).exists():
            known = {r["smiles"] for r in load_csv(Path(args.csv))}
        print(f"Global prior exploration: generating ~{args.n} candidates ...")

    pool = _generate(net, vocab, device, args.n, args.z_scale, args.temperature,
                     seed_latents=seed_latents, spread=args.spread)
    pool = [s for s in dict.fromkeys(pool)
            if s and s not in known and (args.no_filter or chem_ok(s))]
    print(f"  {len(pool)} unique novel candidates")

    preds = _predict_ce(bundle, pool, cond, net, vocab, device)
    ranked = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)[:args.top]
    index = MolportIndex()
    lo, hi = bundle["y_range"]
    print(f"\nTop {len(ranked)} predicted-CE additives (trained CE {lo:.1f}-{hi:.1f}%):")
    print(f"{'rank':>4}  {'pred_CE%':>8}  {'Molport':>9}  SMILES")
    out_rows = []
    for i, (smi, ce) in enumerate(ranked, 1):
        mid = index.get_id(smi, already_canonical=True) if index.available else None
        ce_c = max(lo, min(hi, ce))
        print(f"{i:>4}  {ce_c:>8.2f}  {mid or '-':>9}  {smi}")
        out_rows.append({"rank": i, "pred_ce": round(ce_c, 2), "pred_ce_raw": round(ce, 2),
                         "molport_id": mid or "", "smiles": smi})
    if index.available:
        index.close()
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["rank", "pred_ce", "pred_ce_raw", "molport_id", "smiles"])
            w.writeheader(); w.writerows(out_rows)
        print(f"\nSaved -> {args.out}")
    print("\nNote: predicted CE extrapolates beyond the training additives — use as a "
          "ranking/triage tool, validate top hits, feed results back and retrain (active learning).")
    return out_rows


def design(args):
    """Iterative inverse design (the full loop):
      round 0  : generate a global pool from the VAE prior, score with the CE model
      round 1+ : re-seed generation around the BEST-SCORED candidates so far, score, merge
      output   : top-N predicted-CE novel molecules to test.
    """
    from membership import MolportIndex

    bundle = _load_bundle()
    net, vocab, _, device = infer.load_model(_resolve_ckpt(args.ckpt or bundle.get("ckpt")),
                                             device=args.device)
    cond = _cond_context(bundle, args)
    tm = bundle["metrics"]
    known = set()
    if args.csv and Path(args.csv).exists():
        known = {r["smiles"] for r in load_csv(Path(args.csv))}
    print(f"CE model: {bundle['kind']} [{'+'.join(bundle['features'])}] test R^2 {tm['test']['r2']}, "
          f"grouped R^2 {tm['grouped_r2']} | device {device}")
    print(f"conc context: {dict(zip(bundle['cond_cols'], [round(c,4) for c in cond]))}")
    lo, hi = bundle["y_range"]

    scored: Dict[str, float] = {}

    def add(pool):
        pool = [s for s in dict.fromkeys(pool)
                if s and s not in known and s not in scored and (args.no_filter or chem_ok(s))]
        if not pool:
            return 0, pool
        scored.update(_predict_ce(bundle, pool, cond, net, vocab, device))
        return len(pool), pool

    # round 0: global exploration
    n0, _ = add(_generate(net, vocab, device, args.n, args.z_scale, args.temperature))
    best0 = max(scored.values()) if scored else float("nan")
    print(f"\nround 0 (global prior)  : generated {n0:5d} | best predicted CE {best0:.2f}%")

    # rounds 1..R: exploit around best-scored so far
    for rnd in range(1, args.rounds + 1):
        top = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:args.seed_k]
        lat = _feat_latent([s for s, _ in top], net, vocab, device)
        seeds = torch.tensor(np.array([lat[s] for s, _ in top if s in lat]), device=device)
        n_new, _ = add(_generate(net, vocab, device, args.n, args.z_scale, args.temperature,
                                 seed_latents=seeds, spread=args.spread))
        print(f"round {rnd} (seed top {len(seeds):2d})    : generated {n_new:5d} | "
              f"best predicted CE {max(scored.values()):.2f}% | pool {len(scored)}")

    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:args.top]
    index = MolportIndex()
    print(f"\n=== TOP {len(ranked)} candidate additives to test (trained CE {lo:.1f}-{hi:.1f}%) ===")
    print(f"{'rank':>4}  {'pred_CE%':>8}  {'Molport':>9}  SMILES")
    out_rows = []
    for i, (smi, ce) in enumerate(ranked, 1):
        mid = index.get_id(smi, already_canonical=True) if index.available else None
        ce_c = max(lo, min(hi, ce))
        print(f"{i:>4}  {ce_c:>8.2f}  {mid or '-':>9}  {smi}")
        out_rows.append({"rank": i, "pred_ce": round(ce_c, 2), "pred_ce_raw": round(ce, 2),
                         "molport_id": mid or "", "smiles": smi})
    if index.available:
        index.close()
    out = args.out or "ce_candidates.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "pred_ce", "pred_ce_raw", "molport_id", "smiles"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nSaved {len(out_rows)} candidates -> {out}")
    print("Note: predicted CE extrapolates beyond the 168 training additives -> rank/triage, "
          "not absolute %. Validate top hits, add results to the CSV, retrain (active learning).")
    return out_rows


def predict_cmd(args):
    bundle = _load_bundle()
    net = vocab = device = None
    if "latent" in bundle["features"]:
        net, vocab, _, device = infer.load_model(_resolve_ckpt(args.ckpt or bundle.get("ckpt")),
                                                 device=args.device)
    cond = _cond_context(bundle, args)
    smis = [data.canonical_smiles(s.strip()) for s in args.smiles.split(",") if s.strip()]
    smis = [s for s in smis if s]
    preds = _predict_ce(bundle, smis, cond, net, vocab, device)
    print(f"conc context: {dict(zip(bundle['cond_cols'], [round(c,4) for c in cond]))}")
    for s in smis:
        print(f"  {s:42s} -> CE {preds[s]:.2f}%" if s in preds else f"  {s:42s} -> (unencodable)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["train", "tune", "screen", "suggest", "design", "predict"],
                    default="train")
    ap.add_argument("--csv", default=None,
                    help="CE dataset CSV (default: auto-detect from data/ or $MOLVAE_CE_CSV)")
    ap.add_argument("--features", default="ecfp+rdkit", help="ecfp / rdkit / latent, '+'-joined")
    ap.add_argument("--ckpt", default=None, help="VAE checkpoint (generator + latent featurizer)")
    ap.add_argument("--device", default="cpu", help="cpu (default; keeps GPU free) or cuda")
    ap.add_argument("--rounds", type=int, default=3, help="design mode: refinement rounds")
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--select", choices=["grouped", "test"], default="grouped",
                    help="deploy the model best by grouped (novel-molecule) or test (random-split) R^2")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--seed-k", type=int, default=10)
    ap.add_argument("--z-scale", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--spread", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--zn-mole", type=float, default=None)
    ap.add_argument("--additive-mole", type=float, default=None)
    ap.add_argument("--log-ratio", type=float, default=None)
    ap.add_argument("--smiles", default=None)
    ap.add_argument("--no-filter", action="store_true",
                    help="disable the unstable/reactive-chemistry filter on candidates")
    args = ap.parse_args()
    args.csv = config.resolve_ce_csv_optional(args.csv)
    if args.mode in ("train", "tune", "suggest") and args.csv is None:
        raise SystemExit(
            f"'{args.mode}' needs a CE dataset. Put a single CSV in a 'data/' folder, "
            "pass --csv path/to/your.csv, or set $MOLVAE_CE_CSV."
        )

    if args.mode in ("train", "tune"):
        if args.mode == "tune":
            args.tune = True
        train(args)
    elif args.mode == "screen":
        screen(args, seeded=False)
    elif args.mode == "suggest":
        screen(args, seeded=True)
    elif args.mode == "design":
        design(args)
    elif args.mode == "predict":
        if not args.smiles:
            raise SystemExit("--smiles required for predict mode")
        predict_cmd(args)


if __name__ == "__main__":
    main()
