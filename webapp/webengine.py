"""
webengine.py — bring-your-own-data inverse design for the web app.
=================================================================
Sits on top of the VAE generator (design.DesignEngine) and adds, all IN MEMORY
(user CSVs are never written to disk — only ephemeral session models are kept):

  * train_from_csv(bytes, objective)   parse objective (LLM) -> target column(s)+direction,
                                       train a fast ECFP+RDKit predictor per target, return
                                       a session id + per-target CV R^2.
  * generate(prompt, n, session_id)    sample valid molecules from the VAE; if a session model
                                       exists, score+rank by the user's objective; enrich each
                                       with RDKit features + LLM IUPAC name.
  * molecule_info(smiles)              features + IUPAC + one-line note for a single molecule.
  * add_literature(text, source)       RAG KB (capped) for grounding/novelty.

Sessions expire after SESSION_TTL. No persistence of user data.
"""
from __future__ import annotations

import io
import time
import uuid
from typing import Dict, List, Optional

import numpy as np

import config
import data
import llm

SESSION_TTL = 3600  # seconds
MAX_TRAIN_ROWS = 20000
_SESSIONS: Dict[str, dict] = {}


# --------------------------------------------------------------------------- #
# Featurization (ECFP + RDKit descriptors; no xTB — must be fast for the web)
# --------------------------------------------------------------------------- #
def _features(smiles: List[str]) -> Dict[str, np.ndarray]:
    import ce_model as F            # ECFP + RDKit-descriptor featurizers live here
    ec = F._feat_ecfp(smiles)
    rd = F._feat_rdkit(smiles)
    out = {}
    for s in dict.fromkeys(smiles):
        if s in ec and s in rd:
            out[s] = np.concatenate([ec[s], rd[s]])
    return out


def _matrix(smiles, feats):
    keys = [s for s in smiles if s in feats]
    X = np.asarray([feats[s] for s in keys], np.float32) if keys else np.zeros((0, 1), np.float32)
    return X, keys


# --------------------------------------------------------------------------- #
# Objective parsing + per-target model training (ephemeral)
# --------------------------------------------------------------------------- #
def _detect_smiles_col(df) -> Optional[str]:
    # 1) prefer a column whose NAME mentions smiles (most reliable)
    named = [c for c in df.columns if "smiles" in str(c).lower()]
    if named:
        return named[0]
    # 2) otherwise, the column whose values most often parse as molecules
    best, best_hits = None, 0
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(40)
        hits = sum(1 for v in sample if data.canonical_smiles(v))
        if hits > best_hits:
            best, best_hits = c, hits
    return best if best_hits >= 5 else None


def _parse_objective(objective: str, columns: List[str]) -> List[dict]:
    """NL objective + column names -> [{column, direction: 'max'|'min', weight}]. LLM, with a
    numeric-column fallback."""
    numeric_like = [c for c in columns]
    out = llm.complete(
        "Map a chemist's optimization objective onto columns of their dataset. Return JSON "
        "{'targets':[{'column': <exact column name>, 'direction':'max'|'min', 'weight':0-1}]}. "
        "Use ONLY column names from the provided list. 'increase/improve/raise' -> max, "
        "'reduce/lower/decrease/minimize' -> min. Output ONLY JSON.",
        f"Objective: {objective}\nColumns: {columns}", role="fast", temperature=0, want_json=True)
    targets = []
    if isinstance(out, dict):
        for t in out.get("targets", []):
            col = t.get("column")
            if col in columns:
                targets.append({"column": col, "direction": "min" if str(t.get("direction")) == "min" else "max",
                                "weight": float(t.get("weight", 1.0))})
    return targets


def train_from_csv(csv_bytes: bytes, objective: str, smiles_col: Optional[str] = None) -> dict:
    """Train ephemeral per-target predictors. Returns {session_id, targets, metrics, n_rows}."""
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import r2_score

    df = pd.read_csv(io.BytesIO(csv_bytes))
    if len(df) > MAX_TRAIN_ROWS:
        df = df.sample(MAX_TRAIN_ROWS, random_state=0)
    scol = smiles_col if (smiles_col in df.columns) else _detect_smiles_col(df)
    if not scol:
        return {"ok": False, "error": "Could not find a SMILES column. Name one 'SMILES'."}
    df["_canon"] = df[scol].astype(str).map(data.canonical_smiles)
    df = df.dropna(subset=["_canon"])
    targets = _parse_objective(objective, [c for c in df.columns if c not in (scol, "_canon")])
    if not targets:
        return {"ok": False, "error": "Could not map your objective to any numeric column. "
                "Mention a column name, e.g. 'maximize CE_aver'."}

    feats = _features(df["_canon"].tolist())
    models, metrics, kept_targets = {}, {}, []
    for t in targets:
        col = t["column"]
        sub = df.dropna(subset=[col])
        y = pd.to_numeric(sub[col], errors="coerce")
        mask = y.notna()
        sm = sub["_canon"][mask].tolist(); yv = y[mask].to_numpy(float)
        X, keys = _matrix(sm, feats)
        if len(keys) < 20:
            continue
        yv = yv[[i for i, s in enumerate(sm) if s in feats]]
        rf = RandomForestRegressor(n_estimators=400, max_features=0.3, min_samples_leaf=2,
                                   n_jobs=-1, random_state=0)
        try:
            oof = cross_val_predict(rf, X, yv, cv=KFold(5, shuffle=True, random_state=0), n_jobs=-1)
            r2 = round(float(r2_score(yv, oof)), 3)
        except Exception:
            r2 = None
        rf.fit(X, yv)
        models[col] = {"model": rf, "mean": float(yv.mean()), "std": float(yv.std() or 1.0),
                       "direction": t["direction"], "weight": t["weight"]}
        metrics[col] = {"cv_r2": r2, "n": len(keys), "direction": t["direction"]}
        kept_targets.append(t)
    if not models:
        return {"ok": False, "error": "Not enough valid rows per target to train (need >=20)."}

    sid = uuid.uuid4().hex[:12]
    _gc()
    _SESSIONS[sid] = {"models": models, "targets": kept_targets, "ts": time.time(),
                      "objective": objective, "n_rows": len(df)}
    return {"ok": True, "session_id": sid, "targets": kept_targets, "metrics": metrics,
            "n_rows": len(df), "smiles_col": scol}


def load_pkl_model(pkl_bytes: bytes):  # gated off by default in server (RCE risk)
    import joblib
    return joblib.load(io.BytesIO(pkl_bytes))


def _gc():
    now = time.time()
    for k in [k for k, v in _SESSIONS.items() if now - v["ts"] > SESSION_TTL]:
        _SESSIONS.pop(k, None)


def _rf_mean_std(model, X):
    """RandomForest posterior: mean prediction + epistemic std across trees (the surrogate
    uncertainty that drives Bayesian-optimization acquisition)."""
    per_tree = np.stack([t.predict(X) for t in model.estimators_])  # [n_trees, n]
    return per_tree.mean(0), per_tree.std(0)


def _score(session: dict, smiles: List[str]) -> Dict:
    """Combined objective (z-scored, weighted, direction-aware) with propagated uncertainty.
    Returns mean[], std[] (acquisition uses mean + kappa*std), and per-target raw means."""
    feats = _features(smiles)
    X, keys = _matrix(smiles, feats)
    if not keys:
        return {"keys": []}
    total = np.zeros(len(keys)); var = np.zeros(len(keys))
    per_target = {}
    for col, m in session["models"].items():
        mu, sd = _rf_mean_std(m["model"], X)
        per_target[col] = mu
        sign = 1.0 if m["direction"] == "max" else -1.0
        w = m["weight"] / m["std"]
        total += m["weight"] * sign * (mu - m["mean"]) / m["std"]
        var += (w * sd) ** 2
    return {"keys": keys, "total": total, "std": np.sqrt(var), "per_target": per_target}


# --------------------------------------------------------------------------- #
# Molecule enrichment (features + IUPAC)
# --------------------------------------------------------------------------- #
def rdkit_info(smiles: str) -> Optional[dict]:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return {"formula": rdMolDescriptors.CalcMolFormula(m), "MolWt": round(Descriptors.MolWt(m), 1),
            "MolLogP": round(Crippen.MolLogP(m), 2), "TPSA": round(rdMolDescriptors.CalcTPSA(m), 1),
            "HBD": Lipinski.NumHDonors(m), "HBA": Lipinski.NumHAcceptors(m),
            "RotBonds": Descriptors.NumRotatableBonds(m), "Rings": rdMolDescriptors.CalcNumRings(m),
            "QED": round(QED.qed(m), 2)}


def iupac_names(smiles: List[str]) -> Dict[str, str]:
    """Batch IUPAC naming via LLM (one call). {} if unavailable."""
    if not smiles or not llm.available("fast"):
        return {}
    listing = "\n".join(f"{i}. {s}" for i, s in enumerate(smiles))
    out = llm.complete(
        "Give the IUPAC name for each SMILES. Return JSON {'names':[{'idx':int,'name':str}]}. "
        "If unsure, give the best systematic name. Output ONLY JSON.",
        listing, role="fast", temperature=0, want_json=True)
    res = {}
    if isinstance(out, dict):
        for r in out.get("names", []):
            try:
                res[smiles[int(r["idx"])]] = str(r["name"])
            except (KeyError, ValueError, IndexError, TypeError):
                continue
    return res


# --------------------------------------------------------------------------- #
# Generation (bulk + optional objective scoring)
# --------------------------------------------------------------------------- #
def _bo_search(engine, session, n, rounds=2, kappa=1.0, batch=90, seed_smiles=None):
    """Bayesian optimization over the VAE latent space with the trained RF surrogate and a
    UCB acquisition (mean + kappa*std). Each round: score the pool, pick the highest-acquisition
    molecules, decode latent NEIGHBORS of them (explore), evaluate, accumulate. Returns
    {smi: (per_target_mean, objective_mean, objective_std)} for the whole explored pool."""
    import ce_model
    net, vocab, device = engine.net, engine.vocab, engine.device
    seen: Dict[str, tuple] = {}

    def evaluate(smis):
        smis = [s for s in dict.fromkeys(smis) if s and s not in seen and ce_model.chem_ok(s)]
        if not smis:
            return
        sc = _score(session, smis)
        for i, s in enumerate(sc["keys"]):
            seen[s] = ({c: float(sc["per_target"][c][i]) for c in sc["per_target"]},
                       float(sc["total"][i]), float(sc["std"][i]))

    # round 0: broad prior sample (+ optional user seeds)
    init = ce_model._generate(net, vocab, device, batch, z_scale=1.0, temperature=0.9)
    if seed_smiles:
        init += seed_smiles
    evaluate(init)
    for _ in range(rounds):
        if not seen:
            break
        items = list(seen.items())
        acq = np.array([v[1] + kappa * v[2] for _, v in items])           # UCB
        top = [items[i][0] for i in np.argsort(acq)[::-1][:max(8, n // 2)]]
        lat = ce_model._feat_latent(top, net, vocab, device)
        seeds = np.array([lat[s] for s in top if s in lat], dtype=np.float32)
        if not len(seeds):
            break
        import torch
        nxt = ce_model._generate(net, vocab, device, batch, 1.0, 0.85,
                                 seed_latents=torch.tensor(seeds, device=device), spread=0.55)
        evaluate(nxt)
    return seen


def generate(engine, prompt: str = "", n: int = 20, session_id: Optional[str] = None,
             want_3d_top: int = 0, method: str = "bo", kappa: float = 1.0, rounds: int = 2) -> dict:
    """Generate n candidate molecules. With a trained objective (session), runs latent-space
    Bayesian optimization (method='bo', RF surrogate + UCB) and reports mean +/- uncertainty;
    without one, samples valid molecules from the prior. Enriches results with RDKit features,
    IUPAC names, and 3D for the top few. `engine` is a design.DesignEngine."""
    import ce_model
    net, vocab, device = engine.net, engine.vocab, engine.device
    session = _SESSIONS.get(session_id) if session_id else None
    if session:
        session["ts"] = time.time()

    used_bo = False
    if session and method == "bo":
        # scale BO budget to n so small requests stay fast (esp. on CPU)
        bo_rounds = 1 if n <= 4 else rounds
        bo_batch = int(min(max(n * 12, 30), 160))
        seen = _bo_search(engine, session, n, rounds=bo_rounds, kappa=kappa, batch=bo_batch)
        if seen:
            used_bo = True
            ranked = sorted(seen.items(), key=lambda kv: kv[1][1], reverse=True)[:n]
            ranked = [(s, v[0], v[1], v[2]) for s, v in ranked]
    if not used_bo:
        pool_target = int(min(max(n * 4, 24), 480))
        pool = ce_model._generate(net, vocab, device, pool_target, z_scale=0.95, temperature=0.9)
        pool = [s for s in dict.fromkeys(pool) if s and ce_model.chem_ok(s)]
        if not pool:
            return {"ok": False, "error": "no valid molecules generated"}
        if session:
            sc = _score(session, pool)
            order = np.argsort(sc["total"])[::-1][:n]
            ranked = [(sc["keys"][i], {c: float(sc["per_target"][c][i]) for c in sc["per_target"]},
                       float(sc["total"][i]), float(sc["std"][i])) for i in order]
        else:
            ranked = [(s, {}, 0.0, 0.0) for s in pool[:n]]

    # NOTE: IUPAC naming (an LLM call) and 3D are deferred to the per-molecule /molecule
    # endpoint (fetched when the user opens a result) so bulk generation has zero LLM latency.
    smis = [r[0] for r in ranked]
    from design import molblock_3d
    mols = []
    for rank, (smi, preds, score, unc) in enumerate(ranked, 1):
        info = rdkit_info(smi) or {}
        mid = engine.index.get_id(smi, already_canonical=True) if engine.index.available else None
        mols.append({
            "rank": rank, "smiles": smi, "iupac": "", "features": info,
            "predictions": {c: round(v, 3) for c, v in preds.items()},
            "objective_score": round(score, 3), "uncertainty": round(unc, 3),
            "molport_id": mid, "novel": mid is None,
            "molblock": molblock_3d(smi) if rank <= want_3d_top else None,
        })
    return {"ok": True, "n": len(mols), "molecules": mols, "method": "bayesopt" if used_bo else "sample",
            "objective": session["objective"] if session else None,
            "targets": session["targets"] if session else None}


def add_literature(text: str, source: str = "user") -> int:
    """Add a paper/note to the RAG KB (capped at ce_rag.MAX_SOURCES documents)."""
    import ce_rag
    return ce_rag.add_document(text, source or "user")


def molecule_info(smiles: str) -> dict:
    canon = data.canonical_smiles(smiles)
    if not canon:
        return {"ok": False, "error": "invalid SMILES"}
    from design import molblock_3d
    name = iupac_names([canon]).get(canon, "")
    return {"ok": True, "smiles": canon, "iupac": name, "features": rdkit_info(canon),
            "molblock": molblock_3d(canon)}
