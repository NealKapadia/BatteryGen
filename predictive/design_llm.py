"""
predictive/design_llm.py - LLM layer for the inverse-design workflow.
====================================================================
Every prompt is built from TARGET (system / objective / cation / fatal_flag), so the LLM
reasons about *your* chemistry, not a hardcoded one. Reuses the role routing in core/llm.py
(fast / reasoner / judge / embed). Every function degrades gracefully to a no-op if no API
key is present, so the pipeline never breaks.

Roles by job:
  parse_request  -> fast      (cheap structured extraction of constraints)
  assess_batch   -> reasoner  (synthesizability + stability + mechanism per molecule)
  judge_rerank   -> judge     (frontier model: final pick among top candidates)
  rationale      -> judge     (one mechanistic paragraph for a chosen molecule)
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from molforge.core import llm
from molforge.predictive.target import TARGET

# Descriptor keys the LLM may constrain (must match features.rdkit_features output).
PROPS = ["MolWt", "TPSA", "HDonor", "HAccept", "RotB", "FracCSP3", "MolLogP", "QED"]


def parse_request(prompt: str) -> Dict:
    """NL design request -> constraints. Returns {ranges:{col:[lo,hi]}, context:{col:val},
    avoid:[...], note}. Empty if no LLM."""
    if not llm.available("fast"):
        return {}
    ctx_keys = ", ".join(TARGET.context_cols) or "(none)"
    sys = (
        f"You configure a generator of {TARGET.system} candidates whose goal is to "
        f"{TARGET.objective}. Convert the user's request into JSON with keys: "
        f"'ranges' (object; allowed keys ONLY {', '.join(PROPS)}; each value [low,high]), "
        f"'context' (object of experimental context variables, allowed keys ONLY {ctx_keys}; "
        "number values), 'avoid' (list of short plain-English motifs to exclude, e.g. "
        "'aldehydes'), 'note' (one-sentence design intent). Only include what the user "
        "implies. Output ONLY JSON."
    )
    out = llm.complete(sys, prompt, role="fast", temperature=0, want_json=True)
    if not isinstance(out, dict):
        return {}
    ranges = {k: v for k, v in (out.get("ranges") or {}).items()
              if k in PROPS and isinstance(v, (list, tuple)) and len(v) == 2}
    context = {k: v for k, v in (out.get("context") or {}).items()
               if k in TARGET.context_cols and isinstance(v, (int, float))}
    return {"ranges": ranges, "context": context, "avoid": out.get("avoid") or [],
            "note": out.get("note", "")}


def assess_batch(candidates: List[Dict], role: str = "reasoner", chunk: int = 25) -> Dict[int, Dict]:
    """candidates: [{idx, smiles, pred}]. Returns {idx: {synth_score, stability, mechanism,
    redflag}}. Empty if no LLM. Chunked so large lists stay within context."""
    if not llm.available(role):
        return {}
    stab_options = f"stable, moderate, {TARGET.fatal_flag}" if TARGET.fatal_flag else "stable, moderate, unstable"
    sys = (
        f"You are a synthetic + electrochemistry expert screening candidate {TARGET.system} "
        f"molecules (goal: {TARGET.objective}). For EACH molecule (SMILES with an ML-predicted "
        f"{TARGET.target_name}) return a JSON object 'results' = list of {{idx, synth_score "
        "(0-1, ease of synthesis / commercial availability), stability (one of: "
        f"{stab_options}), mechanism (<=12 words: how it could help - e.g. SEI former, "
        f"{TARGET.cation} coordinator, anion-receptor), redflag (short reason it is impractical, "
        "or empty)}. Be skeptical of reactive/unstable groups (aldehydes, enamines, strained "
        "rings, hydrazines). Output ONLY JSON."
    )
    out: Dict[int, Dict] = {}
    for i in range(0, len(candidates), chunk):
        part = candidates[i:i + chunk]
        user = "Molecules:\n" + "\n".join(
            f"{c['idx']}. {c['smiles']}  (ML {TARGET.target_name}~{c.get('pred', '?')})" for c in part)
        res = llm.complete(sys, user, role=role, temperature=0, want_json=True)
        rows = (res or {}).get("results", []) if isinstance(res, dict) else []
        for r in rows:
            try:
                out[int(r["idx"])] = {
                    "synth_score": float(r.get("synth_score", 0.5)),
                    "stability": str(r.get("stability", "")),
                    "mechanism": str(r.get("mechanism", "")),
                    "redflag": str(r.get("redflag", "")),
                }
            except (KeyError, ValueError, TypeError):
                continue
    return out


def judge_rerank(candidates: List[Dict], top: int = 10, role: str = "judge") -> Optional[List[int]]:
    """Frontier-model final pick. candidates carry idx/smiles/pred/uncertainty/synth/mechanism.
    Returns an ordered list of idx (best first), or None if no LLM."""
    if not llm.available(role):
        return None
    sys = (
        f"You are the lead scientist choosing which candidate {TARGET.system} molecules to "
        f"synthesize first to {TARGET.objective}. Balance predicted {TARGET.target_name}, model "
        "uncertainty (lower better), synthesizability, stability, and mechanistic plausibility. "
        f"Return JSON {{'order': [idx,...]}} with the best {top} idx first. Output ONLY JSON."
    )
    user = "Candidates:\n" + "\n".join(
        f"{c['idx']}. {c['smiles']} | {TARGET.target_name}~{c.get('pred')} unc={c.get('uncertainty')} "
        f"synth={c.get('synth_score')} stab={c.get('stability')} mech={c.get('mechanism')}"
        for c in candidates)
    res = llm.complete(sys, user, role=role, temperature=0.1, want_json=True)
    order = (res or {}).get("order") if isinstance(res, dict) else None
    if not order:
        return None
    seen, clean = set(), []
    for x in order:
        try:
            xi = int(x)
        except (ValueError, TypeError):
            continue
        if xi not in seen:
            seen.add(xi); clean.append(xi)
    return clean or None


def rationale(smiles: str, pred: float, features: Optional[Dict] = None, role: str = "judge",
              context: str = "") -> str:
    """One mechanistic paragraph on why a molecule may help. If `context` (retrieved
    literature) is given, the model must ground its reasoning in it and cite [source] tags."""
    sys = (f"You are a battery-electrolyte chemist. In 3-4 sentences, explain mechanistically "
           f"why this molecule could {TARGET.objective} in a {TARGET.system} context "
           f"(adsorption, SEI/CEI formation, {TARGET.cation} coordination, solvation-shell "
           "tuning, dendrite suppression), name the key functional groups, and give ONE honest "
           "caveat (stability/synthesizability). Plain prose, specific not generic.")
    user = f"SMILES: {smiles}\nML-predicted {TARGET.target_name}: {pred:.2f}\nfeatures: {features or {}}"
    if context:
        sys += (" Ground your reasoning in the provided literature excerpts and cite them as "
                "[source]. If the literature suggests this chemistry is already well-explored, "
                "say so (novelty caveat).")
        user += f"\n\nRelevant literature:\n{context}"
    return (llm.complete(sys, user, role=role, temperature=0.4)
            or "(LLM rationale unavailable - set FOUNDRY_API_KEY / AZURE_OPENAI_KEY.)")


if __name__ == "__main__":
    import sys
    print("LLM available (fast):", llm.available("fast"), "| (judge):", llm.available("judge"))
    q = " ".join(sys.argv[1:]) or f"small stable {TARGET.cation} additives, MW under 200, avoid aldehydes"
    print("parse_request:", json.dumps(parse_request(q), indent=2))
