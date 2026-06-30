"""
ce_llm.py — LLM layer for the Zn-additive CE inverse-design workflow.
====================================================================
Reuses the role routing in llm.py (fast=Kimi, reasoner=DeepSeek-V4-Pro, judge=gpt-5.4,
embed=text-embedding-3-large). Every function degrades gracefully to a no-op/heuristic
if no API key is present, so the pipeline never breaks.

Roles by job:
  parse_request   -> fast      (cheap structured extraction of constraints)
  assess_batch    -> reasoner  (synthesizability + stability + mechanism per molecule)
  judge_rerank    -> judge      (frontier model: final pick among top candidates)
  rationale       -> judge      (one mechanistic paragraph for a chosen molecule)
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import llm

PROPS = ["MolWt", "TPSA", "HDonor", "HAccept", "RotB", "FracCSP3", "MolLogP", "QED"]


# --------------------------------------------------------------------------- #
def parse_request(prompt: str) -> Dict:
    """NL design request -> constraints for ce_design.
    Returns {ranges:{col:[lo,hi]}, lmr, avoid:[smarts/desc], note}. Empty if no LLM."""
    if not llm.available("fast"):
        return {}
    sys = (
        "You configure a zinc-battery electrolyte-ADDITIVE generator that maximizes "
        "Coulombic efficiency (CE). Convert the user's request into JSON with keys: "
        "'ranges' (object; allowed keys ONLY MolWt,TPSA,HDonor,HAccept,RotB,FracCSP3,"
        "MolLogP,QED; each value [low,high]), 'lmr' (target log molar ratio, number or "
        "null), 'avoid' (list of short plain-English motifs to exclude, e.g. 'aldehydes'),"
        " 'note' (one-sentence design intent). Only include ranges the user implies. "
        "Output ONLY JSON."
    )
    out = llm.complete(sys, prompt, role="fast", temperature=0, want_json=True)
    if not isinstance(out, dict):
        return {}
    ranges = {k: v for k, v in (out.get("ranges") or {}).items()
              if k in PROPS and isinstance(v, (list, tuple)) and len(v) == 2}
    return {"ranges": ranges, "lmr": out.get("lmr"), "avoid": out.get("avoid") or [],
            "note": out.get("note", "")}


# --------------------------------------------------------------------------- #
_ASSESS_SYS = (
    "You are a synthetic + electrochemistry expert screening candidate ADDITIVE molecules "
    "for an aqueous zinc battery (goal: high plating/stripping Coulombic efficiency). "
    "For EACH molecule (given as SMILES with an ML-predicted CE), return a JSON object "
    "'results' = list of {idx, synth_score (0-1, ease of synthesis/commercial availability), "
    "stability (one of: stable, moderate, unstable-in-water), mechanism (<=12 words: how it "
    "could raise Zn CE — e.g. SEI former, water displacer, Zn2+ coordinator), "
    "redflag (short reason it is impractical, or empty)}. Be skeptical of reactive/unstable "
    "groups (aldehydes, enamines, strained rings, hydrazines). Output ONLY JSON."
)


def assess_batch(candidates: List[Dict], role: str = "reasoner", chunk: int = 25) -> Dict[int, Dict]:
    """candidates: [{idx, smiles, ce}]. Returns {idx: {synth_score, stability, mechanism,
    redflag}}. Empty if no LLM. Chunked so large lists stay within context."""
    if not llm.available(role):
        return {}
    out: Dict[int, Dict] = {}
    for i in range(0, len(candidates), chunk):
        part = candidates[i:i + chunk]
        user = "Molecules:\n" + "\n".join(
            f"{c['idx']}. {c['smiles']}  (ML CE~{c.get('ce', '?')}%)" for c in part)
        res = llm.complete(_ASSESS_SYS, user, role=role, temperature=0, want_json=True)
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


# --------------------------------------------------------------------------- #
def judge_rerank(candidates: List[Dict], top: int = 10, role: str = "judge") -> Optional[List[int]]:
    """Frontier-model final pick. candidates carry idx/smiles/ce/uncertainty/synth/mechanism.
    Returns an ordered list of idx (best first), or None if no LLM."""
    if not llm.available(role):
        return None
    sys = (
        "You are the lead scientist choosing which candidate zinc-battery additives to "
        "synthesize first to maximize Coulombic efficiency. Balance predicted CE, model "
        "uncertainty (lower better), synthesizability, stability in water, and mechanistic "
        f"plausibility. Return JSON {{'order': [idx,...]}} with the best {top} idx first. "
        "Output ONLY JSON."
    )
    user = "Candidates:\n" + "\n".join(
        f"{c['idx']}. {c['smiles']} | CE~{c.get('ce')}% unc={c.get('uncertainty')} "
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


# --------------------------------------------------------------------------- #
def rationale(smiles: str, ce: float, features: Optional[Dict] = None, role: str = "judge",
              context: str = "") -> str:
    """One mechanistic paragraph on why a molecule may raise Zn CE. If `context` (retrieved
    literature) is given, the model must ground its reasoning in it and cite [source] tags."""
    sys = ("You are a battery-electrolyte chemist. In 3-4 sentences, explain mechanistically "
           "why this molecule could improve Coulombic efficiency in an aqueous zinc battery "
           "(adsorption on Zn, SEI formation, water-activity suppression, Zn2+ coordination, "
           "dendrite suppression), name the key functional groups, and give ONE honest caveat "
           "(stability/synthesizability). Plain prose, specific not generic.")
    user = f"SMILES: {smiles}\nML-predicted CE: {ce:.1f}%\nfeatures: {features or {}}"
    if context:
        sys += (" Ground your reasoning in the provided literature excerpts and cite them as "
                "[source]. If the literature suggests this chemistry is already well-explored, "
                "say so (novelty caveat).")
        user += f"\n\nRelevant literature:\n{context}"
    return (llm.complete(sys, user, role=role, temperature=0.4)
            or "(LLM rationale unavailable — set FOUNDRY_API_KEY / AZURE_OPENAI_KEY.)")


if __name__ == "__main__":
    import sys
    print("LLM available (fast):", llm.available("fast"), "| (judge):", llm.available("judge"))
    q = " ".join(sys.argv[1:]) or "small water-stable Zn additives, MW under 200, avoid aldehydes"
    print("parse_request:", json.dumps(parse_request(q), indent=2))
