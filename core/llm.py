"""Optional LLM helper: turn a natural-language request into a property spec.

Roles are routed to deployments in ONE place (``ROUTES``) so you can swap models
without touching call sites. Endpoints/keys are read from the environment (.env).
If the ``openai`` SDK or the keys are missing, everything falls back to a
dependency-free keyword parser, so the rest of the pipeline never breaks.

  from batterygen.core.llm import nl_to_spec
  nl_to_spec("a small, very soluble, drug-like molecule with few rotatable bonds")
  # -> {"MolWt": 250, "MolLogP": 0.5, "TPSA": 90, "QED": 0.85, "NumRotatableBonds": 2}
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from batterygen.core import config


# Load keys from a .env if python-dotenv is available (optional). Look at the package-root
# .env (batterygen/.env) first, then any .env discovered from the current working directory.
# load_dotenv never overrides variables already set in the real environment.
try:  # pragma: no cover
    from pathlib import Path as _Path
    from dotenv import load_dotenv

    load_dotenv(_Path(__file__).resolve().parents[1] / ".env")
    load_dotenv()
except Exception:
    pass

FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_ENDPOINT", "https://zincbatteryresearch.services.ai.azure.com/openai/v1")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://batterymlresearch.openai.azure.com/openai/v1")
# Foundry hosts the reasoner/fast/embed models; Azure OpenAI hosts the judge. Each route
# carries the actual API key value for its endpoint (the judge falls back to the Foundry key
# if a separate Azure OpenAI key is not configured).
FOUNDRY_API_KEY = os.getenv("FOUNDRY_API_KEY")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY") or FOUNDRY_API_KEY
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5.4")
REASONER_MODEL = os.getenv("REASONER_MODEL", "DeepSeek-V4-Pro")
FAST_MODEL = os.getenv("FAST_MODEL", "Kimi-K2.6")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")


@dataclass
class ModelRoute:
    model: str
    endpoint: str
    key: Optional[str]      # resolved API key value for this route's endpoint


# Pick the model whose strengths match the job; swap deployments here only.
ROUTES: Dict[str, ModelRoute] = {
    "judge":    ModelRoute(JUDGE_MODEL,    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY),
    "reasoner": ModelRoute(REASONER_MODEL, FOUNDRY_ENDPOINT,      FOUNDRY_API_KEY),
    "fast":     ModelRoute(FAST_MODEL,     FOUNDRY_ENDPOINT,      FOUNDRY_API_KEY),
    "embed":    ModelRoute(EMBED_MODEL,    FOUNDRY_ENDPOINT,      FOUNDRY_API_KEY),
}


def _client(route: ModelRoute):
    """Build an OpenAI-compatible client, or None if the key/SDK is unavailable."""
    if not route.key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    return OpenAI(base_url=route.endpoint, api_key=route.key)


_SYSTEM = (
    "You convert a chemist's natural-language request into a JSON object of target "
    "molecular properties. Allowed keys ONLY: {props}. Values are either a single "
    "number or a [low, high] range, in these units: MolWt g/mol, MolLogP, TPSA Å², "
    "QED 0-1, counts are integers. Output ONLY compact JSON, no prose. Omit "
    "properties the user did not imply."
)


def nl_to_spec(prompt: str, role: str = "fast") -> Dict[str, float]:
    """Parse a request into a {property: value|[lo,hi]} spec. LLM if available,
    else keyword fallback. Always returns a (possibly empty) validated dict."""
    route = ROUTES.get(role, ROUTES["fast"])
    client = _client(route)
    if client is not None:
        try:
            sys = _SYSTEM.format(props=", ".join(config.PROPERTIES))
            resp = client.chat.completions.create(
                model=route.model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": prompt}],
                temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
            spec = json.loads(text)
            return _validate(spec)
        except Exception as e:  # pragma: no cover
            print(f"[llm] {route.model} call failed ({e}); using keyword fallback")
    return keyword_spec(prompt)


def _validate(spec: Dict) -> Dict[str, float]:
    valid = {}
    names = set(config.PROPERTIES)
    for k, v in spec.items():
        prop = config.PROPERTY_ALIASES.get(str(k).lower().strip(), k)
        if prop in names:
            valid[prop] = v
    return valid


# --------------------------------------------------------------------------- #
# Dependency-free keyword fallback
# --------------------------------------------------------------------------- #
_KEYWORDS = [
    (r"\b(tiny|fragment|fragment-like)\b", {"MolWt": 180, "HeavyAtomCount": 13}),
    (r"\b(small|low molecular weight|lightweight)\b", {"MolWt": 250}),
    (r"\b(lead-?like)\b", {"MolWt": 300, "QED": 0.8}),
    (r"\b(large|big|heavy|high molecular weight)\b", {"MolWt": 450}),
    (r"\b(soluble|hydrophilic|polar|water[- ]soluble)\b", {"MolLogP": 0.5, "TPSA": 90}),
    (r"\b(lipophilic|greasy|hydrophobic|membrane[- ]permeable)\b", {"MolLogP": 3.5}),
    (r"\b(drug-?like|druglike)\b", {"QED": 0.85}),
    (r"\b(rigid|few rotatable|conformationally constrained)\b", {"NumRotatableBonds": 2}),
    (r"\b(flexible|many rotatable)\b", {"NumRotatableBonds": 8}),
    (r"\b(aromatic|flat|planar)\b", {"NumAromaticRings": 3}),
    (r"\b(saturated|sp3|three[- ]dimensional|3d|aliphatic)\b", {"FractionCSP3": 0.6}),
    (r"\b(few h-?bond donors|low donor)\b", {"NumHDonors": 0}),
    (r"\bcns|brain[- ]penetrant|blood[- ]brain\b", {"TPSA": 50, "MolWt": 320, "NumHDonors": 1}),
]
# explicit "<prop> <number>" mentions, e.g. "mw 350", "logp 2.5", "qed 0.9"
_EXPLICIT = re.compile(
    r"\b(mw|molwt|molecular weight|logp|clogp|tpsa|psa|qed|hbd|donors|hba|acceptors|"
    r"rotatable|rings|heavy atoms?)\b[^\d-]{0,8}(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def keyword_spec(prompt: str) -> Dict[str, float]:
    p = prompt.lower()
    spec: Dict[str, float] = {}
    for pat, mapping in _KEYWORDS:
        if re.search(pat, p):
            spec.update(mapping)
    for word, num in _EXPLICIT.findall(p):
        prop = config.PROPERTY_ALIASES.get(word.lower().strip())
        if prop:
            spec[prop] = float(num)
    return _validate(spec)


_DECREASE = {"decrease", "lower", "reduce", "drop", "cut", "shrink", "down"}
_DELTA_RE = re.compile(
    r"(increase|raise|add|bump|boost|up|decrease|lower|reduce|drop|cut|shrink|down)\s+"
    r"(?:the\s+)?([a-z0-9_ ]+?)\s+by\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_REL_NUDGES = [  # qualitative relative edits -> (property, signed step)
    (r"\bmore soluble\b|\bmore polar\b", [("MolLogP", -1.0), ("TPSA", +20.0)]),
    (r"\bless soluble\b|\bmore lipophilic\b|\bmore greasy\b", [("MolLogP", +1.0)]),
    (r"\bmore rigid\b|\bfewer rotatable\b", [("NumRotatableBonds", -2.0)]),
    (r"\bmore flexible\b", [("NumRotatableBonds", +2.0)]),
    (r"\bmore aromatic\b", [("NumAromaticRings", +1.0)]),
    (r"\bmore drug-?like\b", [("QED", +0.1)]),
    (r"\bbigger\b|\blarger\b|\bheavier\b", [("MolWt", +40.0)]),
    (r"\bsmaller\b|\blighter\b", [("MolWt", -40.0)]),
]


def parse_relative(prompt: str, prev_spec: Dict[str, float],
                   achieved: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Apply a relative edit ('increase MW by 10 with the same optimization') to the
    previous target spec. Unmentioned previous targets are kept ('same optimization')."""
    achieved = achieved or {}
    spec = dict(prev_spec)  # keep prior targets

    def base_of(prop):
        return spec.get(prop, achieved.get(prop, 0.0))

    for verb, word, num in _DELTA_RE.findall(prompt):
        prop = config.PROPERTY_ALIASES.get(word.lower().strip(), word.strip())
        if prop in config.PROPERTIES:
            sign = -1.0 if verb.lower() in _DECREASE else 1.0
            spec[prop] = base_of(prop) + sign * float(num)
    p = prompt.lower()
    for pat, edits in _REL_NUDGES:
        if re.search(pat, p):
            for prop, step in edits:
                spec[prop] = base_of(prop) + step
    # absolute mentions ("set mw to 300") win — but strip the "<prop> by <N>" delta
    # phrases first so they aren't misread as absolute values.
    residual = _DELTA_RE.sub(" ", prompt)
    spec.update(keyword_spec(residual))
    return _validate(spec)


# --------------------------------------------------------------------------- #
# Inverse-design request parsing + mechanistic explanation
# --------------------------------------------------------------------------- #
_CATION_WORDS = {
    "lithium": "Li", "li-ion": "Li", "li metal": "Li", "li ": "Li",
    "sodium": "Na", "na-ion": "Na", "potassium": "K", "k-ion": "K",
    "magnesium": "Mg", "calcium": "Ca", "zinc": "Zn", "aluminum": "Al",
    "aluminium": "Al", "iron": "Fe", "proton": "H", "rubidium": "Rb", "cesium": "Cs",
}
_ROLE_WORDS = {"additive": "additive", "solvent": "solvent", "salt": "salt",
               "diluent": "diluent", "co-solvent": "solvent", "electrolyte": "electrolyte"}


def detect_chemistry(prompt: str) -> Dict[str, object]:
    p = " " + prompt.lower() + " "
    cation = next((v for k, v in _CATION_WORDS.items() if k in p), None)
    role = next((v for k, v in _ROLE_WORDS.items() if k in p), None)
    aqueous = "aqueous" in p or "water" in p
    return {"cation": cation, "role": role, "aqueous": aqueous}


_DESIGN_SYS = (
    "You are an electrochemistry design assistant. Convert the user's request for a "
    "battery electrolyte molecule (solvent/additive/salt) into JSON with keys: "
    "'spec' (object of RDKit property targets, allowed keys ONLY: {props}; number or "
    "[low,high]), 'cation' (Li/Na/K/Mg/Ca/Zn/Al/... or null), 'aqueous' (bool), "
    "'role' (additive/solvent/salt/electrolyte or null), 'summary' (one sentence of the "
    "design intent). Pick property targets that suit the chemistry (e.g. aqueous Zn "
    "additives: small, polar, moderate logP, H-bonding). Output ONLY JSON."
)


def parse_design(prompt: str, role: str = "judge") -> Dict[str, object]:
    """Rich parse of a de-novo design request -> {spec, cation, aqueous, role, summary}."""
    out = {"spec": {}, **detect_chemistry(prompt), "summary": ""}
    route = ROUTES.get(role, ROUTES["judge"])
    client = _client(route)
    if client is not None:
        try:
            sys = _DESIGN_SYS.format(props=", ".join(config.PROPERTIES))
            r = client.chat.completions.create(
                model=route.model, temperature=0,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}])
            txt = re.sub(r"^```(json)?|```$", "", r.choices[0].message.content.strip(), flags=re.MULTILINE).strip()
            d = json.loads(txt)
            out["spec"] = _validate(d.get("spec", {}))
            for k in ("cation", "aqueous", "role", "summary"):
                if d.get(k) is not None:
                    out[k] = d[k]
            return out
        except Exception as e:  # pragma: no cover
            print(f"[llm] parse_design fell back ({e})")
    out["spec"] = keyword_spec(prompt)  # fallback
    return out


def explain(request: str, smiles: str, props: Dict, elec: Optional[Dict] = None,
            role: str = "judge") -> str:
    """Mechanistic, cautious explanation of a generated molecule for the request."""
    route = ROUTES.get(role, ROUTES["judge"])
    client = _client(route)
    if client is None:
        return ("(LLM explanation unavailable — set Azure keys in .env.) "
                f"Generated {smiles} matching the requested properties.")
    elec_txt = f"\nPredicted electrolyte properties: {elec}" if elec else ""
    sys = ("You are a battery-electrolyte chemist. In 3-5 sentences, explain why the "
           "given molecule is a plausible candidate for the user's request: likely role "
           "(solvation, SEI/CEI formation, oxidative/reductive stability, ion transport), "
           "key functional groups, and one honest caveat (e.g. stability, synthesizability). "
           "Be specific and mechanistic, not generic. Plain prose.")
    user = f"Request: {request}\nMolecule (SMILES): {smiles}\nRDKit properties: {props}{elec_txt}"
    try:
        r = client.chat.completions.create(
            model=route.model, temperature=0.4,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}])
        return r.choices[0].message.content.strip()
    except Exception as e:  # pragma: no cover
        return f"(explanation error: {e})"


def available(role: str = "fast") -> bool:
    """True if an LLM client can be built for this role (keys + SDK present)."""
    return _client(ROUTES.get(role, ROUTES["fast"])) is not None


def complete(system: str, user: str, role: str = "fast", temperature: float = 0.2,
             want_json: bool = False):
    """Generic chat completion routed by role. Returns text, or a parsed dict when
    want_json=True, or None if no client / on error. Reused across the pipeline."""
    route = ROUTES.get(role, ROUTES["fast"])
    client = _client(route)
    if client is None:
        return None
    try:
        kw = {}
        if want_json:
            kw["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(
            model=route.model, temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kw)
        txt = r.choices[0].message.content.strip()
        if not want_json:
            return txt
        txt = re.sub(r"^```(json)?|```$", "", txt, flags=re.MULTILINE).strip()
        return json.loads(txt)
    except Exception as e:  # pragma: no cover
        # retry once without response_format (some deployments reject it)
        if want_json:
            try:
                r = client.chat.completions.create(
                    model=route.model, temperature=temperature,
                    messages=[{"role": "system", "content": system + " Output ONLY JSON."},
                              {"role": "user", "content": user}])
                txt = re.sub(r"^```(json)?|```$", "", r.choices[0].message.content.strip(),
                             flags=re.MULTILINE).strip()
                return json.loads(txt)
            except Exception as e2:
                print(f"[llm] {route.model} json call failed ({e2})")
                return None
        print(f"[llm] {route.model} call failed ({e})")
        return None


def embed(texts: List[str], role: str = "embed") -> Optional[List[List[float]]]:
    """Embed text (literature novelty search etc.). None if unavailable."""
    route = ROUTES.get(role, ROUTES["embed"])
    client = _client(route)
    if client is None:
        return None
    try:  # pragma: no cover
        resp = client.embeddings.create(model=route.model, input=texts)
        return [d.embedding for d in resp.data]
    except Exception:
        return None


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "a small, very soluble, drug-like molecule, few rotatable bonds"
    print("prompt:", q)
    print("spec  :", nl_to_spec(q))
