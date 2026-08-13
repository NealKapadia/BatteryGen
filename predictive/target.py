"""predictive/target.py - THE ONE PLACE TO CONFIGURE A NEW CHEMISTRY / TARGET.
================================================================================
Edit ``TARGET`` below (or switch ``PRESET``) to point the ENTIRE predictive +
inverse-design pipeline at a different battery chemistry (Li / Na / K / Zn / ...) or a
different measured property. Nothing else in the pipeline hardcodes a column name, a
cation, or an objective - every stage reads its settings from here:

    features  -> which CSV columns are the SMILES / target / context / replicates
    select    -> how many features to keep, whether to force-keep the context columns
    train     -> how the target is transformed, whether xTB electronic features are used
    design    -> maximize vs. minimize, and the natural-language framing of the search
    llm / rag -> the system / objective / cation strings injected into every prompt

Switch chemistry with one line (or the ``BATTERYGEN_TARGET`` env var):

    TARGET = PRESETS["Li"]          # or set BATTERYGEN_TARGET=Li in the environment

When you bring your own dataset, edit the *_col fields to match your CSV headers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import List, Optional


@dataclass
class TargetConfig:
    # -- data schema (must match your CSV headers) ----------------------------
    smiles_col: str                                   # column holding the molecule SMILES
    target_col: str                                   # the measured value to predict
    context_cols: List[str] = field(default_factory=list)    # extra numeric CSV columns -> features
    replicate_cols: List[str] = field(default_factory=list)  # repeat measurements -> noise filter (optional)
    maximize: bool = True                             # maximize (True) or minimize (False) the target
    target_transform: str = "none"                    # "none" | "lce"  (lce stretches a 0-100 % efficiency tail)

    # -- feature selection ----------------------------------------------------
    use_xtb: bool = True                              # include GFN2-xTB electronic features (HOMO/LUMO/...)
    n_features: Optional[int] = None                  # None = RFE-CV auto; an int forces top-K
    pin_context: bool = True                          # always keep the context columns through selection

    # -- noise filter ---------------------------------------------------------
    max_cov: float = 3.0                              # drop replicate rows whose CoV% exceeds this (ignored if no replicates)

    # -- chemistry framing (interpolated into every LLM + RAG prompt) ----------
    system: str = "battery electrolyte additive"      # what the molecule IS, in words
    objective: str = "improve electrochemical performance"  # what "good" MEANS
    cation: str = ""                                  # working ion, e.g. "Li+", "Zn2+"
    fatal_flag: str = ""                              # stability label that eliminates a candidate ("" = none)

    @property
    def target_name(self) -> str:
        """Short clean label for output columns / logs, e.g. 'CE_aver. (%)' -> 'CE_aver'."""
        base = self.target_col.split("(")[0].strip().rstrip("_. ")
        return base or "target"


# --------------------------------------------------------------------------- #
# Presets. The default (Zn) matches the shipped Coulombic-Efficiency dataset.
# The Li / Na / K presets share that schema as a starting point and change only the
# chemistry framing - edit their *_col fields to match your own dataset's headers.
# --------------------------------------------------------------------------- #
_ZN = TargetConfig(
    smiles_col="Additive_SMILES",
    target_col="CE_aver. (%)",
    context_cols=["LogMolarRatio"],
    replicate_cols=["CE_1 (%)", "CE_2 (%)", "CE_3 (%)"],
    maximize=True,
    target_transform="lce",
    use_xtb=True,
    n_features=None,
    pin_context=True,
    max_cov=3.0,
    system="aqueous zinc-metal battery electrolyte additive",
    objective="maximize zinc plating/stripping Coulombic efficiency (CE)",
    cation="Zn2+",
    fatal_flag="unstable-in-water",
)

PRESETS = {
    "Zn": _ZN,
    "Li": replace(
        _ZN,
        system="lithium-metal battery electrolyte additive",
        objective="maximize lithium plating/stripping Coulombic efficiency (CE)",
        cation="Li+",
        fatal_flag="reactive-with-lithium-metal",
    ),
    "Na": replace(
        _ZN,
        system="sodium-metal battery electrolyte additive",
        objective="maximize sodium plating/stripping Coulombic efficiency (CE)",
        cation="Na+",
        fatal_flag="reactive-with-sodium-metal",
    ),
    "K": replace(
        _ZN,
        system="potassium-metal battery electrolyte additive",
        objective="maximize potassium plating/stripping Coulombic efficiency (CE)",
        cation="K+",
        fatal_flag="reactive-with-potassium-metal",
    ),
}

# The active configuration. Change "Zn" here, or set BATTERYGEN_TARGET=Li/Na/K.
TARGET: TargetConfig = PRESETS[os.getenv("BATTERYGEN_TARGET", "Zn")]
