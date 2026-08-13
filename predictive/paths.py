"""predictive/paths.py - where the predictive pipeline writes its artifacts.

All predictive artifacts live under ``<artifacts>/predictive/`` so they sit apart from the
generator's checkpoints and processed data.
"""
from __future__ import annotations

from batterygen.core import config

PRED_DIR = config.CKPT_DIR.parent / "predictive"

FEATURE_CACHE = PRED_DIR / "feature_cache.pkl"      # features.py output
SELECTED = PRED_DIR / "selected_features.json"      # select.py output (the shortlist)
HPARAMS = PRED_DIR / "best_hparams.pkl"             # tune.py output
MODEL = PRED_DIR / "production_model.pkl"           # train.py output (locked model bundle)
XTB_CACHE = PRED_DIR / "xtb_cache.csv"              # cached GFN2-xTB single points
FIG_DIR = PRED_DIR / "figures"
KB_DIR = PRED_DIR / "kb"                            # literature RAG store
