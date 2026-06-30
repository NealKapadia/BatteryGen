"""Shared pytest fixtures / markers for the MolForge test suite.

The lightweight tests need no model weights and run anywhere. Tests that need the
pretrained checkpoint are skipped automatically unless the artifacts are present
(point MOLVAE_ART_DIR at a folder containing checkpoints/best.pt + processed/*.json).
"""
import sys
from pathlib import Path

# Put the package root and every purpose subfolder on sys.path so the modules
# (config, data, ce_features, ...) resolve by bare name, mirroring the runtime shim
# in molforge/__init__.py.
ROOT = Path(__file__).resolve().parent.parent
for _sub in ("", "core", "generative", "predictive", "electrolyte", "grounding", "webapp"):
    _p = str(ROOT / _sub) if _sub else str(ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402
import config  # noqa: E402


def _weights_available() -> bool:
    return (config.CKPT_DIR / "best.pt").exists() and config.VOCAB_PATH.exists()


def _stats_available() -> bool:
    return config.DESC_STATS_PATH.exists() and config.VOCAB_PATH.exists()


requires_weights = pytest.mark.skipif(
    not _weights_available(),
    reason="model artifacts (checkpoints/best.pt + processed/vocab.json) not found; set MOLVAE_ART_DIR",
)

requires_stats = pytest.mark.skipif(
    not _stats_available(),
    reason="processed/descriptor_stats.json not found; set MOLVAE_ART_DIR",
)


@pytest.fixture(scope="session")
def vocab():
    import data
    return data.Vocab.build()
