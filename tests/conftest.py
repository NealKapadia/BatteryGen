"""Shared pytest fixtures / markers for the MolForge test suite.

The lightweight tests need no model weights and run anywhere. Tests that need the
pretrained checkpoint are skipped automatically unless the artifacts are present
(point MOLVAE_ART_DIR at a folder containing checkpoints/best.pt + processed/*.json).
"""
import sys
from pathlib import Path

# Make the flat modules (config, data, model, ...) importable by bare name,
# exactly as running `python molvae/<script>.py` does.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
