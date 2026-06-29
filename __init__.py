"""MolForge — a pretrained SELFIES-VAE for de-novo molecule & battery-electrolyte design.

The public entry point is :class:`MolForge` (generate / encode / decode / score molecules
on CPU or GPU, no server or API keys required)::

    from molforge import MolForge
    mf = MolForge(device="cpu")
    mf.generate(10)

Packaging note
--------------
This project keeps its original *flat* module layout (``config.py``, ``data.py``,
``model.py``, ``infer.py``, ...) where modules import each other by bare name. That only
resolves when the package directory is on ``sys.path`` — exactly what running
``python molvae/<script>.py`` does via the script's own directory. We reproduce that here
so the bare imports keep working after a ``pip install`` without editing any module, and
without installing common names like ``config``/``data`` as top-level site-packages modules.
"""
from __future__ import annotations

import os as _os
import sys as _sys

# Put this package's own directory on sys.path so the sibling modules can keep
# importing each other by bare name (import config / import data / ...).
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)

from .molforge import MolForge  # noqa: E402  (must follow the sys.path shim)

__all__ = ["MolForge"]
__version__ = "0.1.0"
