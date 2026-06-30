"""MolForge - a pretrained SELFIES-VAE for de-novo molecule & battery-electrolyte design.

Public entry point::

    from molforge import MolForge
    mf = MolForge(device="cpu")
    mf.generate(10)

Layout
------
The code is organized into purpose folders (core, generative, predictive, electrolyte,
grounding, webapp). Internally the modules import each other by bare name (``import config``,
``import data``, ...). To keep that working after a ``pip install`` without rewriting every
module, this package adds each subfolder to ``sys.path`` on import - the same mechanism that
running a script from its own folder relies on. This is why there is no top-level ``molforge.py``
file (it would shadow the package name once ``core/`` is on the path); the class lives in
``core/api.py``.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
for _sub in ("core", "generative", "predictive", "electrolyte", "grounding", "webapp"):
    _p = _os.path.join(_pkg_dir, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)

from .core.api import MolForge  # noqa: E402  (must follow the sys.path setup)

__all__ = ["MolForge"]
__version__ = "0.2.0"
