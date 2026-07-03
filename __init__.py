"""MolForge - a pretrained SELFIES-VAE for de-novo molecule & battery-electrolyte design.

Public entry point::

    from molforge import MolForge
    mf = MolForge(device="cpu")
    mf.generate(10)

Layout
------
The code is organized into purpose subpackages (``core``, ``generative``, ``predictive``,
``electrolyte``, ``grounding``). Modules import each other by their fully-qualified package
path (``from molforge.core import config``), so every entry point runs directly with
``python -m molforge.<subpackage>.<module>`` and after a plain ``pip install`` — no
``sys.path`` manipulation required.
"""
from __future__ import annotations

from molforge.core.api import MolForge

__all__ = ["MolForge"]
__version__ = "0.2.0"
