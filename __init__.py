"""BatteryGen - a pretrained SELFIES-VAE for de-novo molecule & battery-electrolyte design.

Public entry point::

    from batterygen import BatteryGen
    bg = BatteryGen(device="cpu")
    bg.generate(10)

Layout
------
The code is organized into purpose subpackages (``core``, ``generative``, ``predictive``,
``electrolyte``, ``grounding``). Modules import each other by their fully-qualified package
path (``from batterygen.core import config``), so every entry point runs directly with
``python -m batterygen.<subpackage>.<module>`` and after a plain ``pip install`` — no
``sys.path`` manipulation required.
"""
from __future__ import annotations

from batterygen.core.api import BatteryGen

__all__ = ["BatteryGen"]
__version__ = "0.2.0"
