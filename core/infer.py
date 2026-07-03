"""Shared inference helpers: load a trained checkpoint, build condition vectors,
and score molecules. Used by generate.py, search.py and finetune_dft.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from molforge.core import config

from molforge.core import data

from molforge.core import model as M



def load_model(ckpt_path: Optional[str] = None, device=None):
    device = device or config.get_device()
    vocab = data.Vocab.load()
    path = Path(ckpt_path) if ckpt_path else (config.CKPT_DIR / "latest.pt")
    if not path.exists():
        raise SystemExit(f"No checkpoint at {path}. Train first: python molvae/train.py")
    ck = torch.load(path, map_location=device)
    net = M.SelfiesVAE(vocab_size=len(vocab), pad_idx=vocab.pad, unk_idx=vocab.unk,
                       **ck.get("hparams", {})).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, vocab, ck, device


def resolve_spec(spec: Optional[Dict], prompt: Optional[str]) -> Dict[str, float]:
    """Merge an explicit spec dict with one parsed from a natural-language prompt."""
    merged: Dict[str, float] = {}
    if prompt:
        try:
            from molforge.core import llm


            merged.update(llm.nl_to_spec(prompt))
        except Exception as e:  # pragma: no cover - optional path
            print(f"[llm] natural-language parsing unavailable ({e}); using --spec only")
    if spec:
        merged.update(spec)
    return merged


def condition_tensor(spec: Dict[str, float], n: int, device) -> torch.Tensor:
    cond = data.spec_to_condition(spec)  # normalized, unspecified -> 0
    return torch.from_numpy(cond).float().unsqueeze(0).repeat(n, 1).to(device)


def score_smiles(smiles: str) -> Optional[Dict[str, float]]:
    """Return {property: value} for a SMILES, or None if invalid."""
    vals = data.descriptors_for_smiles(smiles)
    if vals is None:
        return None
    return dict(zip(config.PROPERTIES, vals))


def specified_mask(spec: Dict[str, float]) -> np.ndarray:
    name_to_idx = {p: i for i, p in enumerate(config.PROPERTIES)}
    mask = np.zeros(config.N_PROPS, dtype=bool)
    for key in spec:
        prop = config.PROPERTY_ALIASES.get(str(key).lower().strip(), key)
        if prop in name_to_idx:
            mask[name_to_idx[prop]] = True
    return mask


def iter_dataset_descriptors():
    """Yield (canonical_smiles, molport_id, descriptor_row) over all shards."""
    meta = json.load(open(config.META_PATH, encoding="utf-8"))
    for shard in meta["shards"]:
        desc = np.load(config.PROC_DIR / f"{shard}_desc.npy")
        with open(config.PROC_DIR / f"{shard}_ids.txt", encoding="utf-8") as f:
            lines = f.read().splitlines()
        for row, line in zip(desc, lines):
            canon, _, mid = line.partition("\t")
            yield canon, mid, row
