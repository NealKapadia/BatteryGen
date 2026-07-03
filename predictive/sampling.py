"""
predictive/sampling.py - generator-interface helpers for the design loop.
=========================================================================
The thin layer between the SELFIES-VAE generator and the predictive model:
  load_labeled_csv  read (SMILES, target) rows from the dataset (schema from TARGET)
  chem_ok           reject obviously unstable/reactive/non-viable candidate chemistry
  _feat_latent      encode SMILES -> VAE latent means (for re-seeding around good hits)
  _generate         sample the VAE prior (optionally around seed latents) -> SMILES
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from molforge.core import config
from molforge.core import data
from molforge.predictive.target import TARGET


def _resolve_ckpt(ckpt):
    """Default to the validated base (best.pt), not latest.pt (may be mid-continued-train)."""
    return ckpt or str(config.CKPT_DIR / "best.pt")


def load_labeled_csv(path: Path) -> List[dict]:
    """Read [{smiles, target, name}] using the columns configured in TARGET."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            canon = data.canonical_smiles((r.get(TARGET.smiles_col) or "").strip())
            if not canon:
                continue
            try:
                y = float(r[TARGET.target_col])
            except (KeyError, ValueError):
                continue
            rows.append({"smiles": canon, "target": y, "name": (r.get("IUPAC_NAME") or "").strip()})
    return rows


_BAD_SMARTS = None
_OK_ANION_SMARTS = None


def chem_ok(smi: str) -> bool:
    """Reject obviously unstable/reactive or non-viable candidates so the output is testable
    chemistry. Drops acyl halides, N-halamines, peroxides, hydrazines (N-N), allenes/ketenes/
    diazo, strained 3-rings, exotic atoms, bare cations, and charged species that are NOT a
    real anionic-additive class (carboxylate/sulfonate/sulfonimide)."""
    from rdkit import Chem

    global _BAD_SMARTS, _OK_ANION_SMARTS
    if _BAD_SMARTS is None:
        pats = ["[CX3](=O)[F,Cl,Br,I]", "[#7][F,Cl,Br,I]", "[OX2][OX2]", "[#7]-[#7]",
                "[#6]=[#6]=[#6]", "[#6]=[#6]=[#8]", "*=[N+]=[N-]", "[r3]", "[#6]=[#6]=[#7]"]
        _BAD_SMARTS = [Chem.MolFromSmarts(p) for p in pats]
        _OK_ANION_SMARTS = [Chem.MolFromSmarts(p) for p in
                            ("[CX3](=O)[O-]", "[SX4](=O)(=O)[O-]", "[#7-]S(=O)(=O)", "[n-]")]
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    if any(a.GetSymbol() not in config.ALLOWED_ATOMS for a in m.GetAtoms()):
        return False
    chg = Chem.GetFormalCharge(m)
    if chg > 0:
        return False  # bare cations aren't neutral/anionic additives
    if chg < 0 and not any(m.HasSubstructMatch(p) for p in _OK_ANION_SMARTS):
        return False  # anion but not a carboxylate/sulfonate/sulfonimide -> artifact
    return not any(p is not None and m.HasSubstructMatch(p) for p in _BAD_SMARTS)


@torch.no_grad()
def _feat_latent(smiles: List[str], net, vocab, device, batch=256) -> Dict[str, np.ndarray]:
    import selfies as sf

    out, ids_buf, len_buf, smi_buf = {}, [], [], []

    def flush():
        if not ids_buf:
            return
        mu, _ = net.encode(torch.tensor(ids_buf, dtype=torch.long, device=device),
                           torch.tensor(len_buf))
        for s, m in zip(smi_buf, mu.cpu().numpy()):
            out[s] = m
        ids_buf.clear(); len_buf.clear(); smi_buf.clear()

    for s in dict.fromkeys(smiles):
        try:
            sel = sf.encoder(s)
        except Exception:
            continue
        if not sel or sf.len_selfies(sel) > config.MAX_SELFIES_LEN - 2:
            continue
        ids, length = vocab.encode(sel)
        ids_buf.append(ids); len_buf.append(length); smi_buf.append(s)
        if len(ids_buf) >= batch:
            flush()
    flush()
    return out


@torch.no_grad()
def _generate(net, vocab, device, n, z_scale, temperature, seed_latents=None, spread=0.6):
    # validity is ~1.0 (default SELFIES constraints), so generate only ~1.3x the shortfall.
    smis, tries = [], 0
    while len(smis) < n and tries < 6:
        tries += 1
        b = min(512, max(8, int((n - len(smis)) * 1.3)))
        cond = torch.zeros(b, config.N_PROPS, device=device)
        if seed_latents is not None and len(seed_latents):
            idx = torch.randint(0, seed_latents.size(0), (b,), device=device)
            z = seed_latents[idx] + spread * torch.randn(b, net.latent_dim, device=device)
            seqs = net.sample(b, cond, vocab.bos, vocab.eos, z=z, temperature=temperature, device=device)
        else:
            seqs = net.sample(b, cond, vocab.bos, vocab.eos, temperature=temperature,
                              z_scale=z_scale, device=device)
        smis.extend(filter(None, (vocab.decode_to_smiles(s) for s in seqs.tolist())))
    return smis
