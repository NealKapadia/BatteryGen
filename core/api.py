"""
molforge — a small public API for the MolForge generative molecule model.
========================================================================
Load the pretrained SELFIES-VAE and generate / encode / decode / score molecules on
your own CPU or GPU. This is the friendly facade over the internal modules
(config/data/model/infer); it does not require the web app or any Azure keys.

What you need to run it locally:
  - this `molvae/` folder (code)
  - the artifacts bundle:  processed/{vocab.json, descriptor_stats.json, meta.json}
                           checkpoints/best.pt
    Point to it with MOLVAE_ART_DIR, or pass artifacts_dir=... to MolForge().

Quick start
-----------
    from molforge import MolForge
    mf = MolForge(device="cpu")                 # or "cuda"
    mf.generate(10)                             # 10 valid, novel SMILES
    mf.generate(5, spec={"MolWt": 250, "QED": 0.8})   # property-targeted
    mf.properties("OCCN(CCO)CCO")              # RDKit descriptors
    z = mf.encode("CCO"); mf.decode(z)          # latent round-trip

CLI
---
    python molforge.py --n 20 --temperature 0.9
    python molforge.py --n 10 --spec '{"MolWt":300,"NumAromaticRings":1}'
    python molforge.py --n 10 --device cuda --out molecules.csv
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch


class MolForge:
    """Pretrained generative molecule model. ~100% valid SELFIES decoding."""

    def __init__(self, ckpt: Optional[str] = None, device: str = "cpu",
                 artifacts_dir: Optional[str] = None):
        if artifacts_dir:
            os.environ["MOLVAE_ART_DIR"] = artifacts_dir
        from molforge.core import config

        from molforge.core import infer

        self._config, self._infer = config, infer
        from molforge.core import data

        self._data = data
        ckpt = ckpt or str(config.CKPT_DIR / "best.pt")
        self.net, self.vocab, self.ck, self.device = infer.load_model(ckpt, device=device)
        self.props = config.PROPERTIES

    # ------------------------------------------------------------------ generate
    @torch.no_grad()
    def generate(self, n: int = 10, spec: Optional[Dict[str, float]] = None,
                 temperature: float = 0.9, z_scale: float = 1.0,
                 with_properties: bool = False, max_tries: int = 8) -> List:
        """Return n unique valid SMILES. If `spec` is given (e.g. {"MolWt":250}), generation
        is conditioned toward it. with_properties=True returns [{smiles, ...descriptors}]."""
        cond_spec = spec or {}
        cond = torch.from_numpy(self._data.spec_to_condition(cond_spec)).float()
        out, tries = [], 0
        seen = set()
        while len(out) < n and tries < max_tries:
            tries += 1
            b = max(n * 3, 64)
            c = cond.unsqueeze(0).repeat(b, 1).to(self.device)
            seqs = self.net.sample(b, c, self.vocab.bos, self.vocab.eos,
                                   temperature=temperature, z_scale=z_scale, device=self.device)
            for s in seqs.tolist():
                smi = self.vocab.decode_to_smiles(s)
                if smi and smi not in seen:
                    seen.add(smi); out.append(smi)
                    if len(out) >= n:
                        break
        out = out[:n]
        if with_properties:
            return [{"smiles": s, **(self.properties(s) or {})} for s in out]
        return out

    # ------------------------------------------------------------ encode / decode
    @torch.no_grad()
    def encode(self, smiles: str) -> Optional[np.ndarray]:
        """SMILES -> latent vector (mean), or None if it can't be encoded."""
        import selfies as sf
        canon = self._data.canonical_smiles(smiles)
        if not canon:
            return None
        try:
            sel = sf.encoder(canon)
        except Exception:
            return None
        if not sel or sf.len_selfies(sel) > self._config.MAX_SELFIES_LEN - 2:
            return None
        ids, length = self.vocab.encode(sel)
        toks = torch.tensor([ids], dtype=torch.long, device=self.device)
        mu, _ = self.net.encode(toks, torch.tensor([length]))
        return mu.cpu().numpy()[0]

    @torch.no_grad()
    def decode(self, z, spec: Optional[Dict[str, float]] = None, temperature: float = 0.0) -> Optional[str]:
        """Latent vector -> SMILES (greedy if temperature=0)."""
        z = torch.as_tensor(np.asarray(z, np.float32)).unsqueeze(0).to(self.device)
        cond = torch.from_numpy(self._data.spec_to_condition(spec or {})).float().unsqueeze(0).to(self.device)
        seq = self.net.sample(1, cond, self.vocab.bos, self.vocab.eos, z=z,
                              temperature=temperature, greedy=(temperature == 0.0), device=self.device)
        return self.vocab.decode_to_smiles(seq[0].tolist())

    # ------------------------------------------------------------------ utilities
    def properties(self, smiles: str) -> Optional[Dict[str, float]]:
        """RDKit descriptors (the 11 the model is conditioned on)."""
        return self._infer.score_smiles(smiles)

    def predict_properties(self, smiles: str) -> Optional[Dict[str, float]]:
        """The VAE's own property-head prediction from the molecule's latent."""
        z = self.encode(smiles)
        if z is None:
            return None
        mean, std = self._data.load_stats()
        with torch.no_grad():
            p = self.net.prop_head(torch.tensor(z, device=self.device).unsqueeze(0)).cpu().numpy()[0]
        return {n: round(float(v * std[i] + mean[i]), 3) for i, (n, v) in enumerate(zip(self.props, p))}


def _main():
    ap = argparse.ArgumentParser(description="MolForge generative model — local CLI")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--spec", type=str, default=None, help='JSON property targets, e.g. {"MolWt":300}')
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--out", default=None, help="optional CSV path")
    args = ap.parse_args()
    mf = MolForge(ckpt=args.ckpt, device=args.device, artifacts_dir=args.artifacts_dir)
    spec = json.loads(args.spec) if args.spec else None
    rows = mf.generate(args.n, spec=spec, temperature=args.temperature, with_properties=True)
    cols = ["smiles"] + mf.props
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in cols))
    if args.out:
        import csv
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    _main()
