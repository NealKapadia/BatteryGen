"""Search molecules by property.

Two modes:

  --mode dataset : retrieve molecules from the Molport catalog whose properties fall
                   in the requested ranges (exact, fast, over precomputed descriptors).
  --mode latent  : optimize points in the VAE latent space toward a property target
                   using the property head, then decode to (novel) molecules.

Spec values may be a single number (matched within --tol) or a [low, high] range.

  python molvae/search.py --mode dataset --spec "{\"MolWt\":[300,350],\"QED\":[0.85,1.0]}"
  python molvae/search.py --mode latent  --spec "{\"MolWt\":320,\"QED\":0.9,\"TPSA\":60}" --n 10
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import torch

from molforge.core import config

from molforge.core import data

from molforge.core import infer

from molforge.core.membership import MolportIndex


def _bounds(spec: Dict, tol: float) -> Dict[str, Tuple[float, float]]:
    """Normalize each spec entry to a (low, high) band."""
    out = {}
    for key, val in spec.items():
        prop = config.PROPERTY_ALIASES.get(str(key).lower().strip(), key)
        if prop not in config.PROPERTIES:
            continue
        if isinstance(val, (list, tuple)) and len(val) == 2:
            out[prop] = (float(val[0]), float(val[1]))
        else:
            v = float(val)
            pad = abs(v) * tol if v != 0 else tol
            out[prop] = (v - pad, v + pad)
    return out


def search_dataset(spec: Dict, n: int, tol: float) -> List[dict]:
    bounds = _bounds(spec, tol)
    if not bounds:
        raise SystemExit("No valid properties in --spec")
    idx = {p: i for i, p in enumerate(config.PROPERTIES)}
    cols = {idx[p]: b for p, b in bounds.items()}
    hits: List[dict] = []
    # stream shard descriptor matrices for speed
    import json as _json

    meta = _json.load(open(config.META_PATH, encoding="utf-8"))
    for shard in meta["shards"]:
        desc = np.load(config.PROC_DIR / f"{shard}_desc.npy")
        mask = np.ones(len(desc), dtype=bool)
        for c, (lo, hi) in cols.items():
            mask &= (desc[:, c] >= lo) & (desc[:, c] <= hi)
        rows = np.where(mask)[0]
        if len(rows) == 0:
            continue
        with open(config.PROC_DIR / f"{shard}_ids.txt", encoding="utf-8") as f:
            lines = f.read().splitlines()
        for j in rows:
            canon, _, mid = lines[j].partition("\t")
            rec = {"smiles": canon, "molport_id": mid}
            rec.update({p: float(desc[j, idx[p]]) for p in config.PROPERTIES})
            hits.append(rec)
            if len(hits) >= n:
                return hits
    return hits


def search_latent(spec: Dict, n: int, steps: int, lr: float, temperature: float,
                  ckpt) -> List[dict]:
    net, vocab, _, device = infer.load_model(ckpt)
    mean, std = data.load_stats()
    target = torch.from_numpy(data.spec_to_condition(spec)).float().to(device)
    mask = torch.from_numpy(infer.specified_mask(spec)).to(device)

    z = torch.randn(max(n * 4, 32), config.LATENT_DIM, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = net.prop_head(z)
        prop_loss = ((pred - target)[:, mask] ** 2).mean()
        prior = 1e-3 * (z ** 2).mean()  # stay near the prior so decoding works
        (prop_loss + prior).backward()
        opt.step()

    cond = target.unsqueeze(0).repeat(z.size(0), 1)
    with torch.no_grad():
        seqs = net.sample(z.size(0), cond, vocab.bos, vocab.eos,
                          temperature=temperature, z=z.detach(), device=device)
    index = MolportIndex()
    out, seen = [], set()
    for s in seqs.tolist():
        smi = vocab.decode_to_smiles(s)
        if not smi or smi in seen:
            continue
        seen.add(smi)
        props = infer.score_smiles(smi)
        if props is None:
            continue
        mid = index.get_id(smi, already_canonical=True) if index.available else None
        out.append({"smiles": smi, "molport_id": mid or "", **props})
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dataset", "latent"], default="dataset")
    ap.add_argument("--spec", type=str, required=True, help="JSON property spec")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.1, help="fractional tolerance for scalar specs")
    ap.add_argument("--steps", type=int, default=300, help="latent-mode optimization steps")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

    spec = json.loads(args.spec)
    print(f"[{args.mode}] target spec: {spec}")
    if args.mode == "dataset":
        rows = search_dataset(spec, args.n, args.tol)
    else:
        rows = search_latent(spec, args.n, args.steps, args.lr, args.temperature, args.ckpt)

    if not rows:
        print("No matches found.")
        return
    show = [p for p in spec if config.PROPERTY_ALIASES.get(str(p).lower().strip(), p) in config.PROPERTIES]
    show = [config.PROPERTY_ALIASES.get(str(p).lower().strip(), p) for p in show] or ["MolWt", "QED"]
    print(f"\n{'#':>2}  {'SMILES':<42} {'Molport ID':<20} " + " ".join(f"{k:>9}" for k in show))
    for i, r in enumerate(rows, 1):
        vals = " ".join(f"{r.get(k, float('nan')):9.2f}" for k in show)
        print(f"{i:>2}  {r['smiles']:<42} {r.get('molport_id') or '-':<20} {vals}")
    print(f"\n{len(rows)} molecule(s).")


if __name__ == "__main__":
    main()
