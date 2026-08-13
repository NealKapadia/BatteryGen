"""Generate molecules from the trained conditional SELFIES-VAE.

  python -m batterygen.generative.generate --n 20
  python -m batterygen.generative.generate --n 20 --spec "{\"QED\":0.85,\"MolWt\":350}"
  python -m batterygen.generative.generate --n 20 --prompt "small soluble drug-like molecule"
  python -m batterygen.generative.generate --n 20 --spec "{\"QED\":0.85}" --molport-only

Every molecule is decoded from SELFIES (always syntactically valid), de-duplicated,
scored with RDKit, and checked against the Molport membership index (the Molport-xxx
id is shown when present). --molport-only keeps only molecules already in Molport.
"""
from __future__ import annotations

import argparse
import csv
import json
from typing import Dict, List

import torch

from batterygen.core import config

from batterygen.core import infer

from batterygen.core.membership import MolportIndex


def generate(net, vocab, spec: Dict[str, float], n: int, *, temperature: float,
             molport_only: bool, device, max_batches: int = 50) -> List[dict]:
    index = MolportIndex()
    results: List[dict] = []
    seen_smiles = set()
    batches = 0
    target = n * (8 if molport_only else 1)  # oversample when filtering to catalog
    while len(results) < n and batches < max_batches:
        batches += 1
        cond = infer.condition_tensor(spec, target, device)
        seqs = net.sample(target, cond, vocab.bos, vocab.eos,
                          temperature=temperature, device=device)
        for s in seqs.tolist():
            smi = vocab.decode_to_smiles(s)
            if not smi or smi in seen_smiles:
                continue
            seen_smiles.add(smi)
            mid = index.get_id(smi, already_canonical=True) if index.available else None
            in_mp = mid is not None
            if molport_only and not in_mp:
                continue
            props = infer.score_smiles(smi)
            if props is None:
                continue
            results.append({"smiles": smi, "molport_id": mid or "", "in_molport": in_mp, **props})
            if len(results) >= n:
                break
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--spec", type=str, default=None, help='JSON, e.g. {"QED":0.85,"MolWt":350}')
    ap.add_argument("--prompt", type=str, default=None, help="natural-language request")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--molport-only", action="store_true", help="only molecules already in Molport")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="optional CSV output path")
    args = ap.parse_args()

    spec = json.loads(args.spec) if args.spec else {}
    spec = infer.resolve_spec(spec, args.prompt)
    net, vocab, _, device = infer.load_model(args.ckpt)
    if spec:
        print("Property spec (raw units):", spec)
    print(f"Generating {args.n} molecules (T={args.temperature}"
          f"{', Molport-only' if args.molport_only else ''}) ...")

    rows = generate(net, vocab, spec, args.n, temperature=args.temperature,
                    molport_only=args.molport_only, device=device)

    if not rows:
        print("No valid molecules produced — try a higher --temperature or train longer.")
        return
    show = ["MolWt", "MolLogP", "TPSA", "QED", "NumHDonors", "NumHAcceptors"]
    print(f"\n{'#':>2}  {'SMILES':<40} {'Molport ID':<20} " + " ".join(f"{k:>8}" for k in show))
    for i, r in enumerate(rows, 1):
        props = " ".join(f"{r[k]:8.2f}" for k in show)
        print(f"{i:>2}  {r['smiles']:<40} {r['molport_id'] or '-':<20} {props}")
    n_in = sum(r["in_molport"] for r in rows)
    print(f"\n{len(rows)} molecules | {n_in} already in Molport, {len(rows)-n_in} novel")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
