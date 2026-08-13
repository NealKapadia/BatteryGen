"""Measure whether the model learned chemistry or memorized.

Reports, on the held-out (scaffold) validation set + on prior samples:
  * reconstruction   : exact round-trip rate + token accuracy on UNSEEN scaffolds
  * property head     : R^2 / MAE per descriptor on unseen molecules
  * generation        : validity, uniqueness, novelty (vs training set),
                        internal diversity, and property-distribution match

High reconstruction/property scores on held-out scaffolds + high novelty/diversity
= generalization. High train scores but poor held-out = memorization.

  python -m batterygen.generative.evaluate --n 3000 --gen 3000
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from batterygen.core import config

from batterygen.core import data

from batterygen.core import infer

from batterygen.core.membership import MolportIndex


@torch.no_grad()
def reconstruction_and_property(net, vocab, device, n):
    ds = data.MolDataset("val")
    n = min(n, len(ds))
    dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=data.make_collate(vocab.pad))
    mean, std = data.load_stats(); std = np.where(std > 1e-8, std, 1.0)
    exact = tok_correct = tok_total = seen = 0
    preds, trues = [], []
    for toks, lens, cond in tqdm(dl, desc="recon/prop", unit="batch", dynamic_ncols=True):
        toks, cond = toks.to(device), cond.to(device)
        mu, logvar = net.encode(toks, lens)
        # teacher-forced token accuracy
        logits = net.decode(mu, cond, toks[:, :-1])
        tgt = toks[:, 1:]; m = tgt != vocab.pad
        tok_correct += int((logits.argmax(-1)[m] == tgt[m]).sum()); tok_total += int(m.sum())
        # free-running greedy reconstruction from z=mu
        seqs = net.sample(toks.size(0), cond, vocab.bos, vocab.eos, z=mu, device=device, greedy=True)
        for row, s in zip(toks.tolist(), seqs.tolist()):
            orig = vocab.decode_to_smiles(row)
            rec = vocab.decode_to_smiles(s)
            if orig is not None and rec == orig:
                exact += 1
        preds.append(net.prop_head(mu).cpu().numpy()); trues.append(cond.cpu().numpy())
        seen += toks.size(0)
        if seen >= n:
            break
    pred = np.concatenate(preds)[:n]; true = np.concatenate(trues)[:n]
    # de-normalize for human-readable MAE
    pred_r, true_r = pred * std + mean, true * std + mean
    prop = {}
    for i, name in enumerate(config.PROPERTIES):
        ss_res = float(((true[:, i] - pred[:, i]) ** 2).sum())
        ss_tot = float(((true[:, i] - true[:, i].mean()) ** 2).sum()) or 1.0
        prop[name] = {"r2": round(1 - ss_res / ss_tot, 3),
                      "mae": round(float(np.abs(pred_r[:, i] - true_r[:, i]).mean()), 3)}
    return {"exact_recon_rate": round(exact / seen, 3),
            "token_acc": round(tok_correct / max(tok_total, 1), 3),
            "n": seen}, prop


@torch.no_grad()
def generation_metrics(net, vocab, device, m, temperature=0.9, z_scale=1.0, pubchem_bloom=None):
    index = MolportIndex()
    smis = []
    for i in range(0, m, 512):
        b = min(512, m - i)
        cond = torch.zeros(b, config.N_PROPS, device=device)
        seqs = net.sample(b, cond, vocab.bos, vocab.eos,
                          temperature=temperature, z_scale=z_scale, device=device)
        smis.extend(vocab.decode_to_smiles(s) for s in seqs.tolist())
    valid = [s for s in smis if s]
    uniq = list(dict.fromkeys(valid))
    novel = [s for s in uniq if not (index.available and index.contains(s, already_canonical=True))]
    nov_pubchem = None
    if pubchem_bloom is not None:
        nov_pubchem = round(sum(1 for s in uniq if s not in pubchem_bloom) / max(len(uniq), 1), 3)
    # internal pairwise Tanimoto on up to 1000 unique molecules (paper's definition:
    # Diversity = 1 - mean pairwise similarity; Similarity = that mean).
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    samp = uniq[:1000]
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
           for s in samp if Chem.MolFromSmiles(s)]
    sims = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    mean_sim = float(np.mean(sims)) if sims else None
    # property-distribution match: generated vs dataset means
    mean, std = data.load_stats()
    gprops = np.array([data.descriptors_for_smiles(s) for s in valid[:1000] if data.descriptors_for_smiles(s)])
    dist = {}
    if len(gprops):
        for i, name in enumerate(config.PROPERTIES):
            dist[name] = {"gen_mean": round(float(gprops[:, i].mean()), 2),
                          "data_mean": round(float(mean[i]), 2)}
    return {
        "validity": round(len(valid) / max(len(smis), 1), 3),
        "uniqueness": round(len(uniq) / max(len(valid), 1), 3),
        "novelty_vs_training": round(len(novel) / max(len(uniq), 1), 3),
        "novelty_vs_pubchem": nov_pubchem,
        "similarity": round(mean_sim, 3) if mean_sim is not None else None,
        "diversity": round(1 - mean_sim, 3) if mean_sim is not None else None,
        "n_generated": len(smis),
        "property_dist": dist,
    }


# ElectrolyteGPT paper (JACS Au 2026) Table 1, for side-by-side comparison.
_PAPER_TABLE = {
    "Mol-CycleGAN": [0.923, 0.996, 0.951, 0.869, 0.538, 0.462],
    "JT-VAE":       [0.999, 0.999, 0.916, 0.813, 0.604, 0.396],
    "MinGPT":       [0.970, 0.991, 0.712, 0.581, 0.407, 0.593],
    "1Ddiffusion":  [0.219, 0.995, 0.971, 0.904, 0.317, 0.683],
    "Diffusion-LM": [0.278, 0.998, 0.990, 0.925, 0.324, 0.676],
    "ElectrolyteGPT": [0.995, 0.999, 0.997, 0.942, 0.421, 0.579],
}
_COLS = ["Validity", "Uniqueness", "Novelty(train)", "Novelty(PubChem)", "Similarity", "Diversity"]


def print_benchmark_table(gen: dict):
    ours = [gen["validity"], gen["uniqueness"], gen["novelty_vs_training"],
            gen["novelty_vs_pubchem"], gen["similarity"], gen["diversity"]]
    print("\n== Benchmark vs ElectrolyteGPT paper (Table 1) ==")
    hdr = f"{'Model':<16}" + "".join(f"{c:>17}" for c in _COLS)
    print(hdr)
    for name, vals in _PAPER_TABLE.items():
        print(f"{name:<16}" + "".join(f"{v:>17}" for v in vals))
    print("-" * len(hdr))
    print(f"{'OURS (batterygen)':<16}" + "".join(
        f"{('n/a' if v is None else f'{v:.3f}'):>17}" for v in ours))
    print("\nNote: paper numbers are on a 1M *electrolyte* dataset. Compare fairly AFTER "
          "electrolyte specialization (add_data.py). Novelty(PubChem) needs --pubchem-bloom.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="held-out molecules for recon/property")
    ap.add_argument("--gen", type=int, default=3000, help="molecules to sample for gen metrics")
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="softmax temperature for prior sampling (lower = safer/less diverse)")
    ap.add_argument("--z-scale", type=float, default=1.0,
                    help="latent prior std (z ~ N(0, z_scale^2 I); <1 raises validity)")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--out", type=str, default=str(config.CKPT_DIR / "eval_report.json"))
    ap.add_argument("--pubchem-bloom", type=str, default=None,
                    help="prebuilt PubChem Bloom file for novelty-vs-PubChem")
    ap.add_argument("--build-pubchem-from", type=str, default=None,
                    help="build a PubChem Bloom from a SMILES file, then exit")
    ap.add_argument("--build-out", type=str, default=str(config.MEMBER_DIR / "pubchem.bloom"))
    args = ap.parse_args()

    if args.build_pubchem_from:
        from batterygen.core.membership import BloomFilter
        from pathlib import Path
        n = 0
        bloom = BloomFilter(120_000_000, 1e-4)  # PubChem-scale
        with open(args.build_pubchem_from, encoding="utf-8", errors="ignore") as f:
            for line in tqdm(f, desc="PubChem", unit="mol"):
                tok = line.strip().split()[0].split(",")[0].split("\t")[0]
                canon = data.canonical_smiles(tok)
                if canon:
                    bloom.add(canon); n += 1
        bloom.save(Path(args.build_out))
        print(f"Built PubChem Bloom ({n:,} molecules) -> {args.build_out}")
        return

    from pathlib import Path
    pubchem_bloom = None
    bloom_path = Path(args.pubchem_bloom) if args.pubchem_bloom else (config.MEMBER_DIR / "pubchem.bloom")
    if bloom_path.exists():
        from batterygen.core.membership import BloomFilter
        pubchem_bloom = BloomFilter.load(bloom_path)
        print(f"Using PubChem bloom: {bloom_path.name}")

    net, vocab, ck, device = infer.load_model(args.ckpt)
    split_info = "scaffold-held-out" if (config.PROC_DIR / "split_val.npy").exists() else "random"
    print(f"Evaluating on {split_info} validation split | device {device}\n")

    recon, prop = reconstruction_and_property(net, vocab, device, args.n)
    print(f"Generation sampling: temperature={args.temperature}, z_scale={args.z_scale}")
    gen = generation_metrics(net, vocab, device, args.gen, temperature=args.temperature,
                             z_scale=args.z_scale, pubchem_bloom=pubchem_bloom)

    print(f"== Reconstruction ({split_info}, n={recon['n']}) ==")
    print(f"  exact round-trip : {recon['exact_recon_rate']:.1%}   token acc : {recon['token_acc']:.1%}")
    print("\n== Property head (R^2 / MAE on unseen molecules) ==")
    for name, v in prop.items():
        print(f"  {name:20s} R2={v['r2']:6.3f}   MAE={v['mae']}")
    print("\n== Generation ==")
    for k in ("validity", "uniqueness", "novelty_vs_training", "novelty_vs_pubchem",
              "similarity", "diversity"):
        print(f"  {k:22s}: {gen[k]}")
    print_benchmark_table(gen)

    report = {"split": split_info, "sampling": {"temperature": args.temperature, "z_scale": args.z_scale},
              "reconstruction": recon, "property": prop, "generation": gen}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {args.out}")
    print("Read: high held-out recon/property R^2 WITH high novelty/diversity = real chemistry, "
          "not memorization.")


if __name__ == "__main__":
    main()
