"""Fine-tune the trained VAE on xTB (DFT-surrogate) labels.

A small ``dft_head`` is trained to predict xTB properties (homo/lumo/gap/dipole/...)
from the VAE latent. Optionally the encoder/decoder are lightly fine-tuned so the
latent space *organizes* by the DFT property. You can then generate molecules
toward an xTB target via latent optimization with that head.

  python molvae/xtb_label.py --n 400                 # make labels first
  python molvae/finetune_dft.py --target gap          # train the gap head
  python molvae/finetune_dft.py --target gap --finetune-encoder
  python molvae/finetune_dft.py --generate "{\"gap\":3.5}" --n 10 --verify
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from molforge.core import config

from molforge.core import data

from molforge.core import infer

from molforge.core import model as M


DFT_CKPT = config.CKPT_DIR / "dft_latest.pt"


class DftHead(nn.Module):
    def __init__(self, latent_dim: int, n_targets: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, z):
        return self.net(z)


def _encode_mols(net, vocab, smiles: List[str], device, batch=256):
    """Return (mu [N,latent], cond [N,nprops], kept_smiles)."""
    import selfies as sf

    mean, std = data.load_stats()
    std = np.where(std > 1e-8, std, 1.0)
    rows, conds, kept = [], [], []
    for smi in smiles:
        canon = data.canonical_smiles(smi)
        if canon is None:
            continue
        try:
            s = sf.encoder(canon)
        except Exception:
            continue
        if not s or sf.len_selfies(s) > config.MAX_SELFIES_LEN - 2:
            continue
        from rdkit import Chem

        mol = Chem.MolFromSmiles(canon)
        ids, length = vocab.encode(s)
        rows.append((ids, length))
        conds.append((np.asarray(data.compute_descriptors(mol), np.float32) - mean) / std)
        kept.append(canon)

    mus = []
    net.eval()
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            toks = torch.tensor([r[0] for r in chunk], dtype=torch.long, device=device)
            lens = torch.tensor([r[1] for r in chunk], dtype=torch.long)
            maxlen = int(lens.max())
            mu, _ = net.encode(toks[:, :maxlen], lens)
            mus.append(mu)
    mu = torch.cat(mus) if mus else torch.empty(0, config.LATENT_DIM, device=device)
    cond = torch.tensor(np.stack(conds), dtype=torch.float32, device=device) if conds else None
    return mu, cond, kept


def load_labels(path: Path, targets: List[str]) -> Tuple[List[str], np.ndarray]:
    smiles, ys = [], []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                y = [float(r[t]) for t in targets]
            except (ValueError, KeyError, TypeError):
                continue
            smiles.append(r["smiles"])
            ys.append(y)
    return smiles, np.asarray(ys, np.float32)


def train(args):
    net, vocab, _, device = infer.load_model(args.ckpt)
    targets = [t.strip() for t in args.target.split(",") if t.strip()]
    if not Path(args.labels).exists():
        raise SystemExit(f"No labels at {args.labels}. Run xtb_label.py first.")
    smiles, y = load_labels(Path(args.labels), targets)
    if len(smiles) < 16:
        raise SystemExit(f"Only {len(smiles)} usable labels — label more molecules first.")
    print(f"Training DFT head on {len(smiles)} molecules, targets={targets}")

    y_mean, y_std = y.mean(0), y.std(0) + 1e-6
    y_norm = torch.tensor((y - y_mean) / y_std, dtype=torch.float32, device=device)

    mu, cond, kept = _encode_mols(net, vocab, smiles, device)
    y_norm = y_norm[: len(kept)]  # align with successfully encoded molecules

    head = DftHead(config.LATENT_DIM, len(targets)).to(device)
    # ---- phase 1: train the head on frozen latents --------------------------
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    n_val = max(1, len(kept) // 5)
    perm = torch.randperm(len(kept), device=device)
    tr, va = perm[n_val:], perm[:n_val]
    for ep in range(args.epochs):
        head.train()
        opt.zero_grad()
        loss = F.mse_loss(head(mu[tr]), y_norm[tr])
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, args.epochs // 5) == 0:
            head.eval()
            with torch.no_grad():
                vmse = F.mse_loss(head(mu[va]), y_norm[va]).item()
            print(f"  [head] epoch {ep+1}/{args.epochs} train_mse={loss.item():.4f} val_mse={vmse:.4f}")

    # ---- phase 2 (optional): light end-to-end fine-tune ---------------------
    if args.finetune_encoder:
        print("  fine-tuning encoder/decoder + head end-to-end ...")
        toks_lens = _tokenize(vocab, kept)
        opt2 = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=2e-5)
        for ep in range(args.ft_epochs):
            net.train(); head.train()
            idx = torch.randperm(len(kept), device=device)
            for i in range(0, len(kept), 64):
                b = idx[i:i + 64]
                toks = torch.stack([toks_lens[0][j] for j in b]).to(device)
                lens = toks_lens[1][b.cpu()]
                cb = cond[b]
                opt2.zero_grad()
                logits, mu_b, logvar, prop = net(toks, lens, cb)
                total, _ = M.vae_loss(logits, toks[:, 1:], mu_b, logvar, prop, cb, vocab.pad,
                                      beta=config.KL_WEIGHT_MAX)
                dft = F.mse_loss(head(mu_b), y_norm[b])
                (total + 5.0 * dft).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.GRAD_CLIP)
                opt2.step()
            print(f"  [ft] epoch {ep+1}/{args.ft_epochs}")

    torch.save({
        "model": net.state_dict(), "dft_head": head.state_dict(),
        "dft_targets": targets, "dft_mean": y_mean.tolist(), "dft_std": y_std.tolist(),
        "hparams": {"emb_dim": config.EMB_DIM, "enc_hidden": config.ENC_HIDDEN,
                    "dec_hidden": config.DEC_HIDDEN, "latent_dim": config.LATENT_DIM,
                    "enc_layers": config.ENC_LAYERS, "dec_layers": config.DEC_LAYERS},
    }, DFT_CKPT)
    print(f"Saved DFT-fine-tuned model -> {DFT_CKPT}")


def _tokenize(vocab, smiles):
    import selfies as sf

    toks, lens = [], []
    for canon in smiles:
        ids, length = vocab.encode(sf.encoder(canon))
        toks.append(torch.tensor(ids, dtype=torch.long))
        lens.append(length)
    return toks, torch.tensor(lens, dtype=torch.long)


def generate(args):
    if not DFT_CKPT.exists():
        raise SystemExit("No DFT checkpoint. Run training first: finetune_dft.py --target gap")
    device = config.get_device()
    vocab = data.Vocab.load()
    ck = torch.load(DFT_CKPT, map_location=device)
    net = M.SelfiesVAE(vocab_size=len(vocab), pad_idx=vocab.pad, **ck["hparams"]).to(device)
    net.load_state_dict(ck["model"]); net.eval()
    head = DftHead(config.LATENT_DIM, len(ck["dft_targets"])).to(device)
    head.load_state_dict(ck["dft_head"]); head.eval()

    spec = json.loads(args.generate)
    tmean = np.asarray(ck["dft_mean"], np.float32)
    tstd = np.asarray(ck["dft_std"], np.float32)
    target = torch.zeros(len(ck["dft_targets"]), device=device)
    mask = torch.zeros(len(ck["dft_targets"]), dtype=torch.bool, device=device)
    for i, t in enumerate(ck["dft_targets"]):
        if t in spec:
            target[i] = float((float(spec[t]) - tmean[i]) / tstd[i])
            mask[i] = True
    if not bool(mask.any()):
        raise SystemExit(f"--generate spec must mention one of {ck['dft_targets']}")

    z = torch.randn(max(args.n * 6, 48), config.LATENT_DIM, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.1)
    for _ in range(args.steps):
        opt.zero_grad()
        loss = ((head(z) - target)[:, mask] ** 2).mean() + 1e-3 * (z ** 2).mean()
        loss.backward(); opt.step()

    cond = torch.zeros(z.size(0), config.N_PROPS, device=device)  # neutral RDKit props
    with torch.no_grad():
        seqs = net.sample(z.size(0), cond, vocab.bos, vocab.eos,
                          temperature=args.temperature, z=z.detach(), device=device)
    from molforge.core.membership import MolportIndex

    index = MolportIndex()
    rows, seen = [], set()
    for s in seqs.tolist():
        smi = vocab.decode_to_smiles(s)
        if not smi or smi in seen:
            continue
        seen.add(smi)
        rows.append(smi)
        if len(rows) >= args.n:
            break

    verified = {}
    if args.verify and rows:
        import tempfile
        from molforge.grounding import xtb_label


        scratch = Path(tempfile.mkdtemp(prefix="xtbv_"))
        import os

        for smi in rows:
            r = xtb_label.run_xtb(smi, scratch, opt=False, charge=0, timeout=120, env=dict(os.environ))
            if r:
                verified[smi] = r

    print(f"\nTarget xTB spec: {spec}")
    print(f"{'#':>2}  {'SMILES':<44} {'Molport ID':<18} " +
          " ".join(f"{t:>10}" for t in ck["dft_targets"] if t in spec))
    for i, smi in enumerate(rows, 1):
        mid = index.get_id(smi, already_canonical=True) if index.available else None
        vals = ""
        if smi in verified:
            vals = " ".join(f"{verified[smi].get(t, float('nan')):10.3f}" for t in ck["dft_targets"] if t in spec)
        print(f"{i:>2}  {smi:<44} {mid or '-':<18} {vals}")
    if args.verify:
        print("(values are xTB-verified for the generated molecules)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default="gap", help="comma list of xTB targets")
    ap.add_argument("--labels", type=str, default=str(config.DFT_DIR / "labels.csv"))
    ap.add_argument("--epochs", type=int, default=300, help="head training epochs")
    ap.add_argument("--finetune-encoder", action="store_true")
    ap.add_argument("--ft-epochs", type=int, default=3)
    ap.add_argument("--ckpt", type=str, default=None, help="base VAE checkpoint")
    # generation mode
    ap.add_argument("--generate", type=str, default=None, help='JSON xTB target, e.g. {"gap":3.5}')
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--verify", action="store_true", help="re-run xTB on generated molecules")
    args = ap.parse_args()

    if args.generate:
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
