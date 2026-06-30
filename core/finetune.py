"""
finetune.py  —  fine-tune the pretrained SELFIES-VAE on YOUR OWN molecules.
==========================================================================
Unlike ``train.py`` (which continues training over the original preprocessed Molport
shards), this script needs **none of those shards**. It builds a small in-memory
training set directly from a plain SMILES file you supply, loads the pretrained
weights (``best.pt`` from Hugging Face), trains a few epochs, and writes a new
checkpoint you can load straight back into ``MolForge``.

So it runs from a bare ``pip install`` + the Hugging Face artifacts:
    artifacts/processed/{vocab.json, descriptor_stats.json}   (small, on HF)
    artifacts/checkpoints/best.pt                              (the weights, on HF)

It works on CPU or GPU. CPU is fine for a few thousand molecules / a few epochs but
slow; pass --device cuda if you have an NVIDIA GPU.

Quick start
-----------
    # point at a WRITABLE artifacts dir that holds the HF download
    #   (Windows)  $env:MOLVAE_ART_DIR = "C:\\path\\to\\artifacts"
    #   (bash)     export MOLVAE_ART_DIR=/path/to/artifacts

    python -m molforge.finetune --input my_molecules.smi --epochs 3 --device cuda
    python -m molforge.finetune --input solvents.csv --smiles-col 0 --header --delim ,

Then use the result:
    from molforge import MolForge
    mf = MolForge(device="cpu", ckpt="<MOLVAE_ART_DIR>/checkpoints/finetuned.pt")
    mf.generate(20)
"""
from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config
import data
import infer
import model as M

try:  # optional pretty progress; never a hard dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x=None, **k):
        return x if x is not None else None


# --------------------------------------------------------------------------- #
# Read SMILES from .smi / .txt / .csv (optionally .gz)
# --------------------------------------------------------------------------- #
def read_smiles(paths, smiles_col: int, delim, header: bool, limit: int):
    smis = []
    for p in paths:
        p = Path(p)
        opener = gzip.open if p.name.lower().endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                if header and i == 0:
                    continue
                line = line.strip()
                if not line:
                    continue
                parts = line.split(delim) if delim else line.split()
                if smiles_col >= len(parts):
                    continue
                smis.append(parts[smiles_col].strip())
                if limit and len(smis) >= limit:
                    break
        if limit and len(smis) >= limit:
            break
    return smis


# --------------------------------------------------------------------------- #
# Build in-memory tensors via the SAME pipeline used for the base model
# --------------------------------------------------------------------------- #
def build_dataset(smiles, vocab):
    """SMILES -> (tokens int16 [N, L], lengths int16 [N], desc float32 [N, P]).
    Rejects anything the base preprocessor would reject (filters, length cap)."""
    toks, lens, descs = [], [], []
    kept = 0
    for smi in smiles:
        rec = data.process_record(smi, "", vocab)
        if rec is None:
            continue
        _canon, _mid, ids, length, desc = rec
        toks.append(np.asarray(ids, dtype=np.int16))
        lens.append(length)
        descs.append(np.asarray(desc, dtype=np.float32))
        kept += 1
    if kept == 0:
        raise SystemExit(
            "No valid molecules survived filtering. Check the SMILES column / file format "
            "(--smiles-col, --delim, --header)."
        )
    return (np.stack(toks), np.asarray(lens, dtype=np.int16), np.stack(descs))


class MemDataset(Dataset):
    """In-memory dataset returning (tokens, length, normalized-descriptors)."""

    def __init__(self, toks, lens, desc, mean, std):
        self.toks, self.lens, self.desc = toks, lens, desc
        self.mean = mean
        self.std = np.where(std > 1e-8, std, 1.0).astype(np.float32)

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, i):
        d = (self.desc[i] - self.mean) / self.std
        return (
            torch.from_numpy(self.toks[i].astype(np.int64)),
            int(self.lens[i]),
            torch.from_numpy(d.astype(np.float32)),
        )


@torch.no_grad()
def sample_validity(net, vocab, device, n=200):
    cond = torch.zeros(n, config.N_PROPS, device=device)
    seqs = net.sample(n, cond, vocab.bos, vocab.eos, temperature=0.8, device=device)
    smis = [vocab.decode_to_smiles(s) for s in seqs.tolist()]
    valid = [s for s in smis if s]
    return len(valid) / n, valid[:5]


def main():
    ap = argparse.ArgumentParser(description="Fine-tune the MolForge SELFIES-VAE on your own SMILES.")
    ap.add_argument("--input", nargs="+", required=True, help="SMILES file(s): .smi/.txt/.csv[.gz]")
    ap.add_argument("--smiles-col", type=int, default=0, help="column index of the SMILES (default 0)")
    ap.add_argument("--delim", default=None, help="field delimiter (default: any whitespace)")
    ap.add_argument("--header", action="store_true", help="input has a header row to skip")
    ap.add_argument("--limit", type=int, default=0, help="cap molecules read (0=all)")
    ap.add_argument("--ckpt", default=None, help="starting weights (default: <artifacts>/checkpoints/best.pt)")
    ap.add_argument("--out", default="finetuned.pt", help="output checkpoint name or path")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64, help="lower if you hit OOM; raise on a big GPU")
    ap.add_argument("--lr", type=float, default=1e-4, help="fine-tuning LR (lower than base 5e-4)")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=config.KL_WEIGHT_MAX, help="KL weight (kept gentle)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--device", default=None, help="cpu | cuda (default: cuda if available)")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = args.ckpt or str(config.CKPT_DIR / "best.pt")

    print(f"Loading pretrained weights: {ckpt}")
    net, vocab, _ck, device = infer.load_model(ckpt, device=device)

    print(f"Reading molecules from: {', '.join(args.input)}")
    raw = read_smiles(args.input, args.smiles_col, args.delim, args.header, args.limit)
    toks, lens, desc = build_dataset(raw, vocab)
    print(f"Kept {len(lens):,} / {len(raw):,} molecules after canonicalize + filter.")

    mean, std = data.load_stats()  # reuse base normalization so conditioning stays calibrated
    n_val = max(1, int(len(lens) * args.val_frac)) if len(lens) > 20 else 0
    perm = np.random.permutation(len(lens))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    collate = data.make_collate(vocab.pad)
    use_amp = (device.type == "cuda") and not args.no_amp
    train_dl = DataLoader(
        MemDataset(toks[tr_idx], lens[tr_idx], desc[tr_idx], mean, std),
        batch_size=args.batch, shuffle=True, drop_last=len(tr_idx) >= args.batch, collate_fn=collate,
    )
    val_dl = None
    if n_val:
        val_dl = DataLoader(
            MemDataset(toks[val_idx], lens[val_idx], desc[val_idx], mean, std),
            batch_size=args.batch, shuffle=False, collate_fn=collate,
        )

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Device: {device} | AMP: {use_amp} | batch: {args.batch} | epochs: {args.epochs} | "
          f"params: {M.count_params(net):,}")

    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        running = {"recon": 0.0, "kl": 0.0, "prop": 0.0}
        nb = 0
        bar = tqdm(train_dl, desc=f"epoch {epoch+1}/{args.epochs}", unit="batch")
        for toks_b, lens_b, cond_b in bar:
            toks_b, cond_b = toks_b.to(device), cond_b.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, mu, logvar, prop = net(toks_b, lens_b, cond_b)
                total, comps = M.vae_loss(
                    logits, toks_b[:, 1:], mu, logvar, prop, cond_b, vocab.pad, beta=args.beta
                )
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            for k in running:
                running[k] += comps[k]
            nb += 1
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(recon=f"{comps['recon']:.2f}", kl=f"{comps['kl']:.2f}",
                                prop=f"{comps['prop']:.3f}")

        avg = {k: running[k] / max(nb, 1) for k in running}
        msg = f"epoch {epoch+1}: recon={avg['recon']:.3f} kl={avg['kl']:.3f} prop={avg['prop']:.3f}"

        if val_dl is not None:
            net.eval()
            tot_recon = tot_tok = correct = 0
            with torch.no_grad():
                for toks_b, lens_b, cond_b in val_dl:
                    toks_b, cond_b = toks_b.to(device), cond_b.to(device)
                    logits, mu, logvar, prop = net(toks_b, lens_b, cond_b)
                    targets = toks_b[:, 1:]
                    mask = targets != vocab.pad
                    correct += int((logits.argmax(-1)[mask] == targets[mask]).sum())
                    tot_tok += int(mask.sum())
            msg += f" | val_token_acc={correct / max(tot_tok, 1):.3f}"
        print(msg)

    rate, examples = sample_validity(net, vocab, device)
    print(f"Post-finetune sample validity: {rate:.3f}  e.g. {examples}")

    # Save a checkpoint compatible with infer.load_model / MolForge.
    out = Path(args.out)
    if not out.is_absolute() and out.parent == Path("."):
        config.ensure_dirs()
        out = config.CKPT_DIR / out.name
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "properties": config.PROPERTIES,
            "vocab_size": len(vocab),
            "pad_idx": vocab.pad,
            "hparams": {
                "emb_dim": config.EMB_DIM, "enc_hidden": config.ENC_HIDDEN,
                "dec_hidden": config.DEC_HIDDEN, "latent_dim": config.LATENT_DIM,
                "enc_layers": config.ENC_LAYERS, "dec_layers": config.DEC_LAYERS,
            },
            "finetuned_on": [str(p) for p in args.input],
            "n_train": int(len(tr_idx)),
        },
        out,
    )
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved -> {out}")
    print(f"Use it:  MolForge(ckpt=r\"{out}\")")


if __name__ == "__main__":
    main()
