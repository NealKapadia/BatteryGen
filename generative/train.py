"""Train the conditional SELFIES-VAE on the preprocessed Molport molecules.

* Runs on CUDA (your RTX 3060) with mixed precision.
* tqdm progress bar shows molecules-seen, recon / KL / property loss, lr, beta, VRAM.
* Checkpoints every CHECKPOINT_EVERY molecules (default 500,000) to
  molvae_artifacts/checkpoints/, plus latest.pt for resuming.

Usage:
    python molvae/train.py --epochs 3
    python molvae/train.py --resume
    python molvae/train.py --max-mols 100000 --ckpt-every 20000   # quick run
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from molforge.core import config

from molforge.core import data

from molforge.core import model as M



def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cyclical_beta(seen: int, total: int) -> float:
    """KL weight: ramp 0->max over the first KL_RATIO of each of KL_CYCLES cycles, then hold."""
    if total <= 0:
        return config.KL_WEIGHT_MAX
    period = total / max(1, config.KL_CYCLES)
    tau = (seen % period) / period
    if tau >= config.KL_RATIO:
        return config.KL_WEIGHT_MAX
    return config.KL_WEIGHT_MAX * (tau / config.KL_RATIO)


def lr_at(seen: int, total: int) -> float:
    """Linear warm-up then cosine decay to LR_MIN."""
    if seen < config.LR_WARMUP_MOLECULES:
        return config.LR * seen / max(1, config.LR_WARMUP_MOLECULES)
    p = min(1.0, max(0.0, (seen - config.LR_WARMUP_MOLECULES) / max(1, total - config.LR_WARMUP_MOLECULES)))
    return config.LR_MIN + 0.5 * (config.LR - config.LR_MIN) * (1 + math.cos(math.pi * p))


def save_checkpoint(path: Path, net, opt, scaler, molecules_seen, epoch, vocab):
    torch.save(
        {
            "model": net.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "molecules_seen": molecules_seen,
            "epoch": epoch,
            "vocab_size": len(vocab),
            "pad_idx": vocab.pad,
            "properties": config.PROPERTIES,
            "hparams": {
                "emb_dim": config.EMB_DIM, "enc_hidden": config.ENC_HIDDEN,
                "dec_hidden": config.DEC_HIDDEN, "latent_dim": config.LATENT_DIM,
                "enc_layers": config.ENC_LAYERS, "dec_layers": config.DEC_LAYERS,
            },
        },
        path,
    )


def prune_checkpoints(keep: int):
    ckpts = sorted(config.CKPT_DIR.glob("checkpoint_mols_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    for p in ckpts[:-keep] if keep > 0 else []:
        p.unlink(missing_ok=True)


@torch.no_grad()
def validate(net, val_dl, vocab, device, max_batches=20):
    net.eval()
    tot_recon = tot_tok = tot_correct = 0
    for bi, (toks, lens, cond) in enumerate(val_dl):
        if bi >= max_batches:
            break
        toks, cond = toks.to(device), cond.to(device)
        logits, mu, logvar, prop = net(toks, lens, cond)
        targets = toks[:, 1:]
        mask = targets != vocab.pad
        pred = logits.argmax(-1)
        tot_correct += int((pred[mask] == targets[mask]).sum())
        tot_tok += int(mask.sum())
        tot_recon += float(torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=vocab.pad, reduction="sum"))
    # prior-sample validity rate
    n = 200
    condz = torch.zeros(n, config.N_PROPS, device=device)
    seqs = net.sample(n, condz, vocab.bos, vocab.eos, temperature=0.8, device=device)
    valid = sum(1 for s in seqs.tolist() if vocab.decode_to_smiles(s))
    return {
        "val_token_acc": tot_correct / max(tot_tok, 1),
        "val_recon_per_tok": tot_recon / max(tot_tok, 1),
        "valid_sample_rate": valid / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay (regularization)")
    ap.add_argument("--patience", type=int, default=0, help="early-stop after N epochs w/o val improvement (0=off)")
    ap.add_argument("--max-mols", type=int, default=0, help="cap total molecules seen (quick runs)")
    ap.add_argument("--ckpt-every", type=int, default=config.CHECKPOINT_EVERY)
    ap.add_argument("--keep", type=int, default=10, help="periodic checkpoints to retain")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset-schedule", action="store_true",
                    help="load weights from latest.pt but restart epoch/LR/KL schedule "
                         "(clean continued-training on new data, fresh optimizer)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    args = ap.parse_args()

    set_seed(config.SEED)
    config.ensure_dirs()
    device = config.get_device()
    use_amp = config.USE_AMP and not args.no_amp and device.type == "cuda"

    vocab = data.Vocab.load()
    train_ds = data.MolDataset("train")
    val_ds = data.MolDataset("val")
    collate = data.make_collate(vocab.pad)
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                          collate_fn=collate, num_workers=args.num_workers, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        collate_fn=collate, num_workers=0, pin_memory=pin)

    net = M.build_model(vocab, device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    molecules_seen, start_epoch = 0, 0
    latest = config.CKPT_DIR / "latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location=device)
        net.load_state_dict(ck["model"])
        if args.reset_schedule:
            print(f"Loaded weights from {latest.name}; restarting schedule (fresh optimizer).")
        else:
            opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
            molecules_seen = ck["molecules_seen"]
            start_epoch = ck["epoch"]
            print(f"Resumed from {latest.name}: {molecules_seen:,} molecules, epoch {start_epoch}")

    print(f"Train molecules: {len(train_ds):,} | val: {len(val_ds):,} | params: {M.count_params(net):,}")
    print(f"Device: {device} | AMP: {use_amp} | batch: {args.batch} | "
          f"checkpoint every {args.ckpt_every:,} molecules")

    log_path = config.CKPT_DIR / "train_log.jsonl"
    next_ckpt = (molecules_seen // args.ckpt_every + 1) * args.ckpt_every
    total_planned = args.max_mols if args.max_mols else args.epochs * len(train_ds)
    best_val, no_improve = -1.0, 0
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        net.train()
        running = {"recon": 0.0, "kl": 0.0, "prop": 0.0}
        n_batches = 0
        bar = tqdm(total=len(train_ds), unit="mol", dynamic_ncols=True,
                   desc=f"epoch {epoch+1}/{args.epochs}")
        for toks, lens, cond in train_dl:
            toks, cond = toks.to(device, non_blocking=True), cond.to(device, non_blocking=True)
            beta = cyclical_beta(molecules_seen, total_planned)
            lr_now = lr_at(molecules_seen, total_planned)
            for g in opt.param_groups:
                g["lr"] = lr_now
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, mu, logvar, prop = net(toks, lens, cond)
                total, comps = M.vae_loss(logits, toks[:, 1:], mu, logvar, prop, cond,
                                          vocab.pad, beta=beta)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            bs = toks.size(0)
            molecules_seen += bs
            n_batches += 1
            for k in running:
                running[k] += comps[k]
            if n_batches % 20 == 0:
                vram = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
                bar.set_postfix(seen=f"{molecules_seen/1e6:.2f}M", recon=f"{comps['recon']:.2f}",
                                kl=f"{comps['kl']:.2f}", prop=f"{comps['prop']:.3f}",
                                beta=f"{beta:.3f}", lr=f"{lr_now:.1e}", vram=f"{vram:.1f}G")
            bar.update(bs)

            # checkpoint every N molecules
            if molecules_seen >= next_ckpt:
                ckpt_path = config.CKPT_DIR / f"checkpoint_mols_{molecules_seen}.pt"
                save_checkpoint(ckpt_path, net, opt, scaler, molecules_seen, epoch, vocab)
                save_checkpoint(latest, net, opt, scaler, molecules_seen, epoch, vocab)
                prune_checkpoints(args.keep)
                bar.write(f"  [checkpoint] {ckpt_path.name} @ {molecules_seen:,} molecules")
                next_ckpt += args.ckpt_every

            if args.max_mols and molecules_seen >= args.max_mols:
                break
        bar.close()

        val = validate(net, val_dl, vocab, device)
        avg = {k: running[k] / max(n_batches, 1) for k in running}
        rec = {"epoch": epoch + 1, "molecules_seen": molecules_seen,
               "train": avg, "val": val, "elapsed_min": (time.time() - t_start) / 60}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch+1}: recon={avg['recon']:.3f} kl={avg['kl']:.3f} "
              f"prop={avg['prop']:.3f} | val_token_acc={val['val_token_acc']:.3f} "
              f"valid_sample_rate={val['valid_sample_rate']:.3f}")

        # end-of-epoch checkpoint
        save_checkpoint(latest, net, opt, scaler, molecules_seen, epoch + 1, vocab)

        # best-validation checkpoint + optional early stopping (anti-overfit)
        score = val["val_token_acc"] + 0.25 * val["valid_sample_rate"]
        if score > best_val + 1e-4:
            best_val, no_improve = score, 0
            save_checkpoint(config.CKPT_DIR / "best.pt", net, opt, scaler, molecules_seen, epoch + 1, vocab)
            print(f"  [best] saved best.pt (val_token_acc={val['val_token_acc']:.3f})")
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                print(f"Early stop: no val improvement in {args.patience} epochs. Best -> best.pt")
                break

        if args.max_mols and molecules_seen >= args.max_mols:
            print("Reached --max-mols cap; stopping.")
            break

    print(f"Training done. {molecules_seen:,} molecules in {(time.time()-t_start)/60:.1f} min. "
          f"Latest checkpoint: {latest}")


if __name__ == "__main__":
    main()
