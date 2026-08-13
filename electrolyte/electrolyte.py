"""Electrolyte specialization: formulation-aware, multi-target property model with
cation conditioning + candidate screening.

A formulation feature = [weighted solvent latent(s) ; anion latent ; cation one-hot
(all battery chemistries) ; source one-hot ; conc ; temp ; #solvents]. Trains a
multi-output MLP to predict conductivity / coordination / viscosity / density from
CALiSol-23 + OEDB (electrolyte_data.py), then screens VAE-generated molecules for a
chosen cation system.

The "source" one-hot lets us combine experimental (CALiSol) and MD-simulated (OEDB)
data without conflating their different conductivity scales.

  python -m batterygen.electrolyte.electrolyte_data                      # build electrolyte_train.csv
  python -m batterygen.electrolyte.electrolyte --mode train --csv ...electrolyte_train.csv \
      --mix-col mix --cation-col cation --anion-smiles-col anion_smiles \
      --conc-col conc --temp-col temp --source-col source \
      --target-cols conductivity,coord_cat_anion,coord_cat_solvent --log-target
  python -m batterygen.electrolyte.electrolyte --mode screen --cation Mg --conc 1.0 --temp 298 --n 200
  python -m batterygen.electrolyte.electrolyte --mode demo
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from batterygen.core import config


CATIONS = ["Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba",
           "Zn", "Al", "Fe", "H", "NH4", "other"]
SOURCES = ["calisol23", "oedb", "experimental", "user", "other"]
CAT_IDX = {c: i for i, c in enumerate(CATIONS)}
SRC_IDX = {s: i for i, s in enumerate(SOURCES)}
ELEC_CKPT = config.CKPT_DIR / "electrolyte_model.pt"


def _onehot(name, table, n):
    v = np.zeros(n, np.float32)
    key = str(name).strip()
    v[table.get(key, table.get(key.capitalize().replace("Nh4", "NH4"), n - 1))] = 1.0
    return v


# --------------------------------------------------------------------------- VAE
def get_vae(ckpt, allow_untrained=False):
    from batterygen.core import infer

    from batterygen.core import model as M

    from batterygen.core import data as _data


    device = config.get_device()
    path = Path(ckpt) if ckpt else (config.CKPT_DIR / "latest.pt")
    if path.exists():
        return infer.load_model(ckpt)
    if not allow_untrained:
        raise SystemExit(f"No VAE checkpoint at {path}. Train the VAE first.")
    print("[warn] no VAE checkpoint — using UNTRAINED encoder (plumbing only).")
    vocab = _data.Vocab.load()
    return M.build_model(vocab, device).eval(), vocab, {}, device


@torch.no_grad()
def latents_for(net, vocab, smiles, device, batch=256):
    import selfies as sf
    from batterygen.core import data as _data


    uniq = [s for s in dict.fromkeys(smiles) if s]
    rows, keys = [], []
    for s in uniq:
        canon = _data.canonical_smiles(s)
        if canon is None:
            continue
        try:
            sel = sf.encoder(canon)
        except Exception:
            continue
        if not sel or sf.len_selfies(sel) > config.MAX_SELFIES_LEN - 2:
            continue
        ids, length = vocab.encode(sel)
        rows.append((ids, length)); keys.append(s)
    out = {}
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        toks = torch.tensor([r[0] for r in chunk], dtype=torch.long, device=device)
        lens = torch.tensor([r[1] for r in chunk], dtype=torch.long)
        mu, _ = net.encode(toks[:, :int(lens.max())], lens)
        for k, m in zip(keys[i:i + batch], mu.cpu().numpy()):
            out[k] = m
    return out


# --------------------------------------------------------------- featurization
class Featurizer:
    def __init__(self, latent_dim, use_anion):
        self.latent_dim = latent_dim
        self.use_anion = use_anion
        self.dim = latent_dim + (latent_dim if use_anion else 0) + len(CATIONS) + len(SOURCES) + 3

    def vector(self, solvent_lats, fracs, anion_lat, cation, source, conc, temp):
        if not solvent_lats:
            return None
        w = np.asarray(fracs, np.float32)
        w = w / (w.sum() + 1e-8)
        solv = np.average(np.stack(solvent_lats), axis=0, weights=w).astype(np.float32)
        parts = [solv]
        if self.use_anion:
            parts.append(anion_lat if anion_lat is not None else np.zeros(self.latent_dim, np.float32))
        parts.append(_onehot(cation, CAT_IDX, len(CATIONS)))
        parts.append(_onehot(source, SRC_IDX, len(SOURCES)))
        try:
            c = float(conc)
        except (TypeError, ValueError):
            c = 1.0
        try:
            t = float(temp) / 298.0
        except (TypeError, ValueError):
            t = 1.0
        parts.append(np.asarray([c, t, float(len(solvent_lats))], np.float32))
        return np.concatenate(parts)


def _parse_mix(cell):
    """'smi:frac;smi:frac' -> ([smi,...], [frac,...])."""
    smis, fr = [], []
    for tok in str(cell).split(";"):
        tok = tok.strip()
        if not tok:
            continue
        smi, _, f = tok.rpartition(":")
        if not smi:
            smi, f = f, "1"
        smis.append(smi)
        try:
            fr.append(float(f))
        except ValueError:
            fr.append(1.0)
    return smis, fr


def build_dataset(rows, mp, net, vocab, device):
    # collect every solvent + anion SMILES for one batched encode
    all_smiles = []
    for r in rows:
        if mp.get("mix_col"):
            all_smiles += _parse_mix(r.get(mp["mix_col"], ""))[0]
        else:
            all_smiles += [r[c] for c in mp.get("smiles_cols", []) if r.get(c)]
        if mp.get("anion_smiles_col") and r.get(mp["anion_smiles_col"]):
            all_smiles.append(r[mp["anion_smiles_col"]])
    lat = latents_for(net, vocab, all_smiles, device)
    feat = Featurizer(config.LATENT_DIM, use_anion=bool(mp.get("anion_smiles_col")))
    targets = mp["target_cols"]

    X, Y = [], []
    for r in rows:
        if mp.get("mix_col"):
            smis, fr = _parse_mix(r.get(mp["mix_col"], ""))
        else:
            smis = [r[c] for c in mp.get("smiles_cols", []) if r.get(c)]
            fr = [float(r.get(mp["frac_cols"][j], 1.0)) if mp.get("frac_cols") else 1.0
                  for j, c in enumerate(mp.get("smiles_cols", [])) if r.get(c)]
        solv = [lat[s] for s in smis if s in lat]
        frv = [f for s, f in zip(smis, fr) if s in lat]
        if not solv:
            continue
        anion = lat.get(r.get(mp.get("anion_smiles_col", ""), ""), None)
        vec = feat.vector(solv, frv, anion, r.get(mp.get("cation_col", ""), "Li"),
                          r.get(mp.get("source_col", ""), "user"),
                          r.get(mp.get("conc_col", ""), 1.0), r.get(mp.get("temp_col", ""), 298.0))
        if vec is None:
            continue
        yr = []
        for t in targets:
            try:
                yr.append(float(r[t]))
            except (KeyError, ValueError, TypeError):
                yr.append(np.nan)
        if np.all(np.isnan(yr)):
            continue
        X.append(vec); Y.append(yr)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), feat


# ------------------------------------------------------------------- the model
class PropNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def train_model(X, Y, feat, targets, device, epochs=400, log_target=False):
    if log_target:
        Y = np.where(Y > 0, np.log10(np.clip(Y, 1e-6, None)), np.nan)
    ym = np.nanmean(Y, 0); ys = np.nanstd(Y, 0) + 1e-6
    Yn = (Y - ym) / ys
    mask = ~np.isnan(Y)
    Yn = np.nan_to_num(Yn)

    Xt = torch.tensor(X, device=device)
    xm, xs = Xt.mean(0), Xt.std(0) + 1e-6
    Xn = (Xt - xm) / xs
    Yt = torch.tensor(Yn, device=device)
    Mt = torch.tensor(mask, dtype=torch.float32, device=device)

    n_val = max(1, len(X) // 5)
    perm = torch.randperm(len(X), device=device)
    va, tr = perm[:n_val], perm[n_val:]
    net = PropNet(feat.dim, len(targets)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best = (1e9, None)
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        pred = net(Xn[tr])
        loss = ((pred - Yt[tr]) ** 2 * Mt[tr]).sum() / Mt[tr].sum().clamp(min=1)
        loss.backward(); opt.step()
        if (ep + 1) % max(1, epochs // 8) == 0:
            net.eval()
            with torch.no_grad():
                pv = net(Xn[va])
                vmse = (((pv - Yt[va]) ** 2 * Mt[va]).sum() / Mt[va].sum().clamp(min=1)).item()
                r2s = []
                for j in range(len(targets)):
                    m = Mt[va][:, j].bool()
                    if m.sum() > 2:
                        yj, pj = Yt[va][m, j], pv[m, j]
                        ss = ((yj - pj) ** 2).sum().item()
                        tot = ((yj - yj.mean()) ** 2).sum().item() or 1.0
                        r2s.append(1 - ss / tot)
                    else:
                        r2s.append(float("nan"))
            print(f"  epoch {ep+1}/{epochs} val_mse={vmse:.3f} R2=" +
                  ", ".join(f"{t}:{r:.2f}" for t, r in zip(targets, r2s)))
            if vmse < best[0]:
                best = (vmse, {k: v.clone() for k, v in net.state_dict().items()})
    if best[1]:
        net.load_state_dict(best[1])
    stats = {"xm": xm.cpu().tolist(), "xs": xs.cpu().tolist(), "ym": ym.tolist(),
             "ys": ys.tolist(), "log_target": log_target, "dim": feat.dim,
             "use_anion": feat.use_anion, "targets": targets}
    return net, stats


def save_model(net, stats, mapping):
    torch.save({"model": net.state_dict(), "stats": stats, "mapping": mapping,
                "cations": CATIONS, "sources": SOURCES}, ELEC_CKPT)
    print(f"Saved electrolyte property model ({len(stats['targets'])} targets) -> {ELEC_CKPT}")


# ---------------------------------------------------------------------- screen
@torch.no_grad()
def screen(args):
    if not ELEC_CKPT.exists():
        raise SystemExit("Train a model first (--mode train or --mode demo).")
    ck = torch.load(ELEC_CKPT, map_location="cpu")
    st = ck["stats"]; targets = st["targets"]
    net_vae, vocab, _, device = get_vae(args.ckpt, allow_untrained=True)
    pnet = PropNet(st["dim"], len(targets)).to(device)
    pnet.load_state_dict(ck["model"]); pnet.eval()
    feat = Featurizer(config.LATENT_DIM, st["use_anion"])
    xm = torch.tensor(st["xm"], device=device); xs = torch.tensor(st["xs"], device=device)
    ym = np.asarray(st["ym"]); ys = np.asarray(st["ys"])

    from batterygen.generative import generate as G

    spec = json.loads(args.spec) if args.spec else {"MolWt": 110, "NumAromaticRings": 0}
    rows = G.generate(net_vae, vocab, spec, args.n, temperature=args.temperature,
                      molport_only=False, device=device)
    cand = [r["smiles"] for r in rows]
    lat = latents_for(net_vae, vocab, cand, device)
    anion_lat = None
    if st["use_anion"] and args.anion:
        anion_lat = next(iter(latents_for(net_vae, vocab, [args.anion], device).values()), None)

    from batterygen.core.membership import MolportIndex
    index = MolportIndex()
    scored = []
    for smi in cand:
        if smi not in lat:
            continue
        vec = feat.vector([lat[smi]], [1.0], anion_lat, args.cation, args.source, args.conc, args.temp)
        x = (torch.tensor(vec, device=device) - xm) / xs
        pred = pnet(x.unsqueeze(0)).cpu().numpy()[0] * ys + ym
        if st["log_target"]:
            pred = 10 ** pred
        scored.append((smi, pred, index.get_id(smi, already_canonical=True) if index.available else None))
    primary = 0
    scored.sort(key=lambda t: t[1][primary], reverse=True)

    print(f"\nTop solvents for {args.cation}+ @ {args.conc} mol/kg, {args.temp} K "
          f"(source={args.source}); predicting {targets}:")
    print(f"{'#':>2}  {'SMILES':<38} " + " ".join(f"{t[:12]:>13}" for t in targets) + "  catalog")
    for i, (smi, pred, mid) in enumerate(scored[:args.top], 1):
        vals = " ".join(f"{v:>13.4g}" for v in pred)
        print(f"{i:>2}  {smi:<38} {vals}  {mid or 'novel'}")


# ------------------------------------------------------------------------ demo
def demo(args):
    from batterygen.core import data as _data

    from batterygen.core import infer

    print("DEMO: synthetic electrolyte data (validates the pipeline end-to-end).\n")
    net, vocab, _, device = get_vae(args.ckpt, allow_untrained=True)
    rng = np.random.RandomState(0)
    rows = []
    for canon, _m, _r in infer.iter_dataset_descriptors():
        d = _data.descriptors_for_smiles(canon)
        if d is None:
            continue
        tpsa = d[config.PROPERTIES.index("TPSA")]
        cation = CATIONS[rng.randint(0, 6)]; conc = round(float(rng.uniform(.2, 2)), 2)
        temp = float(rng.choice([253, 273, 298, 313]))
        cond = max(0.01, 0.02 * tpsa + (temp - 253) / 80 + conc * (1.2 - .3 * conc)
                   + (.6 if cation in ("Li", "Na") else .2) + rng.normal(0, .15))
        rows.append({"mix": f"{canon}:1", "cation": cation, "conc": conc, "temp": temp,
                     "cond": cond, "source": "user"})
        if len(rows) >= 1500:
            break
    mp = {"mix_col": "mix", "cation_col": "cation", "conc_col": "conc", "temp_col": "temp",
          "source_col": "source", "target_cols": ["cond"], "anion_smiles_col": None}
    X, Y, feat = build_dataset(rows, mp, net, vocab, device)
    print(f"Featurized {len(X)} formulations (dim={feat.dim}). Training ...")
    pnet, stats = train_model(X, Y, feat, ["cond"], device, epochs=args.epochs)
    save_model(pnet, stats, mp)
    print("\nDemo OK.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "screen", "describe", "demo"], default="demo")
    ap.add_argument("--csv")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--mix-col", default=None)
    ap.add_argument("--smiles-cols", nargs="*", default=[])
    ap.add_argument("--frac-cols", nargs="*", default=None)
    ap.add_argument("--anion-smiles-col", default=None)
    ap.add_argument("--cation-col", default=None)
    ap.add_argument("--source-col", default=None)
    ap.add_argument("--conc-col", default=None)
    ap.add_argument("--temp-col", default=None)
    ap.add_argument("--target-cols", default="conductivity", help="comma list")
    ap.add_argument("--log-target", action="store_true")
    ap.add_argument("--epochs", type=int, default=400)
    # screen
    ap.add_argument("--cation", default="Li", help="Li/Na/K have data; others need added data")
    ap.add_argument("--anion", default=None)
    ap.add_argument("--source", default="oedb", help="a source the model was trained on (calisol23/oedb)")
    ap.add_argument("--conc", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=298.0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--spec", default=None)
    args = ap.parse_args()
    config.ensure_dirs()

    if args.mode == "describe":
        rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
        print(f"{len(rows)} rows. Columns:")
        for c in (rows[0].keys() if rows else []):
            print(f"  {c}  (e.g. {rows[0][c]!r})")
    elif args.mode == "demo":
        demo(args)
    elif args.mode == "train":
        if not args.csv:
            raise SystemExit("train needs --csv")
        net, vocab, _, device = get_vae(args.ckpt)
        mp = {"mix_col": args.mix_col, "smiles_cols": args.smiles_cols, "frac_cols": args.frac_cols,
              "anion_smiles_col": args.anion_smiles_col, "cation_col": args.cation_col,
              "source_col": args.source_col, "conc_col": args.conc_col, "temp_col": args.temp_col,
              "target_cols": [t.strip() for t in args.target_cols.split(",") if t.strip()]}
        rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
        X, Y, feat = build_dataset(rows, mp, net, vocab, device)
        print(f"Featurized {len(X)} formulations (dim={feat.dim}), targets={mp['target_cols']}.")
        if len(X) < 20:
            raise SystemExit("Too few usable rows — check --mix-col / column mapping.")
        pnet, stats = train_model(X, Y, feat, mp["target_cols"], device,
                                  epochs=args.epochs, log_target=args.log_target)
        save_model(pnet, stats, mp)
    elif args.mode == "screen":
        screen(args)


if __name__ == "__main__":
    main()
