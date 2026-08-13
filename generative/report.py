"""Build an HTML report of the whole training + evaluation run.

Reads the training log (train_log.jsonl) and the eval report (eval_report.json,
from evaluate.py) and renders: training curves, the property-head R^2, the
generated-vs-data property distributions, the 6-metric benchmark table vs the
ElectrolyteGPT paper, and (optionally) a latent-space PCA. Purely from saved
artifacts, so it never needs the GPU.

  python -m batterygen.generative.evaluate            # first, to produce eval_report.json
  python -m batterygen.generative.report              # -> batterygen_artifacts/report.html
  python -m batterygen.generative.report --latent     # also embed a latent-space PCA (uses the model)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from batterygen.core import config


PAPER = {
    "Mol-CycleGAN": [0.923, 0.996, 0.951, 0.869, 0.538, 0.462],
    "JT-VAE": [0.999, 0.999, 0.916, 0.813, 0.604, 0.396],
    "MinGPT": [0.970, 0.991, 0.712, 0.581, 0.407, 0.593],
    "1Ddiffusion": [0.219, 0.995, 0.971, 0.904, 0.317, 0.683],
    "Diffusion-LM": [0.278, 0.998, 0.990, 0.925, 0.324, 0.676],
    "ElectrolyteGPT": [0.995, 0.999, 0.997, 0.942, 0.421, 0.579],
}
COLS = ["Validity", "Uniqueness", "Novelty(train)", "Novelty(PubChem)", "Similarity", "Diversity"]


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(title, b64):
    return f"<h3>{title}</h3><img src='data:image/png;base64,{b64}'/>"


def training_curves():
    p = config.CKPT_DIR / "train_log.jsonl"
    if not p.exists():
        return "<p><i>No train_log.jsonl yet — train first.</i></p>"
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not recs:
        return "<p><i>train log empty.</i></p>"
    ep = [r["epoch"] for r in recs]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [r["train"]["recon"] for r in recs], "-o", label="recon")
    ax[0].plot(ep, [r["train"]["kl"] for r in recs], "-o", label="KL")
    ax[0].plot(ep, [r["train"]["prop"] for r in recs], "-o", label="prop")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Training losses"); ax[0].legend(); ax[0].grid(alpha=.3)
    if "val" in recs[0]:
        ax[1].plot(ep, [r["val"].get("val_token_acc", 0) for r in recs], "-o", label="val token acc")
        ax[1].plot(ep, [r["val"].get("valid_sample_rate", 0) for r in recs], "-o", label="valid sample rate")
        ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("epoch"); ax[1].set_title("Validation"); ax[1].legend(); ax[1].grid(alpha=.3)
    return _img("Training curves", _b64(fig))


def _load_eval():
    p = config.CKPT_DIR / "eval_report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def property_r2(ev):
    prop = ev.get("property", {})
    if not prop:
        return ""
    names = list(prop)
    fig, axx = plt.subplots(figsize=(10, 3.5))
    axx.bar(names, [prop[n]["r2"] for n in names], color="#4cc2ff")
    axx.set_ylabel("R²"); axx.set_title("Property head R² on held-out molecules"); axx.set_ylim(0, 1)
    axx.tick_params(axis="x", rotation=45); axx.grid(axis="y", alpha=.3)
    return _img("Property prediction (generalization)", _b64(fig))


def property_dist(ev):
    dist = ev.get("generation", {}).get("property_dist", {})
    if not dist:
        return ""
    names = list(dist)
    import numpy as np
    x = np.arange(len(names))
    fig, axx = plt.subplots(figsize=(10, 3.5))
    axx.bar(x - 0.2, [dist[n]["gen_mean"] for n in names], 0.4, label="generated", color="#3fb950")
    axx.bar(x + 0.2, [dist[n]["data_mean"] for n in names], 0.4, label="dataset", color="#888")
    axx.set_xticks(x); axx.set_xticklabels(names, rotation=45, ha="right")
    axx.set_title("Generated vs dataset property means"); axx.legend(); axx.grid(axis="y", alpha=.3)
    return _img("Distribution match", _b64(fig))


def benchmark_table(ev):
    g = ev.get("generation", {})
    ours = [g.get("validity"), g.get("uniqueness"), g.get("novelty_vs_training"),
            g.get("novelty_vs_pubchem"), g.get("similarity"), g.get("diversity")]
    rows = "".join(
        "<tr><td>" + name + "</td>" + "".join(f"<td>{v:.3f}</td>" for v in vals) + "</tr>"
        for name, vals in PAPER.items())
    ours_cells = "".join(f"<td>{'—' if v is None else f'{v:.3f}'}</td>" for v in ours)
    head = "".join(f"<th>{c}</th>" for c in COLS)
    return (f"<h3>Benchmark vs ElectrolyteGPT (JACS Au 2026, Table 1)</h3>"
            f"<table class='bm'><tr><th>Model</th>{head}</tr>{rows}"
            f"<tr class='ours'><td>OURS (batterygen)</td>{ours_cells}</tr></table>"
            "<p class='note'>Paper numbers are on a 1M <i>electrolyte</i> dataset; compare fairly "
            "after electrolyte specialization (add_data.py). Similarity=mean pairwise Tanimoto, "
            "Diversity=1−Similarity.</p>")


def latent_pca():
    import numpy as np
    import torch
    from batterygen.core import data as _data

    from batterygen.core import infer

    net, vocab, _, device = infer.load_model(None, device="cpu")
    ds = _data.MolDataset("val")
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=256, collate_fn=_data.make_collate(vocab.pad))
    mus, mw = [], []
    mean, std = _data.load_stats()
    with torch.no_grad():
        for toks, lens, cond in dl:
            mu, _ = net.encode(toks, lens)
            mus.append(mu.numpy())
            mw.extend((cond[:, 0].numpy() * std[0] + mean[0]).tolist())
            if sum(len(m) for m in mus) >= 1500:
                break
    X = np.concatenate(mus)[:1500]
    from sklearn.decomposition import PCA
    xy = PCA(2).fit_transform(X)
    fig, axx = plt.subplots(figsize=(6, 5))
    sc = axx.scatter(xy[:, 0], xy[:, 1], c=mw[:1500], s=6, cmap="viridis")
    fig.colorbar(sc, label="MolWt"); axx.set_title("Latent space (PCA), colored by MolWt")
    return _img("Latent space", _b64(fig))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", action="store_true", help="also embed a latent PCA (uses the model on CPU)")
    ap.add_argument("--out", default=str(config.ART_DIR / "report.html"))
    args = ap.parse_args()

    ev = _load_eval()
    parts = ["<h1>batterygen — training &amp; evaluation report</h1>", training_curves()]
    if ev:
        parts += [benchmark_table(ev), property_r2(ev), property_dist(ev),
                  f"<p>Reconstruction on {ev.get('split','?')} split: "
                  f"exact {ev['reconstruction']['exact_recon_rate']:.1%}, "
                  f"token acc {ev['reconstruction']['token_acc']:.1%}.</p>"]
    else:
        parts.append("<p><i>No eval_report.json — run <code>python -m batterygen.generative.evaluate</code> first.</i></p>")
    if args.latent:
        try:
            parts.append(latent_pca())
        except Exception as e:
            parts.append(f"<p><i>latent PCA skipped: {e}</i></p>")

    html = ("<!doctype html><meta charset='utf-8'><title>batterygen report</title>"
            "<style>body{font:14px system-ui;margin:32px;max-width:1000px}"
            "img{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:6px 0}"
            "table.bm{border-collapse:collapse;margin:8px 0}.bm td,.bm th{border:1px solid #ccc;padding:4px 9px;text-align:center}"
            ".bm tr.ours{font-weight:700;background:#eef9ee}.note{color:#666;font-size:12px}"
            "h3{margin-top:26px}</style>" + "".join(parts))
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"Wrote report -> {args.out}")


if __name__ == "__main__":
    main()
