"""
predictive/design.py  (pipeline step 5)  - inverse design with the production model.
====================================================================================
Closes the loop for whatever chemistry TARGET describes:
  VAE generator -> production model (ensemble on the shortlisted features) -> LLM triage -> RAG
  -> top candidate molecules to test.

Two-stage scoring keeps it tractable on CPU:
  1. SCREEN   a large generated pool fast with imputed-xTB (RDKit + context signal),
              re-seeding generation around the best-scored each round.
  2. REFINE   run real GFN2-xTB on the shortlist, re-score with full features, then rank by
              objective +- lambda*uncertainty (direction from TARGET.maximize).
  3. TRIAGE   (optional --llm) synthesizability / stability / mechanism assessment + a
              frontier-model judge re-rank; drop candidates flagged with TARGET.fatal_flag.

Run:  python -m batterygen.predictive.design --n 1200 --rounds 3 --shortlist 60 --top 30 --llm
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch

from batterygen.core import config
from batterygen.core import infer
from batterygen.predictive.target import TARGET
from batterygen.predictive import train, features, design_llm, sampling

try:  # Windows consoles are cp1252 - LLM text has unicode (arrows, superscripts, etc.)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _objkey(v: float) -> float:
    """Higher = better, regardless of maximize/minimize."""
    return v if TARGET.maximize else -v


def _parse_context(spec: str) -> dict:
    out = {}
    for kv in (spec or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="dataset CSV for novelty/seeding (default: auto-detect from data/; optional)")
    ap.add_argument("--ckpt", default=None, help="VAE generator ckpt (default best.pt)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=1200, help="candidates generated per round")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed-k", type=int, default=12, help="top molecules to re-seed from")
    ap.add_argument("--seed-best", type=int, default=15,
                    help="seed round 0 from this many best training molecules")
    ap.add_argument("--shortlist", type=int, default=150, help="how many to xTB-refine")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--lam", type=float, default=0.5, help="uncertainty penalty in final rank")
    ap.add_argument("--min-target", type=float, default=None,
                    help="only keep candidates that beat this target value (direction-aware)")
    ap.add_argument("--apply-ranges", action="store_true",
                    help="hard-filter by LLM-parsed property ranges (off by default)")
    ap.add_argument("--rag", action="store_true", help="ground the rationale in the literature KB + novelty")
    ap.add_argument("--z-scale", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--spread", type=float, default=0.6)
    ap.add_argument("--context", default="", help="context overrides 'col=val,col=val' (default: training medians)")
    ap.add_argument("--prompt", default=None, help="NL design request (LLM -> property/context constraints)")
    ap.add_argument("--llm", action="store_true", help="LLM synth/stability/mechanism triage + judge re-rank")
    ap.add_argument("--out", default="candidates.csv")
    args = ap.parse_args()
    args.csv = config.resolve_ce_csv_optional(args.csv)

    bundle = train.load_production()
    ckpt = sampling._resolve_ckpt(args.ckpt)
    net, vocab, _, device = infer.load_model(ckpt, device=args.device)

    # context = training medians, overridden by CLI and then by an LLM-parsed request
    context = {c: float(bundle["feat_med"][c]) for c in bundle.get("context_cols", [])}
    context.update({k: v for k, v in _parse_context(args.context).items() if k in context})

    known = set()
    if args.csv and Path(args.csv).exists():
        known = {r["smiles"] for r in sampling.load_labeled_csv(Path(args.csv))}

    ranges = {}
    if args.prompt:
        req = design_llm.parse_request(args.prompt)
        if req:
            ranges = req.get("ranges", {}) or {}
            for k, v in (req.get("context") or {}).items():
                if k in context:
                    context[k] = float(v)
            print(f"LLM parsed request: {req.get('note', '')}")
            if ranges:
                print(f"  property ranges: {ranges}")
            if req.get("avoid"):
                print(f"  avoid: {req['avoid']} (covered by stability filter + LLM triage)")
        else:
            print("[llm] request parsing unavailable; using defaults.")

    def in_ranges(smi):
        if not ranges or not args.apply_ranges:
            return True
        f = features.rdkit_features(smi)
        return f is not None and all(
            (k not in f) or (float(v[0]) <= f[k] <= float(v[1])) for k, v in ranges.items())

    tname = bundle.get("target_name", TARGET.target_name)
    direction = "maximize" if TARGET.maximize else "minimize"
    print(f"Model: ensemble on {len(bundle['feature_order'])} selected features | "
          f"random R^2 {bundle['random_R2']}, scaffold R^2 {bundle['scaffold_R2']} | "
          f"device {device} | {direction} {tname} | context {context}")

    # ---- stage 1: generate + fast screen (imputed xTB) ----
    screened = {}
    best = lambda: max(_objkey(v) for v in screened.values())

    def gen_and_screen(seeds):
        pool = sampling._generate(net, vocab, device, args.n, args.z_scale, args.temperature,
                                  seed_latents=seeds, spread=args.spread)
        pool = [s for s in dict.fromkeys(pool)
                if s and s not in known and s not in screened and sampling.chem_ok(s) and in_ranges(s)]
        if not pool:
            return 0
        preds = train.predict(bundle, pool, context, compute_xtb=False)
        for s, o in preds.items():
            screened[s] = o["pred"]
        return len(preds)

    n0 = gen_and_screen(None)
    if not screened:
        raise SystemExit("No valid candidates generated - try more --n or a different --ckpt.")
    disp = (lambda: max(screened.values()) if TARGET.maximize else min(screened.values()))
    print(f"\nround 0 (global)      : screened {n0:5d} | best {tname} {disp():.3f}")

    if args.seed_best and args.csv and Path(args.csv).exists():
        best_rows = sorted(sampling.load_labeled_csv(Path(args.csv)),
                           key=lambda r: r["target"], reverse=TARGET.maximize)
        top_smis = list(dict.fromkeys(r["smiles"] for r in best_rows))[:args.seed_best]
        latb = sampling._feat_latent(top_smis, net, vocab, device)
        seedb = torch.tensor(np.array([latb[s] for s in top_smis if s in latb]), device=device)
        nb = gen_and_screen(seedb)
        print(f"round 0b (seed best {len(seedb):2d}): screened {nb:5d} | best {tname} {disp():.3f}")

    for rnd in range(1, args.rounds + 1):
        top = sorted(screened.items(), key=lambda kv: _objkey(kv[1]), reverse=True)[:args.seed_k]
        lat = sampling._feat_latent([s for s, _ in top], net, vocab, device)
        seeds = torch.tensor(np.array([lat[s] for s, _ in top if s in lat]), device=device)
        nn = gen_and_screen(seeds)
        print(f"round {rnd} (seed top {len(seeds):2d}) : screened {nn:5d} | "
              f"best {tname} {disp():.3f} | pool {len(screened)}")

    # ---- stage 2: xTB-refine the shortlist ----
    shortlist = [s for s, _ in sorted(screened.items(), key=lambda kv: _objkey(kv[1]),
                                      reverse=True)[:args.shortlist]]
    print(f"\nRefining top {len(shortlist)} with real GFN2-xTB (cached) ...")
    refined = train.predict(bundle, shortlist, context, compute_xtb=True, xtb_cache={})

    if args.min_target is not None:
        refined = {s: o for s, o in refined.items()
                   if (o["pred"] >= args.min_target if TARGET.maximize else o["pred"] <= args.min_target)}
        print(f"  --min-target {args.min_target}: {len(refined)} candidates remain")
        if not refined:
            print("  (none met --min-target; relax it or run more rounds / larger --n)")
            return

    ml_ranked = sorted(refined.items(),
                       key=lambda kv: _objkey(kv[1]["pred"]) - args.lam * kv[1]["unc"], reverse=True)

    # ---- stage 3 (optional): LLM triage + judge re-rank ----
    assess = {}
    if args.llm:
        cand = [{"idx": i, "smiles": s, "pred": round(o["pred"], 2)} for i, (s, o) in enumerate(ml_ranked)]
        print(f"\nLLM triage of {len(cand)} candidates (synth/stability/mechanism) ...")
        by_idx = design_llm.assess_batch(cand, role="reasoner")
        assess = {cand[i]["smiles"]: v for i, v in by_idx.items() if i < len(cand)}
        if TARGET.fatal_flag:
            survivors = [(s, o) for s, o in ml_ranked
                         if TARGET.fatal_flag not in assess.get(s, {}).get("stability", "")]
        else:
            survivors = list(ml_ranked)
        n_dropped = len(ml_ranked) - len(survivors)
        survivors = survivors or ml_ranked
        jin = [{"idx": i, "smiles": s, "pred": round(o["pred"], 2),
                "uncertainty": round(o["unc"], 2), **assess.get(s, {})}
               for i, (s, o) in enumerate(survivors)]
        order = design_llm.judge_rerank(jin, top=args.top, role="judge")
        if order:
            chosen = [survivors[i] for i in order if i < len(survivors)]
            rest = [so for k, so in enumerate(survivors) if k not in set(order)]
            survivors = chosen + rest
        print(f"  dropped {n_dropped} flagged; judge re-rank: {'applied' if order else 'unavailable'}.")
        ranked = survivors[:args.top]
    else:
        ranked = ml_ranked[:args.top]

    # ---- output ----
    from batterygen.core.membership import MolportIndex
    index = MolportIndex()
    lo, hi = bundle["y_range"]
    pcol = "pred_" + (re.sub(r"\W+", "_", tname).strip("_") or "value")
    print(f"\n=== TOP {len(ranked)} candidates to test (xTB-refined; trained {tname} {lo:.2f}-{hi:.2f}) ===")
    hdr = f"{'rank':>4} {tname[:8]:>8} {'+-unc':>5} {'sim':>5} {'HOMO':>7} {'Molport':>9}"
    print(hdr + ("  synth stab     mechanism / SMILES" if args.llm else "  SMILES"))

    rag_ctx = {}
    if args.rag:
        from batterygen.predictive import rag
        print("\nRAG: scoring literature novelty against the KB ...")
        for smi, o in ranked:
            ctx, top = rag.context_for(smi, assess.get(smi, {}).get("mechanism", ""))
            rag_ctx[smi] = (ctx, round(1.0 - top, 3))

    rows = []
    for i, (smi, o) in enumerate(ranked, 1):
        mid = index.get_id(smi, already_canonical=True) if index.available else None
        val = max(lo, min(hi, o["pred"]))
        a = assess.get(smi, {})
        nov = rag_ctx.get(smi, ("", ""))[1]
        base = (f"{i:>4} {val:>8.2f} {o['unc']:>5.2f} {o['domain_sim']:>5.2f} "
                f"{o['xtb_homo']:>7.2f} {mid or '-':>9}")
        if args.llm:
            print(base + f"  {a.get('synth_score','?'):>4} {a.get('stability','')[:8]:>8}  "
                  f"{a.get('mechanism','')[:40]} | {smi}")
        else:
            print(base + f"  {smi}")
        rows.append({"rank": i, pcol: round(val, 3), "uncertainty": round(o["unc"], 3),
                     "domain_sim": o["domain_sim"], "xtb_homo": round(o["xtb_homo"], 4),
                     "synth_score": a.get("synth_score", ""), "stability": a.get("stability", ""),
                     "mechanism": a.get("mechanism", ""), "redflag": a.get("redflag", ""),
                     "lit_novelty": nov, "molport_id": mid or "", "smiles": smi})
    if index.available:
        index.close()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank", pcol, "uncertainty", "domain_sim", "xtb_homo",
                                          "synth_score", "stability", "mechanism", "redflag",
                                          "lit_novelty", "molport_id", "smiles"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {len(rows)} candidates -> {args.out}")
    if args.llm and ranked:
        ctx = rag_ctx.get(ranked[0][0], ("", ""))[0] if args.rag else ""
        print("\nMechanistic rationale for the #1 candidate%s:" % (" (RAG-grounded)" if ctx else ""))
        print("  " + design_llm.rationale(ranked[0][0], ranked[0][1]["pred"], context=ctx))
    print(f"\nNote: random-split R^2={bundle['random_R2']}, scaffold R^2={bundle['scaffold_R2']} -> "
          "ranking/triage for NOVEL molecules is approximate. domain_sim~1 = close to a known "
          "molecule (more reliable). Validate top hits, add measurements to the CSV, re-run "
          "features/select/train (active learning).")


if __name__ == "__main__":
    main()
