"""
ce_design.py  — inverse design with the production CE model.
===========================================================
Closes the loop: VAE generator -> production CE model (XGBoost+ExtraTrees on
RDKit+xTB features) -> top candidate additives to test.

Two-stage scoring keeps it tractable on CPU:
  1. SCREEN   a large generated pool fast with imputed-xTB (RDKit+LMR signal),
              re-seeding generation around the best-scored each round.
  2. REFINE   run real GFN2-xTB on the shortlist and re-score with full features,
              then rank by  CE - lambda*uncertainty  (uncertainty = ensemble std).

Run:  python molvae/ce_design.py --n 1200 --rounds 3 --shortlist 60 --top 30 --out ce_candidates.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

import config
import data
import infer
import ce_train
import ce_features
import ce_model  # reuse _generate, _feat_latent, chem_ok
import ce_llm

try:  # Windows consoles are cp1252 — LLM text has unicode (Zn²⁺, arrows, etc.)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CSV = "Supplementary_Data_1.csv"


def _known(csv_path):
    s = set()
    p = Path(csv_path)
    if p.exists():
        s = {r["smiles"] for r in ce_model.load_csv(p)}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--ckpt", default=None, help="VAE generator ckpt (default best.pt)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=1200, help="candidates generated per round")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed-k", type=int, default=12, help="top molecules to re-seed from")
    ap.add_argument("--seed-best", type=int, default=15, help="seed round 0 from this many top-CE training additives (bias toward high CE)")
    ap.add_argument("--shortlist", type=int, default=150, help="how many to xTB-refine")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--lam", type=float, default=0.5, help="uncertainty penalty in final rank")
    ap.add_argument("--min-ce", type=float, default=None, help="only keep candidates with predicted CE >= this")
    ap.add_argument("--apply-ranges", action="store_true", help="hard-filter by LLM-parsed property ranges (off by default)")
    ap.add_argument("--rag", action="store_true", help="ground the rationale in the literature KB (ce_rag) + novelty")
    ap.add_argument("--z-scale", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--spread", type=float, default=0.6)
    ap.add_argument("--lmr", type=float, default=None, help="LogMolarRatio context (default training median)")
    ap.add_argument("--prompt", default=None, help="NL design request (LLM -> property/LMR constraints)")
    ap.add_argument("--llm", action="store_true", help="LLM synthesizability/stability/mechanism triage + judge re-rank")
    ap.add_argument("--out", default="ce_candidates.csv")
    args = ap.parse_args()

    bundle = ce_train.load_production()
    ckpt = ce_model._resolve_ckpt(args.ckpt)
    net, vocab, _, device = infer.load_model(ckpt, device=args.device)
    lmr = args.lmr if args.lmr is not None else float(bundle["feat_med"]["LMR"])
    known = _known(args.csv)

    # NL request -> constraints (LLM). ranges filter the generated pool by RDKit descriptors.
    ranges = {}
    if args.prompt:
        req = ce_llm.parse_request(args.prompt)
        if req:
            ranges = req.get("ranges", {}) or {}
            if req.get("lmr") is not None:
                lmr = float(req["lmr"])
            print(f"LLM parsed request: {req.get('note', '')}")
            if ranges:
                print(f"  property ranges: {ranges}")
            if req.get("avoid"):
                print(f"  avoid: {req['avoid']} (covered by stability filter + LLM triage)")
        else:
            print("[llm] request parsing unavailable; using defaults.")

    def in_ranges(smi):
        if not ranges or not args.apply_ranges:   # ranges are advisory unless --apply-ranges
            return True
        f = ce_features.rdkit_features(smi)
        return f is not None and all(
            (k not in f) or (float(v[0]) <= f[k] <= float(v[1])) for k, v in ranges.items())

    print(f"CE model: ExtraTrees+XGBoost on RDKit+xTB | random R^2 {bundle['random_R2']}, "
          f"scaffold R^2 {bundle['scaffold_R2']} | device {device} | LMR={lmr:.3f}")

    # ---- stage 1: generate + fast screen (imputed xTB) ----
    screened = {}  # smi -> fast CE estimate
    seed_latents = None

    def gen_and_screen(seeds):
        pool = ce_model._generate(net, vocab, device, args.n, args.z_scale, args.temperature,
                                  seed_latents=seeds, spread=args.spread)
        pool = [s for s in dict.fromkeys(pool)
                if s and s not in known and s not in screened and ce_model.chem_ok(s)
                and in_ranges(s)]
        if not pool:
            return 0
        preds = ce_train.predict(bundle, pool, lmr, compute_xtb=False)
        for s, o in preds.items():
            screened[s] = o["ce"]
        return len(preds)

    n0 = gen_and_screen(None)
    print(f"\nround 0 (global)      : screened {n0:5d} | best fast-CE {max(screened.values()):.2f}%")
    # round 0b: seed from the highest-CE TRAINING additives (bias the search toward 98+)
    if args.seed_best and Path(args.csv).exists():
        best_rows = sorted(ce_model.load_csv(Path(args.csv)), key=lambda r: r["ce"], reverse=True)
        top_smis = list(dict.fromkeys(r["smiles"] for r in best_rows))[:args.seed_best]
        latb = ce_model._feat_latent(top_smis, net, vocab, device)
        seedb = torch.tensor(np.array([latb[s] for s in top_smis if s in latb]), device=device)
        nb = gen_and_screen(seedb)
        print(f"round 0b (seed best {len(seedb):2d}): screened {nb:5d} | best fast-CE "
              f"{max(screened.values()):.2f}%  (seeded from CE {best_rows[len(seedb)-1]['ce']:.1f}-{best_rows[0]['ce']:.1f}%)")
    for rnd in range(1, args.rounds + 1):
        top = sorted(screened.items(), key=lambda kv: kv[1], reverse=True)[:args.seed_k]
        lat = ce_model._feat_latent([s for s, _ in top], net, vocab, device)
        seeds = torch.tensor(np.array([lat[s] for s, _ in top if s in lat]), device=device)
        nn = gen_and_screen(seeds)
        print(f"round {rnd} (seed top {len(seeds):2d}) : screened {nn:5d} | "
              f"best fast-CE {max(screened.values()):.2f}% | pool {len(screened)}")

    # ---- stage 2: xTB-refine the shortlist ----
    shortlist = [s for s, _ in sorted(screened.items(), key=lambda kv: kv[1], reverse=True)[:args.shortlist]]
    print(f"\nRefining top {len(shortlist)} with real GFN2-xTB (cached) ...")
    refined = ce_train.predict(bundle, shortlist, lmr, compute_xtb=True, xtb_cache={})

    n_ge98 = sum(1 for o in refined.values() if o["ce"] >= 98.0)
    print(f"  {n_ge98} of {len(refined)} refined candidates predict CE >= 98%")
    if args.min_ce:
        refined = {s: o for s, o in refined.items() if o["ce"] >= args.min_ce}
        print(f"  --min-ce {args.min_ce}: {len(refined)} candidates remain")
        if not refined:
            print("  (none met --min-ce; lower it or run more rounds / larger --n)")
            return

    # ML ranking: CE penalized by uncertainty
    ml_ranked = sorted(refined.items(), key=lambda kv: kv[1]["ce"] - args.lam * kv[1]["uncertainty"],
                       reverse=True)

    # ---- stage 3 (optional): LLM synthesizability/stability/mechanism triage ----
    assess = {}
    if args.llm:
        cand = [{"idx": i, "smiles": s, "ce": round(o["ce"], 1)} for i, (s, o) in enumerate(ml_ranked)]
        print(f"\nLLM triage of {len(cand)} candidates (reasoner: synth/stability/mechanism) ...")
        by_idx = ce_llm.assess_batch(cand, role="reasoner")
        assess = {cand[i]["smiles"]: v for i, v in by_idx.items() if i < len(cand)}
        # drop only water-UNSTABLE (fatal for aqueous Zn); red-flags stay as advisory
        # annotations so we still return a full list and let the judge weigh them.
        survivors = [(s, o) for s, o in ml_ranked
                     if "unstable" not in assess.get(s, {}).get("stability", "")]
        n_dropped = len(ml_ranked) - len(survivors)
        survivors = survivors or ml_ranked
        jin = [{"idx": i, "smiles": s, "ce": round(o["ce"], 1),
                "uncertainty": round(o["uncertainty"], 2), **assess.get(s, {})}
               for i, (s, o) in enumerate(survivors)]
        order = ce_llm.judge_rerank(jin, top=args.top, role="judge")  # idx == survivors index
        if order:
            chosen = [survivors[i] for i in order if i < len(survivors)]
            rest = [so for k, so in enumerate(survivors) if k not in set(order)]
            survivors = chosen + rest
        print(f"  dropped {n_dropped} unstable/red-flagged; "
              f"judge re-rank: {'applied' if order else 'unavailable'}.")
        ranked = survivors[:args.top]
    else:
        ranked = ml_ranked[:args.top]

    from membership import MolportIndex
    index = MolportIndex()
    lo, hi = bundle["y_range"]
    print(f"\n=== TOP {len(ranked)} candidate additives to test (xTB-refined; trained CE {lo:.1f}-{hi:.1f}%) ===")
    hdr = f"{'rank':>4} {'CE%':>6} {'+-unc':>5} {'sim':>5} {'HOMO':>7} {'Molport':>9}"
    print(hdr + ("  synth stab     mechanism / SMILES" if args.llm else "  SMILES"))
    # optional RAG: literature-novelty per candidate (1 - nearest-KB similarity)
    rag_ctx = {}
    if args.rag:
        import ce_rag
        print("\nRAG: scoring literature novelty against the KB ...")
        for smi, o in ranked:
            ctx, top = ce_rag.context_for(smi, assess.get(smi, {}).get("mechanism", ""))
            rag_ctx[smi] = (ctx, round(1.0 - top, 3))  # novelty = 1 - similarity

    rows = []
    for i, (smi, o) in enumerate(ranked, 1):
        mid = index.get_id(smi, already_canonical=True) if index.available else None
        ce = max(lo, min(hi, o["ce"]))
        a = assess.get(smi, {})
        nov = rag_ctx.get(smi, ("", ""))[1]
        base = (f"{i:>4} {ce:>6.2f} {o['uncertainty']:>5.2f} {o['domain_sim']:>5.2f} "
                f"{o['xtb_homo']:>7.2f} {mid or '-':>9}")
        if args.llm:
            print(base + f"  {a.get('synth_score','?'):>4} {a.get('stability','')[:8]:>8}  "
                  f"{a.get('mechanism','')[:40]} | {smi}")
        else:
            print(base + f"  {smi}")
        rows.append({"rank": i, "pred_ce": round(ce, 2), "uncertainty": round(o["uncertainty"], 3),
                     "lce": round(o["lce"], 3), "domain_sim": o["domain_sim"],
                     "xtb_homo": round(o["xtb_homo"], 4),
                     "synth_score": a.get("synth_score", ""), "stability": a.get("stability", ""),
                     "mechanism": a.get("mechanism", ""), "redflag": a.get("redflag", ""),
                     "lit_novelty": nov, "molport_id": mid or "", "smiles": smi})
    if index.available:
        index.close()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "pred_ce", "uncertainty", "lce", "domain_sim",
                                          "xtb_homo", "synth_score", "stability", "mechanism",
                                          "redflag", "lit_novelty", "molport_id", "smiles"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {len(rows)} candidates -> {args.out}")
    if args.llm and ranked:
        ctx = rag_ctx.get(ranked[0][0], ("", ""))[0] if args.rag else ""
        print("\nMechanistic rationale for the #1 candidate (judge%s):" % (", RAG-grounded" if ctx else ""))
        print("  " + ce_llm.rationale(ranked[0][0], ranked[0][1]["ce"], context=ctx))
    print("\nNote: random-split R^2 ~0.73, scaffold ~0.36 -> ranking/triage for NOVEL molecules is "
          "approximate. domain_sim~1 = close to a known additive (more reliable). Validate top "
          "hits, add measured CE to the CSV, re-run ce_features/ce_train (active learning).")


if __name__ == "__main__":
    main()
