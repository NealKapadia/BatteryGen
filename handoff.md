# molvae — Hand-off Summary

**Project:** a conditional SELFIES-VAE for **de-novo battery electrolyte / additive
discovery** across all battery chemistries (Li / Na / K / Mg / Ca / Zn / Al / …),
pretrained on the Molport catalog and specialized on electrolyte data, with an
inverse-design web app, property prediction, and a one-command training pipeline.

**Goal / positioning:** the best *open* generative + property model purpose-built for
battery electrolytes. Direct competitor: **ElectrolyteGPT** (Kim et al., *JACS Au* 2026,
6, 2288–2302). Our structural edge over their GPT: a **latent space** (smooth traversal,
"increase MW by 10 keeping everything else") + SELFIES (~100% validity).

Last updated: 2026-06-27.

---

## 1. Current state (read this first)

- **Preprocessing: DONE.** 6,088,143 molecules → token/descriptor shards + Molport
  membership index + descriptor stats. (`molvae_artifacts/processed/`, `…/membership/`.)
- **Base training: DONE / RECOVERED (2026-06-27).** The original 12-epoch run diverged
  after ~epoch 4 AND its good early checkpoints were lost — `prune_checkpoints(--keep 10)`
  silently deletes all but the last 10 periodic checkpoints, so by the time it was stopped
  only the diverged 40–44.5M checkpoints survived (no `best.pt` either, that run predated it).
  **Fix: a clean re-run** `train.py --epochs 6 --batch 320 --patience 2 --keep 0` (the
  `--keep 0` disables pruning). It did NOT diverge; early-stopped at epoch 4.
  - **`best.pt` = epoch 2 (~12M mols)**, selected by `val_token_acc + 0.25·valid_rate`
    ([train.py:239](train.py:239)). Metrics: val_token_acc 0.991, recon 1.93. `latest.pt` = epoch 4.
- **Validity bug FOUND + FIXED (2026-06-27): 0.767 → 1.000.** Root cause was NOT sampling —
  `data.CONSTRAINTS` was `"hypervalent"`, which lets the SELFIES decoder emit pentavalent
  neutral nitrogen that RDKit rejects (~22% of prior samples). Default constraints already
  permit S=6/P=5 (sulfonyl/phosphate), so switching to **default** lost nothing and took
  validity 0.77→1.00 AND recon 0.93→0.98 on the SAME checkpoint (no retraining). Changed in
  [data.py:30](data.py). Full eval now BEATS ElectrolyteGPT on every benchmark column
  (validity 1.0, uniq 1.0, novelty 0.994, diversity 0.893) — before electrolyte specialization.
- **Schedule fixed** (so divergence can't recur): monotonic KL (`KL_CYCLES=1`), gentler max
  (0.10), default epochs 6, `best.pt` + `--patience` early stopping, `--reset-schedule` for
  clean continued-training. **Always pass `--keep 0`** (or a large `--keep`) so good
  checkpoints aren't pruned.
- **All ~28 modules compile and pass CPU validation.** GPU *runtime* paths (app serving,
  electrolyte prediction, batch generation) are **not yet live-smoke-tested** (the GPU was
  busy training). The CPU-testable logic (LLM parsing, chemistry detection, data converters,
  report generation, relative-edit parsing) is verified.
- **In progress (incomplete):** CE / overpotential / electrochemical-window outputs and
  500-molecule **mass generation**. `config.DFT_PROPERTIES` was extended with `ip`/`ea`/
  `window`; the xTB redox computation, the `/batch` endpoint, the batch table UI, and the
  DFT-head readout in `design.py` are **not finished** (see §7).

---

## 2. Architecture & key decisions (the "why")

1. **SELFIES-VAE, not a literal GraphVAE.** The original GraphVAE (dense adjacency + graph
   matching) only works for ≤~20-heavy-atom molecules and can't scale to drug-like
   molecules on a 6 GB GPU. SELFIES trains on *all* molecules, fits 6 GB, and every decode
   is a valid molecule — while still giving the smooth latent space the use-cases need.
2. **Model = "big" profile, ~42M params** (`emb512 / GRU-1024 ×2 bi-enc + 2-layer dec /
   latent 256`), conditional on 11 RDKit descriptors, + a latent→property head. Decoder
   **word-dropout 0.25** (Bowman et al.) forces a meaningful latent. Sized for 6 GB.
3. **Vocab = SELFIES robust alphabet** (79 tokens, `hypervalent` constraints) — fixed, no
   vocab-building pass, perfect round-trip on halogens/charges/sulfonyl.
4. **Per-shard preprocessing** (13 shards), memmap-backed `MolDataset` — resumable, low RAM.
5. **Membership index** = SQLite (exact canonical-SMILES → Molport id) + Bloom (fast
   novelty pre-filter). One canonicalization (`data.canonical_smiles`, largest fragment,
   isomeric) used everywhere so generated ↔ catalog compare consistently.
6. **Property targeting = "Both":** conditional generation (sample to a spec) **and**
   latent-space optimization (gradient search via the property head).
7. **DFT grounding is tiered, honestly:** xTB (GFN2, semi-empirical) for fast labels on
   thousands of molecules + **QM9** (real DFT, 134k) for the property head. Full DFT for
   6 M molecules is infeasible and not attempted.
8. **Electrolyte = a system, not a molecule.** The VAE generates single neutral organic
   molecules (solvents / additives / anion backbones). A separate **formulation-aware,
   multi-target property model** (`electrolyte.py`) predicts conductivity / coordination /
   viscosity from `[solvent latent(s) ; anion latent ; cation one-hot ; SOURCE one-hot ;
   conc ; temp]`. The **source one-hot** reconciles CALiSol-23 (experimental) vs OEDB
   (MD-simulated) conductivity scales. Cation one-hot spans all chemistries.
9. **Inverse-design core = `design.py`** (`DesignEngine`), shared by the local stdlib
   server (`app.py`) and the FastAPI production server (`server.py`). NL → spec (LLM) →
   novelty-controlled latent optimization → RDKit 3D → electrolyte readout → LLM
   explanation.
10. **LLM via Azure routes** (`llm.py`): `judge`=gpt-5.4, `reasoner`=DeepSeek-V4-Pro,
    `fast`=Kimi-K2.6, `embed`=text-embedding-3-large. Used for request parsing +
    mechanistic explanations; **graceful keyword fallback** if keys/SDK absent.
11. **Anti-overfitting:** scaffold-held-out split (`make_split.py`), word-dropout, KL
    bottleneck + free-bits, AdamW weight decay, cross-dataset **dedup** (`add_data --dedup`),
    `best.pt` + early stopping. Benchmarks (validity/uniqueness/novelty/diversity) don't
    need the split; the split is for *proving* reconstruction generalization.
12. **Hardware-driven knobs:** **batch 320** is the measured throughput/VRAM sweet spot
    (~912 mol/s @ 3.14 GB); **≥384 tips over the 6 GB-laptop VRAM-spill cliff** (WDDM →
    system RAM → ~4× slowdown).

---

## 3. Datasets wired

| Dataset | Path / source | Role |
|---|---|---|
| Molport "All Stock" | `../SMILES/*.gz` (+ `*.sdf.gz`) | base pre-training (6.09M) |
| ZINC-250k | `../250k_rndm_zinc_drugs_clean_3.csv` | continued pre-training |
| ChEMBL-37 | `../chembl_37.sdf` (8 GB, ~2.4M) | continued pre-training (sampled, RDKit-streamed) |
| CALiSol-23 | `../calisol23_DOI_...csv` | electrolyte conductivity (wide format, Li) |
| OEDB-electrolytes | `../oedb-electrolytes-...csv` | conductivity + **coordination** + viscosity (Li/Na/K, MD) |
| OEDB-experimental | `../oedb-experimental-...csv` (100 MB) | Raman spectra — **NOT wired** (future) |
| OEDB-rdf | `../oedb-rdf-...csv` (35 MB) | g(r) radial-distribution — **NOT wired** (future) |
| QM9 | auto-download via PyG | real DFT labels → property head |
| PubChem-124M | HF `hheiden/PubChem-124M-...` (streaming) | pretrain sample + novelty-vs-PubChem bloom |
| OpenQDC / atomic-datasets | optional pip | 3D/quantum grounding (optional) |

`electrolyte_data.py` already merged CALiSol-23 + OEDB → **`molvae_artifacts/electrolyte_train.csv`
(18,918 formulations, cations Li/Na/K)**. Solvent-abbrev→SMILES and salt→cation/anion
dictionaries are in `solvent_lib.py`.

---

## 4. Codebase map (everything is in `molvae/`)

**Core ML**
- `config.py` — all paths, hyper-parameters, property lists, the `big`/`small` profile switch.
- `data.py` — canonicalization, RDKit descriptors, SELFIES `Vocab`, `MolDataset`, scaffold-split loading.
- `model.py` — `SelfiesVAE` (encoder/decoder/property-head) + `vae_loss` (recon + KL free-bits + prop).
- `membership.py` — `BloomFilter` + SQLite `MolportIndex`.
- `preprocess.py` — streaming multiprocessing preprocessor (`process_file` reusable; `--dedup` support).
- `train.py` — GPU loop, AMP, cyclical/monotonic KL, cosine LR, 500k checkpoints, `best.pt`,
  `--patience`, `--reset-schedule`.
- `infer.py` — shared loaders/helpers (`load_model`, `spec_to_condition`, `score_smiles`, dataset iter).

**Generation / evaluation**
- `generate.py` — CLI sampling (spec / NL prompt / `--molport-only`).
- `search.py` — property search: `--mode dataset` (catalog filter) or `--mode latent` (optimization).
- `evaluate.py` — the 6 ElectrolyteGPT metrics + side-by-side table; auto-detects `pubchem.bloom`.
- `report.py` — HTML report (training curves + benchmark + property dist + latent PCA).
- `make_split.py` — Bemis-Murcko scaffold-held-out validation split.

**Grounding / specialization**
- `xtb_label.py` — RDKit 3D → xtb.exe GFN2 → HOMO/LUMO/gap/dipole (redox IP/EA **TODO**, §7).
- `finetune_dft.py` — train a latent→DFT head (multi-target) + DFT-targeted generation.
- `qm9.py` — QM9 → labels CSV for `finetune_dft`.
- `solvent_lib.py`, `electrolyte_data.py` — electrolyte dataset normalization.
- `electrolyte.py` — formulation multi-target property model (`--mix-col`, source one-hot) + screening.
- `add_data.py` — continued-pretrain append (SDF/CSV/txt readers, `--limit`, `--dedup`, `--no-…`).
- `openqdc_data.py`, `hf_data.py` — optional OpenQDC / HF PubChem loaders.

**App / serving / deploy**
- `design.py` — `DesignEngine`: NL→design, novelty-controlled optimization, electrolyte readout,
  RDKit 3D, LLM `explain`, light user `finetune`.
- `llm.py` — Azure routes, `nl_to_spec`, `parse_relative`, `parse_design`, `explain`, `embed`.
- `app.py` — local zero-dep stdlib server (`/design /refine /explain /finetune /health`).
- `server.py` — FastAPI production server (same endpoints, threadpool, CORS).
- `static/index.html` — UI: prompt, novelty + property sliders, cation/conc/temp, 3Dmol viewer,
  electrolyte panel, explanation, "teach the model", history.
- `Dockerfile`, `docker-compose.yml`, `Caddyfile`, `.dockerignore` — production container + auto-HTTPS.
- `pipeline.py` — **the one-command orchestrator** (6 resumable stages).

**Docs / config**
- `README.md` (usage), `DEPLOY.md` (full Azure deploy), `requirements.txt`, `.env.example`,
  `.env` (real Azure keys present), this `handoff.md`.

**Artifacts** (`../molvae_artifacts/`): `processed/` (shards, vocab.json, descriptor_stats.json,
meta.json), `checkpoints/` (checkpoint_mols_*.pt, latest.pt), `membership/`
(molport.sqlite + .bloom), `dft/` (xtb labels), `electrolyte_train.csv`.

---

## 5. Environment

Windows 11, PowerShell. **Python 3.13 system install** (`C:\Python313`, no venv — a venv
would be cleaner but isn't required). `torch 2.6.0+cu124` on an **RTX 3060 Laptop (6 GB)**,
`rdkit 2026.03`, `selfies 2.1.1`, `torch_geometric 2.8`, `numpy/tqdm/matplotlib/sklearn`,
`openai`+`python-dotenv` (real Azure keys in `.env`, tested working). **xTB 6.7.1** binary at
`C:\Users\nkapa\Downloads\xtb-6.7.1pre-windows-x86_64\xtb-6.7.1\bin\xtb.exe` (config.XTB_EXE).
**Not installed** (install when needed; can shift torch/numpy so do it when *not* training):
`fastapi`, `uvicorn`, `datasets`, `openqdc`, `atomic-datasets`.

Run scripts with **absolute paths** (the PowerShell CWD has drifted into `molvae/` before;
scripts add their own dir to `sys.path` so location doesn't matter). Tee/grep pipes
block-buffer Python output — use `python -u` for live logs.

---

## 6. Immediate next steps (in order)

1. **Recover the base model** (the diverged run's best checkpoint):
   ```powershell
   # Ctrl+C the training if still running, then with the GPU free:
   python molvae\evaluate.py --ckpt molvae_artifacts\checkpoints\checkpoint_mols_12000000.pt --gen 3000
   python molvae\evaluate.py --ckpt molvae_artifacts\checkpoints\checkpoint_mols_24000000.pt --gen 3000
   copy molvae_artifacts\checkpoints\checkpoint_mols_12000000.pt molvae_artifacts\checkpoints\best.pt
   copy molvae_artifacts\checkpoints\checkpoint_mols_12000000.pt molvae_artifacts\checkpoints\latest.pt
   ```
   (Optional clean re-run for a marginally better base, ~6–9 h, auto-stops near epoch 4:
   `python molvae\train.py --epochs 6 --batch 320 --patience 2`.)
2. **Run the specialization pipeline** (now with the fixed gentle schedule + `--reset-schedule`):
   ```powershell
   python molvae\pipeline.py --dry-run     # preview
   python molvae\pipeline.py               # electrolyte_data → add_data(dedup) → continued_train
                                           # → QM9 ground → electrolyte_train → eval
   ```
3. **Finish the in-progress features** (§7): xTB redox (IP/EA/window), DFT-head readout in
   `design.py`, `/batch` mass-generation endpoint + table UI.
4. **Optional grounding/benchmark extras:** `pip install datasets` →
   `python molvae\hf_data.py --mode bloom --max 20000000` (novelty-vs-PubChem).
5. **Smoke-test the app on GPU:** `python molvae\app.py --device cuda` → try
   *"possible additives for a zinc aqueous battery"*; verify electrolyte readout + explanation.
6. **Deploy:** `pip install fastapi "uvicorn[standard]"`, then follow **`DEPLOY.md`**
   (Azure GPU VM + NVIDIA toolkit + domain + DNS + Caddy auto-HTTPS, or the cheaper CPU
   Container-Apps path).

---

## 7. Unfinished work / TODO (CE, overpotential, mass generation)

**DONE this session (2026-06-27)** in `design.py` (engine only; endpoints/UI still TODO):
- `DesignEngine.batch(prompt, sliders, n, spec=..., pool_mult=...)` — range-aware mass
  generation: over-generates a large latent-optimized pool (cheap at validity 1.0), filters
  to any `[lo, hi]` ranges, dedups, scores (RDKit props + novelty + optional electrolyte +
  optional DFT head), returns a ranked table. **This is the production answer to soft
  conditioning** (see caveat below): MolWt-range 100–250 now yields 100%-in-range molecules;
  point target MolWt 400/QED 0.85 → realized ~380/0.83.
- `DesignEngine.suggest_similar(tested, n, spread)` — "enter molecules you've tested
  (optionally with a performance score) → suggest novel molecules similar to the best
  performer(s)." Encodes seeds, samples latent neighbors conditioned on the seed's
  descriptors, ranks by Tanimoto-to-seed × seed performance.
- `DesignEngine._load_dft()` + `dft_predict(smiles)` — loads `dft_latest.pt` (its own
  encoder+head) and predicts homo/lumo/gap/dipole/ip/ea/window; graceful no-op until the
  pipeline produces that file.
- **Bug fixed:** `_sample_pool` was `@torch.no_grad()` but runs gradient latent optimization
  → `design.optimize()` (the app's core call) crashed on any property spec. Decorator removed.
- **Bug fixed (Windows):** `add_data --dedup` opened a read-only `MolportIndex` SQLite handle
  that stayed open through `preprocess.finalize()`'s `MOLPORT_DB.unlink()` → `PermissionError
  [WinError 32]` (can't delete an open file). Added `MolportIndex.close()` and call it in
  `add_data` before `finalize`. This had blocked the whole specialization pipeline at stage 2.

**Still TODO:**
- `xtb_label.py`: add a `--redox` path — vertical IP = E(cation,+1) − E(neutral),
  EA = E(neutral) − E(anion,−1) (×27.2114 eV/Eh), window = IP − EA; run the three
  single-points on the *same* RDKit/MMFF geometry. Add to the labels CSV.
- `app.py` + `server.py`: add `/batch` (calls `DesignEngine.batch`) and `/similar`
  (calls `suggest_similar`) endpoints.
- `static/index.html`: "Batch" tab (sortable table, CSV export, click-row→3D) + a
  "my tested molecules" input. (UI deferred per user.)
- **Soft-conditioning caveat:** point conditioning AND gradient latent-search are both
  compressed toward the training mean (MolWt target 200→500 → realized ~317→401; QED nearly
  flat). `batch()`'s over-generate+range-filter is the robust workaround. A stronger fix
  (FiLM conditioning / higher prop-loss weight / lower-KL informative latent) would need
  retraining and is optional. Small electrolyte molecules (EC/DEC/MeCN) are OOD for the
  drug-like base — electrolyte specialization (the pipeline) addresses that.
**CE inverse design — DONE (2026-06-28), `ce_model.py`:** Zn-additive Coulombic-Efficiency
predictor + inverse-design loop over `Supplementary_Data_1.csv` (575 rows, 168 unique
additives, target `CE_aver.`). Featurizer is pluggable (`--features ecfp+rdkit` default;
`latent` = VAE). Modes: `train` (model-select Ridge/RF/HGB, `--select grouped|test`, Optuna
fallback), `screen` (global), `suggest` (seed from known winners), `design` (the full loop:
global generate → score → re-seed from best-scored → repeat `--rounds` → top-N CSV),
`predict`. Runs CPU (`--device cpu`) so it won't fight the GPU pipeline. Candidates pass a
`chem_ok` stability filter (drops acyl halides, N-N, N-halamine, peroxide, allene, 3-rings).
- **v1 (ECFP+RDKit, RF):** random R²≈0.66, grouped≈0.17. Superseded by v2 below.

**CE v2 — xTB-physics workflow (2026-06-28), `ce_features.py` + `ce_tune.py` + `ce_train.py`
+ `ce_design.py`:** following the user's step2b/step3 design.
- **Features = RDKit-19 + xTB-7 (HOMO/LUMO/gap/chi/eta/omega/dipole, GFN2 single-point via
  `xtb_label.run_xtb`, cached in `ce/xtb_cache.csv`) + LMR.** `xtb_omega` (electrophilicity) is
  the top feature — physically right for CE. `ce_features.py` builds `ce/feature_cache.pkl`
  (X, y, COV from CE triplicates, scaffold groups, Morgan fps).
- **Model = ExtraTrees + XGBoost blend** (`et_w` tuned). **SVR was DROPPED** (it degraded R²
  here, 0.33 alone) and **ECFP was DROPPED** (diluted the xTB signal, 0.72→0.64). Optuna
  (`ce_tune.py`, dual scaffold+random objective) tunes both + blend weight.
- **Honest metrics: random-split R²≈0.73 (5-fold CV), scaffold R²≈0.36.** Big lift over v1
  (esp. scaffold 0.17→0.36 from xTB). **0.75 is NOT reachable with 168 molecules** — it's a
  data-quantity ceiling, not tuning. Levers: more CE data (active learning) or richer physics
  features (Fukui, solvation).
- `ce_train.predict(bundle, smiles, lmr, compute_xtb=False/True)` returns
  {ce, lce, uncertainty(ensemble std), domain_sim, xtb_homo}. `compute_xtb=False` imputes xTB
  from medians (fast screen); True runs real xTB (accurate shortlist).
- **`ce_design.py`** = inverse-design loop: VAE generate -> RDKit-screen large pool ->
  re-seed from best -> **xTB-refine shortlist** -> rank by `CE - lam*uncertainty` ->
  `ce_candidates.csv` (CE, uncertainty, LCE, domain_sim, xTB HOMO, Molport id). All CPU.
- Deps added: `xgboost`, `optuna` (pip, CPU). Default generator ckpt = `best.pt`.

**LLM integration (2026-06-28), `ce_llm.py` + `llm.complete()`:** reuses the existing
`llm.py` Azure routes (fast=Kimi-K2.6, reasoner=DeepSeek-V4-Pro, judge=gpt-5.4,
embed=text-embedding-3-large; keys in `molvae/.env`, all 4 verified live). Added a generic
`llm.complete(system,user,role,want_json)` chat helper + `llm.available(role)`. `ce_llm.py`:
`parse_request` (NL → property ranges/LMR/avoid, fast), `assess_batch` (per-molecule
synth_score/stability/mechanism/redflag, reasoner), `judge_rerank` (frontier final pick,
judge), `rationale` (mechanistic CE paragraph, judge). All degrade to no-ops without keys.
- **`ce_design.py --prompt "..." --llm`**: NL request constrains generation; after xTB-refine,
  LLM triages (drops water-unstable/red-flagged), judge re-ranks survivors, writes
  synth_score/stability/mechanism/redflag columns to `ce_candidates.csv`, and prints a
  gpt-5.4 rationale for #1. Console reconfigured to utf-8 (LLM text has Zn²⁺ etc.).
- The VAE app side (`design.py`/`app.py`) already used `llm.parse_design`/`explain`, so LLM now
  spans both the generator and the CE predictor.

**RAG literature KB (2026-06-28), `ce_rag.py`:** user-extendable store over `text-embedding-3-large`
in `ce/kb/` (chunks.jsonl + emb.npy, cosine retrieval). `--add file` / `--add-text` / `--query`
/ `--list`. `ce_design --rag` grounds the #1 rationale in retrieved passages (cites [source])
and writes a `lit_novelty` column (1 − nearest-KB similarity). Seeded with one starter note.

**`ce_design.py` updates (2026-06-28):** property ranges from `--prompt` are now ADVISORY
(only hard-filter with `--apply-ranges`); `--seed-best K` seeds round 0 from the top-K
highest-CE training additives (bias toward the 98–99% region); defaults `--top 50`,
`--shortlist 80`; `--min-ce` filter; `--rag` for grounded rationale + novelty.
- **KEY LIMITATION — the model cannot predict ≥98% for NOVEL molecules (confirmed twice).**
  ExtraTrees/XGBoost can't extrapolate above training leaf-averages; only the exact 98–99%
  training molecules sit there. **LCE-target** training (`ce_train.py --target lce`, now default;
  CE = 100·(1−10^−pred)) lets the *known* performers recover their true 98+ with real xTB
  (histidine 98.8, TEA 98.9, Na-acetate 98.9) and nudged the *novel* ceiling 97.47→97.66 — but
  novel molecules still cap ~97.6. CE-space OOF R² 0.69 (vs 0.73 raw; log amplifies high-end
  error). **Active learning (`ce_features.py --append-smiles ... --append-ce ...` then
  `ce_train.py`) is the only real path to confirmed novel 98+.**
- **Fast-screen under-predicts ~2–3%** (imputes constant xTB); the xTB-refine stage gives the
  true (higher) CE — so always trust the refined column, not the screen.
- **chem_ok now also rejects bare cations and non-carboxylate/sulfonate anions** (e.g. alkoxide
  dianions [O-]C[O-] the model over-scored as "polyol-like" but are just strong bases). Keeps
  neutrals + carboxylate/sulfonate/sulfonimide anions.
- **Active-learning helper:** `ce_features.py --append-smiles SMI --append-ce X [--append-lmr/-zn/-addmole]`
  appends a measured row (triplicate=CE, conc cols default to medians) and rebuilds the cache;
  then `ce_train.py` retrains. One measurement at a time closes the loop.

**valid_sample_rate is NOT inflated** (user asked): [train.py:104-112](train.py:104) uses
`valid/n` with n=200 requested and `net.sample` returns all 200 (no pre-filter) — denominator
is the full request. The 1.000 in continued-train is the genuine effect of the default-SELFIES
constraint fix (100% RDKit-valid decodes), same as evaluate.py validity 1.000.
- **Perf note:** continued-train epoch 1 ran at 159 mol/s (12 h) because CPU-heavy CE work
  (xTB ×168 + Optuna) was running concurrently; epoch 2 recovered to ~1088 mol/s. Don't run
  big CPU jobs (xTB, large `ce_design`) during GPU training.

- **CE / overpotential / cycle-life:** these are *system/experimental* and can't be computed
  from one structure. The multi-target `electrolyte.py` already predicts any target column —
  so when CE/overpotential-labeled formulation data is available (literature / EDB / a forward
  model), add it as columns to `electrolyte_train.csv` and they're learned. The LLM
  `explain()` covers qualitative CE/SEI reasoning.

---

## 8. Known caveats / risks

- **Base run diverged** — recover the epoch-2/4 checkpoint; the schedule is fixed for re-runs.
- **Electrolyte property model has data only for Li/Na/K.** The cation conditioning supports
  all chemistries and the *generator* proposes molecules for any cation, but *ranking*
  Mg/Zn/Ca/Al by conductivity needs labeled data for them (or xTB/DFT-generated labels).
- **CALiSol (experimental) vs OEDB (MD) conductivity scales differ** — handled by the source
  one-hot; screen with `--source oedb` or `--source calisol23` (a trained source).
- **xTB ≠ full DFT** (semi-empirical; trend-correct, not benchmark-accurate).
- **OEDB experimental/rdf** (Raman / g(r)) not yet used — a future spectral/solvation-structure target.
- **App serving SMOKE-TESTED (2026-06-28, CPU).** `design.DesignEngine.design` works for both
  slider and NL-prompt requests + `/explain`; returns molecule + 3D molblock + electrolyte
  readout. Fixed 3 crash bugs on the LLM-range-spec path (`data.spec_to_condition`,
  `design.optimize` err, design response serializer all did `float(val)` on `[lo,hi]` lists).
- **Web app v2 (2026-06-29) — prompt-driven, bring-your-own-data inverse design.** New
  `webengine.py` (ephemeral, in-memory; user CSVs never written to disk): `train_from_csv`
  (LLM maps an NL objective like "increase CE while reducing overpotential" → target
  columns+direction, trains a per-target RandomForest on ECFP+RDKit, returns session_id +
  CV R²), `generate` (VAE pool → score by the session objective → enrich with RDKit features
  + LLM IUPAC name + 3D top), `molecule_info`, `add_literature` (capped RAG). Rewrote
  `server.py` (FastAPI v2): `/generate /train /train_pkl(gated MOLVAE_ALLOW_PKL) /molecule
  /literature /health`; **dropped public /finetune**. Rewrote `static/index.html`: no sliders,
  prompt + suggested-prompt chips, bulk count, CSV-upload + objective box, literature box,
  result cards (features/IUPAC/Molport badge/3Dmol viewer). Verified backend end-to-end on CPU
  (train CV R² 0.61 on the Zn CSV; generate returns scored+named+3D molecules, ~57 s/n=5).
  **Dockerfile.cpu deps extended** (pandas, scikit-learn, joblib, python-multipart) — REQUIRED
  for v2. `.pkl` upload = RCE risk, disabled by default. Sessions expire after 1 h.
- **Web v2.1 (2026-06-29) — Bayesian optimization + async + transparency.** `webengine._bo_search`:
  latent-space BO with an **RF surrogate + UCB acquisition** (mean + kappa*std); each round picks
  highest-acquisition molecules, decodes latent NEIGHBORS (encode→perturb→decode), evaluates,
  accumulates. RF posterior = per-tree variance (`_rf_mean_std`); combined-objective uncertainty
  propagated. `generate(method='bo')` is default when a session model exists. Output now carries
  per-molecule `uncertainty`; `/train` already returns CV R² (shown in UI). **Async jobs**:
  `/generate` returns `{job_id}` and runs in a daemon thread (BO is 1-4 min on CPU → would time
  out a sync request); UI polls `/job/{id}`. BO verified on CPU (n=6: oxazolidine CE 96.95±0.20).
  **Cost guidance:** CPU min-replicas 0 ≈ $3-10/mo ($150 ~1yr); min-replicas 1 ≈ $45/mo (~3mo);
  GPU T4 VM always-on ≈ $380/mo (use ON-DEMAND only). LLM ~$0.05-0.15/request. GPU makes BO
  ~10-20s vs minutes on CPU — recommended for live demos via on-demand VM, deallocate after.
- **Web v2.2 (2026-06-29) — speed + packaging.** BO/sample budget now SCALES with n (n<=4 -> 1
  round, batch=min(max(n*12,30),160); sample pool=min(max(n*4,24),480)) so small requests are
  fast. **3D deferred** (`want_3d_top=0`): no molblock until the user clicks "Show 3D" (/molecule).
  Name = **MolForge**. For ~1-2x/month: CPU Container App 4 vCPU/8 GiB scale-to-zero (bill only
  ~40s/request) as the always-on link, OR on-demand GPU VM (`Dockerfile.gpu`,
  `gpu_start.ps1`/`gpu_stop.ps1`, ~$0.50/hr only while on).
- **`molforge.py` — public Python library/API** (local CPU/GPU, no Azure): `MolForge(device=)
  .generate(n, spec=)`, `encode`/`decode`, `properties`, `predict_properties`; CLI
  `python molforge.py --n 20 --device cuda --out x.csv`. Verified. Needs molvae code + artifacts
  (processed/ + checkpoints/best.pt). `requirements-lib.txt` + `LIBRARY.md` document local use.
- **Web v2.3 (2026-06-29) — major speed fix.** `ce_model._generate` was over-sampling 4x (legacy
  from validity 0.77); now generates ~1.3x the shortfall (validity ~1.0) — **n=2 no-model generate
  dropped from ~4 min to 0.4 s locally.** **IUPAC naming deferred** out of bulk generate (it was a
  failing/slow synchronous LLM call → "unnamed") to the per-molecule /molecule click (now "Name &
  3D structure" button fetches name + 3D together). **MUST redeploy** (Dockerfile.cpu :v5) — live
  site still runs the slow build.
- **API vs library decision:** to run on a user's OWN hardware -> Python LIBRARY (`molforge.py`),
  not a hosted API (which runs on the provider's server = the public web app for zero-install use).
  The library also subsumes a local REST API via `uvicorn server:app`. Recommended distribution:
  GitHub repo + weights on Hugging Face; full `pip install molforge` packaging is a small follow-up.
- **Deploy assets ready:** `Dockerfile.cpu` (CPU image, bakes a staged `artifacts/` bundle),
  `deploy_prepare.ps1` (stages minimal bundle: best.pt+vocab+stats+electrolyte_model, ~481 MB,
  skips the 922 MB Molport sqlite), and a copy-paste **Azure Container Apps** path in DEPLOY.md.
  Verified the staged bundle loads + serves via MOLVAE_ART_DIR. RAG capped at 10 docs
  (`CE_RAG_MAX_SOURCES`). Deployed demo = generative designer only (server.py: /design /refine
  /explain /finetune); CE/RAG remain CLI (Phase 2). `molvae/artifacts/` is the deploy staging dir.
- **Formulation generation** is molecule-level + a property model; ElectrolyteGPT's `fLine`
  (whole salt+solvent+ratio+conc strings) is a possible future extension.

---

## 9. Quick command reference

```powershell
# preprocess (DONE) / continued data
python molvae\preprocess.py                         # full catalog (done)
python molvae\add_data.py --input file.smi --tag x --dedup

# train / recover / continue
python molvae\train.py --epochs 6 --batch 320 --patience 2
python molvae\train.py --resume --reset-schedule --epochs 3 --patience 2

# generate / search / evaluate / report
python molvae\generate.py --n 20 --prompt "small soluble carbonate solvent" --molport-only
python molvae\search.py --mode dataset --spec "{\"MolWt\":[300,350]}"
python molvae\evaluate.py            ;  python molvae\report.py --latent

# grounding + electrolyte specialization
python molvae\xtb_label.py --n 1000  ;  python molvae\qm9.py
python molvae\finetune_dft.py --target homo,lumo,gap,dipole --labels molvae_artifacts\dft\qm9_labels.csv
python molvae\electrolyte_data.py
python molvae\electrolyte.py --mode train --csv molvae_artifacts\electrolyte_train.csv ^
   --mix-col mix --cation-col cation --anion-smiles-col anion_smiles --conc-col conc ^
   --temp-col temp --source-col source --target-cols conductivity,coord_cat_anion,coord_cat_solvent --log-target
python molvae\electrolyte.py --mode screen --cation Li --source oedb --n 200

# app / pipeline / deploy
python molvae\app.py --device cuda
python molvae\pipeline.py
# deployment: see DEPLOY.md
```

Project memory (persists across sessions): see
`~/.claude/projects/.../memory/` — `molvae-pipeline.md`, `molvae-environment.md`,
`molvae-electrolyte-roadmap.md`.
