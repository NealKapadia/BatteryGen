# molvae — a conditional SELFIES-VAE over the Molport stock catalog

A variational autoencoder that learns a continuous latent space of ~6.5M purchasable
Molport molecules, lets you **search by molecular properties**, **generate new molecules
to a spec**, check whether a molecule is **already in Molport**, and **fine-tune on
xTB (semi-empirical DFT-surrogate) labels**.

Why SELFIES instead of a literal GraphVAE: the original GraphVAE reconstructs a dense
adjacency matrix with graph matching and only works for tiny molecules (≤~20 heavy
atoms). A SELFIES sequence-VAE trains on *all* of your drug-like molecules, fits in
6 GB of VRAM, and **every** string it decodes is a valid molecule. It still gives you
the thing you actually wanted: a smooth latent space you can search and condition on.

```
raw SMILES.gz  ──preprocess──▶  SELFIES tokens + RDKit descriptors + Molport index
                                          │
                                          ▼
                              conditional SELFIES-VAE  ──train (GPU)──▶ checkpoints
                                          │
              ┌───────────────┬───────────┴───────────┬────────────────┐
              ▼               ▼                       ▼                ▼
        generate to      latent-space            in_molport /     xTB labeling +
        a property       property search        --molport-only   DFT-surrogate
        spec             / optimization          membership       fine-tune
```

## 0. One-time setup

```powershell
# from this folder's parent:  C:\Users\nkapa\Downloads\Molport_Full_Database\All Stock Compounds
pip install -r molvae\requirements.txt
copy molvae\.env.example molvae\.env      # only needed for the optional LLM parser
```

Everything is driven by `molvae\config.py`. Artifacts (shards, checkpoints, the
Molport index, samples) are written to `molvae_artifacts\` next to the data.

## 1. Preprocess  →  tokens + descriptors + Molport index

```powershell
# Quick subset to validate the pipeline (fast):
python molvae\preprocess.py --limit 50000

# The full catalog (uses all CPU cores; takes a while, see progress bar):
python molvae\preprocess.py
```

Produces `tokens.npy`, `lengths.npy`, `descriptors.npy`, `vocab.json`, `meta.json`,
and the Molport membership index (`molport.sqlite` + `molport.bloom`).

## 2. Train (GPU, progress bar, checkpoint every 500k molecules)

The default is the **"big" profile** — a ~42M-param model (emb512 / GRU-1024×2 /
latent256) with decoder word-dropout, cyclical-KL annealing, and cosine-LR warm-up.
On a 6 GB RTX 3060 Laptop the measured sweet spot is **batch 320 (~912 mol/s, 3.1 GB)**:

```powershell
# Best-quality long run (~18 h ≈ 10 epochs over the full 6.09M):
python molvae\train.py --epochs 10 --batch 320

# Resume from the latest checkpoint at any time (stop/resume freely):
python molvae\train.py --resume --batch 320

# Fast/small model instead (the original 4M-param profile):
$env:MOLVAE_PROFILE="small"; python molvae\train.py --epochs 10
```

⚠️ **Don't raise the batch above ~320** on a 6 GB card: batch ≥384 pushes peak VRAM
over the spill cliff (WDDM falls back to system RAM → ~4× slowdown). Scale epochs, not
batch, for a longer run.

The tqdm bar shows molecules-seen, recon / KL / property loss, beta, lr, and VRAM. A
checkpoint is written every `CHECKPOINT_EVERY` (default 500,000) molecules to
`molvae_artifacts\checkpoints\` plus a `latest.pt` for resuming.

## 3. Generate molecules

```powershell
# Unconditional samples:
python molvae\generate.py --n 20

# To a property spec (RDKit descriptor targets):
python molvae\generate.py --n 20 --spec "{\"QED\":0.85,\"MolWt\":350,\"MolLogP\":2.5}"

# From a natural-language request (uses the LLM parser if keys are set, else keywords):
python molvae\generate.py --n 20 --prompt "small, very soluble, drug-like, few rotatable bonds"

# Restrict output to molecules that already exist in Molport:
python molvae\generate.py --n 20 --spec "{\"QED\":0.85}" --molport-only
```

Each generated molecule is reported with its properties, validity, and—if present—its
`Molport-xxx` ID.

## 4. Property search over the trained latent space

```powershell
python molvae\search.py --spec "{\"MolWt\":300,\"QED\":0.9,\"TPSA\":60}" --n 10
```

Gradient optimization in latent space toward the target (uses the property head),
decoded to valid molecules and de-duplicated.

## 5. xTB labeling + DFT-surrogate fine-tune

```powershell
# Compute HOMO/LUMO/gap/dipole with GFN2-xTB on a subset (uses your xtb.exe):
python molvae\xtb_label.py --n 2000 --source dataset

# Fine-tune the model + property head on those labels (e.g. target the HOMO-LUMO gap):
python molvae\finetune_dft.py --target gap --epochs 30
```

After fine-tuning you can `generate.py --spec "{\"gap\":3.5}"` to bias toward an
xTB HOMO–LUMO gap of ~3.5 eV.

## Files

| file | what it does |
|------|--------------|
| `config.py` | all paths, hyper-parameters, the property list |
| `data.py` | SELFIES vocab, descriptors, the torch `Dataset` |
| `membership.py` | Molport Bloom filter + SQLite (in_molport, get id) |
| `preprocess.py` | streaming SMILES → tokens/descriptors/index (multiprocessing) |
| `model.py` | conditional SELFIES-VAE + latent property head |
| `train.py` | GPU training loop, AMP, KL anneal, 500k checkpointing |
| `generate.py` | sampling, spec/NL conditioning, `--molport-only` |
| `search.py` | latent-space property optimization |
| `xtb_label.py` | RDKit 3D → xtb.exe GFN2 → HOMO/LUMO/gap/dipole |
| `finetune_dft.py` | fine-tune on xTB labels |
| `llm.py` | Azure routes + natural-language → property spec |

## 6. Interactive de-novo designer (inverse design)

A web app for full inverse design — type a need, get a novel molecule in 3D with a
mechanistic explanation, and steer it with sliders.

```powershell
python molvae\app.py                 # local, CPU by default; opens http://localhost:8000
python molvae\app.py --device cuda   # after training
```
Flow: **"possible additives for a zinc aqueous battery"** → the LLM (your Azure judge
model) infers cation=Zn / aqueous / role=additive + property targets → novelty-controlled
latent optimization → RDKit 3D **ball-and-stick** (3Dmol.js) → predicted **conductivity +
coordination** for that cation → **"Explain this molecule"** (LLM mechanistic rationale).
Controls:
- **Novelty slider** — blends toward known-catalog (low) vs genuinely novel (high).
- **Mechanistic sliders** — enable MW / logP / TPSA / QED / rotatable bonds / aromatic rings to constrain them.
- **Cation / conc / temp** — drives the electrolyte property readout.
- **Refine** — relative edits ("increase MW by 10 with the same optimization") that re-optimize from the *same* latent point (a VAE-only capability vs ElectrolyteGPT's GPT).
- **Teach the model** — paste your SMILES for a short low-LR fine-tune (`user.pt`).

**Production**: `server.py` (FastAPI) + `Dockerfile` + `docker-compose.yml` (Caddy
auto-HTTPS). Full Azure deployment — GPU VM, domain, DNS, TLS, secrets, costs — is in
**[DEPLOY.md](DEPLOY.md)**.

## 7. Benchmark against the ElectrolyteGPT paper

```powershell
python molvae\make_split.py             # (once, before training) honest scaffold-held-out val
python molvae\evaluate.py               # validity/uniqueness/novelty/similarity/diversity + table
# optional novelty-vs-PubChem: build a Bloom from a PubChem SMILES dump, then pass it
python molvae\evaluate.py --build-pubchem-from pubchem_smiles.txt
python molvae\evaluate.py --pubchem-bloom molvae_artifacts\membership\pubchem.bloom
python molvae\report.py --latent        # HTML report: curves + benchmark + latent PCA
```
`evaluate.py` prints our 6 metrics next to the paper's Table 1 (Mol-CycleGAN, JT-VAE,
MinGPT, diffusion models, ElectrolyteGPT). Similarity = mean pairwise Tanimoto,
Diversity = 1 − Similarity (the paper's convention).

## 8. Specialize for electrolytes (continued learning)

```powershell
# continued pre-training on electrolyte-relevant molecules (keeps conditioning calibrated)
python molvae\add_data.py --input zinc_solvents.smi edb_solvents.smi --tag electrolyte
python molvae\train.py --resume --batch 320

# ground electronic properties on real DFT (QM9), then xTB top-ups
python molvae\qm9.py
python molvae\finetune_dft.py --target homo,lumo,gap,dipole --labels molvae_artifacts\dft\qm9_labels.csv

# formulation conductivity model across ALL cations + screening
python molvae\electrolyte.py --mode demo                         # synthetic smoke test
python molvae\electrolyte.py --mode train --csv CALiSol-23.csv --smiles-cols solvent_smiles \
      --cation-col cation --conc-col molality --temp-col T_K --target-col conductivity --log-target
python molvae\electrolyte.py --mode screen --cation Li --source oedb --conc 1.0 --temp 298 --n 200
```
The cation **conditioning** spans every chemistry (Li/Na/K/Rb/Cs/Mg/Ca/Sr/Ba/Zn/Al/Fe/H/NH4/organic),
but the **conductivity model is only as good as its data** — the wired CALiSol-23 + OEDB cover
**Li/Na/K**. To screen Mg/Zn/Ca etc., add labeled data for those cations (or generate xTB/DFT
labels) and retrain stage 5. The generator proposes candidate molecules for any cation regardless.

## 9. ⭐ One command to finish the model (after base training)

Once `train.py --epochs 12 --batch 320` has finished, **one command** does the rest —
continued pre-training on more data, DFT grounding, and electrolyte specialization:

```powershell
python molvae\pipeline.py            # runs all 6 stages, resumable
python molvae\pipeline.py --dry-run  # preview the plan first
```

Stages (each skipped if already done; resume with `--from <stage>`):
1. **electrolyte_data** — CALiSol-23 + OEDB → `electrolyte_train.csv` (18,918 formulations, Li/Na/K)
2. **add_data** — continued-pretrain corpus: ZINC-250k + ChEMBL sample (`--chembl-limit`, default 800k) + electrolyte solvents → new shards
3. **continued_train** — `train.py --resume` on the expanded ~7M dataset (`--epochs`, default 15)
4. **ground** — QM9's real DFT labels → `finetune_dft` property head
5. **electrolyte_train** — multi-target conductivity + **coordination** model, cation-aware (Li/Na/K/Mg/Ca/Zn/…)
6. **eval** — benchmark vs ElectrolyteGPT + HTML report

Knobs: `--chembl-limit 0` (all 2.4M), `--epochs 14` (less continued training), `--qm9-max`.
Rough time on your RTX 3060: stage 3 dominates (~6 h for 3 extra epochs over 7M); the
rest is minutes. Optional extras (install when *not* mid-training, they can change
torch/numpy): `pip install openqdc atomic-datasets` then `python molvae\openqdc_data.py`
before the pipeline to fold in GEOM 3D/quantum data.

After it finishes:
```powershell
python molvae\app.py                                              # interactive designer
python molvae\electrolyte.py --mode screen --cation Mg --source experimental --n 200
```

## Notes / limits

- **xTB ≠ full DFT.** GFN2-xTB is a fast semi-empirical surrogate. Good for trends and
  fine-tuning at the scale of thousands of molecules; not a substitute for high-level DFT.
- The **Molport membership** check is exact via SQLite (canonical SMILES → Molport ID);
  the Bloom filter is just a fast pre-filter for the `--molport-only` toggle.
- Generation validity is guaranteed by SELFIES, but novelty/synthesizability is not —
  use `--molport-only` or the xTB step to ground results.
