# MolForge

MolForge is an AI toolkit for **inventing new molecules** and **designing better battery
electrolyte additives**. It pairs a pretrained generative model (it dreams up novel, valid
molecules) with a predictive model (it estimates how well a molecule will perform) and ties
them together into an automated inverse-design loop.

Under the hood the generator is a *conditional variational autoencoder* trained on roughly
six million purchasable molecules from the Molport catalog, using
[SELFIES](https://github.com/aspuru-guzik-group/selfies) so that **almost every structure it
produces is chemically valid**. It learns a smooth map of chemical space that you can sample,
search, and steer toward properties you care about.

This guide is written so that someone with limited programming experience can get each part
running. Follow the steps in order.

---

## Table of contents

1. [What you can do](#1-what-you-can-do)
2. [What you need](#2-what-you-need)
3. [Setup, step by step](#3-setup-step-by-step)
4. [Workflow A: Generate molecules with the trained model](#4-workflow-a-generate-molecules-with-the-trained-model)
5. [Encode, decode, and score molecules](#5-encode-decode-and-score-molecules)
6. [Workflow B: Fine-tune the model on your own molecules](#6-workflow-b-fine-tune-the-model-on-your-own-molecules)
7. [Workflow C: Train the generative model from scratch](#7-workflow-c-train-the-generative-model-from-scratch)
8. [Workflow D: Train the predictive model with Optuna tuning](#8-workflow-d-train-the-predictive-model-with-optuna-tuning)
9. [Workflow E: Run the full inverse-design pipeline](#9-workflow-e-run-the-full-inverse-design-pipeline)
10. [Optional: enable natural-language requests (LLM keys)](#10-optional-enable-natural-language-requests-llm-keys)
11. [Optional: the web app](#11-optional-the-web-app)
12. [Other research and grounding tools](#12-other-research-and-grounding-tools)
13. [Project layout](#13-project-layout)
14. [Troubleshooting](#14-troubleshooting)
15. [License and attribution](#15-license-and-attribution)

---

## 1. What you can do

- **Generate** new, valid, novel molecules, optionally aimed at target properties (Workflow A).
- **Fine-tune** the trained model on your own molecules (Workflow B).
- **Train** a fresh generative model from scratch on your own data (Workflow C).
- **Train a predictive model**, with automatic Optuna hyperparameter tuning, to estimate a
  property such as Coulombic Efficiency (Workflow D).
- **Run the full inverse-design pipeline**: generate, predict, and rank candidate additives,
  optionally guided by a language model (Workflow E).

Everything runs on an ordinary laptop. A computer with an NVIDIA GPU is faster but not required.

---

## 2. What you need

- **Python 3.10 or newer.** Check with `python --version`.
- **About 1 GB of free disk space** for the model weights and dependencies.
- **(Optional) An NVIDIA GPU** for faster generation and training.
- **(Optional) The xTB program** ([install guide](https://xtb-docs.readthedocs.io/)) only if
  you use the quantum-chemistry features (full-accuracy CE design or DFT grounding).
- **(Optional) Language-model API keys** only for the natural-language features (Section 10).

The core generative features need none of the optional items.

---

## 3. Setup, step by step

### Step 1 — Install Python

If `python --version` is below 3.10, install it from
[python.org/downloads](https://www.python.org/downloads/). On Windows, check
"Add Python to PATH" during installation.

### Step 2 — Download the code

```bash
git clone https://github.com/NealKapadia/molforge.git
cd molforge
```

(No git? Use the green "Code" button on GitHub, then "Download ZIP", unzip, and open a
terminal in that folder.)

### Step 3 — (Recommended) Create a clean environment

```bash
python -m venv molforge-env
# Activate it:
#   Windows (PowerShell):  molforge-env\Scripts\Activate.ps1
#   macOS / Linux:         source molforge-env/bin/activate
```

### Step 4 — Install MolForge

```bash
pip install .            # core: generate, encode/decode, score, fine-tune
pip install ".[full]"    # everything: CE design, web app, training tools, reports
```

You can also install one capability at a time, for example `pip install ".[ce]"` (predictive
model) or `pip install ".[app]"` (web app). A familiar alternative is
`pip install -r requirements.txt`.

> **NVIDIA GPU users:** install the CUDA build of PyTorch *first*, then the command above:
> `pip install torch --index-url https://download.pytorch.org/whl/cu124`

### Step 5 — Get the model weights

The trained weights (about 0.5 GB) live on Hugging Face, separate from the code. The easiest
option is to let the code download them automatically on first use (Workflow A). To download
them yourself into a folder you control:

```python
from huggingface_hub import snapshot_download
print("Weights at:", snapshot_download("NealKapadia/Molforge"))
```

This folder holds `checkpoints/best.pt` (the model) and `processed/` (vocabulary and
normalization files). The code finds it automatically; you can also set the environment
variable `MOLVAE_ART_DIR` to point at it.

> **Tip for the training and CE workflows:** those write new files into the weights folder, so
> use a **writable copy** rather than the read-only download cache. Stage one once:
> ```bash
> python -c "from huggingface_hub import snapshot_download; import shutil; shutil.copytree(snapshot_download('NealKapadia/Molforge'),'artifacts',dirs_exist_ok=True)"
> ```
> then point at it:
> `export MOLVAE_ART_DIR="$PWD/artifacts"` (Windows: `$env:MOLVAE_ART_DIR = "$PWD\artifacts"`).

---

## 4. Workflow A: Generate molecules with the trained model

Create `try_it.py` and run it with `python try_it.py`:

```python
from huggingface_hub import snapshot_download
from molforge import MolForge

# Loads the model (downloads weights on first run). Use device="cuda" if you have a GPU.
mf = MolForge(device="cpu", artifacts_dir=snapshot_download("NealKapadia/Molforge"))

print(mf.generate(10))                              # 10 valid, novel molecules
print(mf.generate(5, spec={"MolWt": 300, "QED": 0.8}))   # aimed at target properties
for row in mf.generate(5, with_properties=True):    # molecules plus their properties
    print(row)
```

From the command line:

```bash
molforge --n 20
molforge --n 10 --spec '{"MolWt":250,"NumAromaticRings":1}'
molforge --n 10 --device cuda --out molecules.csv
```

Property targets are *soft*: the model moves toward them but does not hit them exactly. For
strict requirements, generate extra molecules and filter by their computed properties.

---

## 5. Encode, decode, and score molecules

```python
z = mf.encode("OCCN(CCO)CCO")   # molecule -> 256-number latent vector
mf.decode(z)                     # latent vector -> molecule
mf.properties("CCO")             # standard RDKit descriptors
mf.predict_properties("CCO")     # the model's own property predictions
```

To explore variations around a molecule, encode it, nudge the vector, and decode:

```python
import numpy as np
z = mf.encode("CC(=O)[O-]")
for _ in range(10):
    print(mf.decode(z + np.random.randn(*z.shape).astype("float32") * 0.6))
```

---

## 6. Workflow B: Fine-tune the model on your own molecules

Fine-tuning adapts the trained model toward your chemistry. It builds its training set
directly from a text file of SMILES, so it does **not** need the original training data.

1. Put your molecules in a file, one SMILES per line, e.g. `my_molecules.smi`.
2. Point `MOLVAE_ART_DIR` at a **writable** copy of the weights (see the tip in Step 5).
3. Run the fine-tuner:

```bash
python -m molforge.finetune --input my_molecules.smi --epochs 3          # add --device cuda for a GPU
python -m molforge.finetune --input solvents.csv --smiles-col 0 --header --delim ,   # CSV input
```

The result is saved as `artifacts/checkpoints/finetuned.pt`. Use it:

```python
from molforge import MolForge
mf = MolForge(device="cpu", ckpt="artifacts/checkpoints/finetuned.pt")
print(mf.generate(20))
```

A pure-CPU laptop handles a few thousand molecules and a few epochs comfortably (just more
slowly than a GPU). Your molecules are filtered by the same rules as the base model (3 to 60
heavy atoms, common organic elements); the run prints how many survived.

---

## 7. Workflow C: Train the generative model from scratch

Use this to build a brand-new model on your own dataset instead of adapting the existing one.
A GPU is strongly recommended; on CPU it is only practical for small datasets.

1. Choose a writable output folder for all artifacts:

   ```bash
   export MOLVAE_ART_DIR="$PWD/my_model"     # Windows: $env:MOLVAE_ART_DIR = "$PWD\my_model"
   ```

2. Convert your SMILES into training data (this also builds the vocabulary and the property
   normalization statistics):

   ```bash
   python -m molforge.add_data --input mydata.smi --tag mydata --recompute-stats
   ```

3. (Optional but recommended) Create an honest held-out validation split:

   ```bash
   python -m molforge.make_split
   ```

4. Train. The model checkpoints periodically and can be stopped and resumed at any time:

   ```bash
   python -m molforge.train --epochs 6 --batch 320          # remove/lower --batch on small GPUs
   python -m molforge.train --resume                        # continue a stopped run
   ```

   The best checkpoint is saved as `best.pt` in your artifacts folder.

5. Check quality against standard generative metrics:

   ```bash
   python -m molforge.evaluate
   ```

Your new model loads exactly like the pretrained one:
`MolForge(ckpt="my_model/checkpoints/best.pt")`.

---

## 8. Workflow D: Train the predictive model with Optuna tuning

The predictive model estimates a performance property (by default Coulombic Efficiency, CE)
from a molecule's structure. Training is three steps: build features, tune, train. It needs
the `ce` extra (`pip install ".[ce]"`), your labeled dataset (a CSV of molecules with measured
values), and, for the physics-based features, the xTB program.

1. Build the feature table from your dataset. This runs xTB once per molecule and caches the
   result, so re-runs are fast:

   ```bash
   python -m molforge.ce_features --csv Supplementary_Data_1.csv
   ```

2. Tune the model's hyperparameters automatically with **Optuna**. It searches many settings
   and keeps the best, evaluating on both random and scaffold splits for an honest estimate:

   ```bash
   python -m molforge.ce_tune
   ```

3. Train and save the final predictor using the tuned settings:

   ```bash
   python -m molforge.ce_train
   ```

   The trained model is written to `<artifacts>/ce/production_model.pkl`, and the run prints
   honest accuracy (R-squared) for both random and scaffold splits.

**Important limitation:** the predictor learns from measured data and cannot reliably predict
values *higher* than the best it has seen. In practice the ceiling for brand-new molecules is
around 97.6% CE, even though some known additives reach 98 to 99%. The dependable way past
that ceiling is to measure a promising candidate, add the measurement, and retrain:

```bash
python -m molforge.ce_features --append-smiles "<SMILES>" --append-ce 98.5
python -m molforge.ce_train
```

---

## 9. Workflow E: Run the full inverse-design pipeline

With the generative model (Workflow A or C) and the predictive model (Workflow D) in place,
this closes the loop: it generates many candidates, screens them with the predictor,
re-seeds generation around the best, refines the shortlist with full xTB physics, and writes
a ranked list of proposed additives.

```bash
python -m molforge.ce_design --n 1200 --rounds 3 --shortlist 60 --top 30 --out ce_candidates.csv
```

The output `ce_candidates.csv` ranks candidates by predicted performance minus an uncertainty
penalty, with their properties and (where available) their Molport catalog IDs.

To guide the search with plain English and add a language-model review of each candidate (this
needs the keys from Section 10):

```bash
python -m molforge.ce_design --prompt "water-stable zinc additive, molecular weight under 200" --llm
```

Add `--rag` to ground the explanation of the top candidate in a literature knowledge base you
provide. See `python -m molforge.ce_design --help` for all options.

---

## 10. Optional: enable natural-language requests (LLM keys)

Some features can turn a plain-English request into property targets and write mechanistic
explanations, using a language model. This is optional and off by default; without keys these
features fall back to simple keyword matching and everything else works unchanged.

1. Copy the example configuration:

   ```bash
   cp .env.example .env       # Windows: copy .env.example .env
   ```

2. Open `.env` in a text editor and fill in your keys. The file supports Azure AI Foundry and
   Azure OpenAI endpoints (used as the reasoning, fast, and judge models). Example fields:

   ```
   FOUNDRY_API_KEY=your-key-here
   FOUNDRY_ENDPOINT=https://your-resource.services.ai.azure.com/openai/v1
   AZURE_OPENAI_KEY=your-key-here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/openai/v1
   ```

3. Save the file. The language-model features (for example `ce_design --llm`, and the web app's
   "explain" button) will now use it automatically.

**Never share or commit your `.env` file.** It is excluded from the repository on purpose.

---

## 11. Optional: the web app

A point-and-click interface for inverse design, running entirely on your own machine:

```bash
pip install ".[app]"
uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in your browser. (A lighter, zero-dependency local version
is available with `python app.py`.)

---

## 12. Other research and grounding tools

Optional tools that extend the pipeline. Run them as modules
(for example `python -m molforge.qm9 --help`).

| Command | Purpose |
|---|---|
| `search.py` | Latent-space property search and optimization. |
| `report.py` | Build an HTML report (training curves, benchmark, latent map). |
| `xtb_label.py` | Compute quantum (xTB) electronic properties for molecules. |
| `qm9.py` / `finetune_dft.py` | Add real-DFT (QM9) property grounding to the model. |
| `electrolyte.py` | Train and screen a formulation-level conductivity model. |
| `pipeline.py` | Run the multi-stage generative-model specialization pipeline. |

---

## 13. Project layout

**Core library**

| File | Role |
|---|---|
| `molforge.py` | The `MolForge` class: generate, encode, decode, score. |
| `finetune.py` | Fine-tune the model on your own molecules. |
| `config.py` | All paths, settings, and the property list. |
| `data.py` | Chemistry: canonicalization, descriptors, the SELFIES vocabulary. |
| `model.py` | The neural network (encoder, decoder, property head). |
| `infer.py` | Loads a trained checkpoint. |

**Predictive model and battery design:** `ce_features.py`, `ce_tune.py`, `ce_train.py`,
`ce_design.py`, `ce_model.py`, `ce_llm.py`, `ce_rag.py`, `electrolyte.py`,
`electrolyte_data.py`, `solvent_lib.py`

**Generative training pipeline:** `preprocess.py`, `add_data.py`, `train.py`, `make_split.py`,
`membership.py`, `generate.py`, `search.py`, `evaluate.py`, `report.py`, `pipeline.py`

**Quantum grounding:** `xtb_label.py`, `qm9.py`, `finetune_dft.py`, `openqdc_data.py`, `hf_data.py`

**Web app and language model:** `server.py`, `app.py`, `webengine.py`, `design.py`, `llm.py`, `static/`

**Tests:** `tests/` (run with `python -m pytest tests/`)

---

## 14. Troubleshooting

- **`ModuleNotFoundError` when running `python -m molforge....`** — make sure you ran
  `pip install .`, and run from a directory *other than* the source folder itself. Inside the
  source folder the local `molforge.py` file shadows the installed package; there, run the
  scripts directly instead, e.g. `python finetune.py ...`.
- **"weights not found"** — set `MOLVAE_ART_DIR` to the folder where the weights were
  downloaded (Step 5), or pass `artifacts_dir=...` to `MolForge(...)`.
- **Training or CE design cannot write its output** — point `MOLVAE_ART_DIR` at a *writable*
  copy of the weights, not the read-only download cache (see the tip in Step 5).
- **Out of memory on a GPU** — lower the batch size, e.g. `--batch 32`.
- **xTB errors in the CE workflows** — confirm xTB is installed and set the `XTB_EXE`
  environment variable to its executable, or run feature building with `--no-xtb` to skip it.
- **It is slow on CPU** — expected for large jobs. Use a smaller count, or a GPU.

---

## 15. License and attribution

Both the **code and the model weights** are released under **CC BY-NC 4.0** (Attribution,
NonCommercial) — see the [LICENSE](LICENSE) file. The weights are derived from the Molport
"All Stock" catalog, which is itself CC BY-NC 4.0, so the same terms apply: you must credit
MolForge and Molport, and you may not use the project for commercial purposes. See the
[Hugging Face model card](https://huggingface.co/NealKapadia/Molforge) for details.

If you use MolForge in your work, please cite this repository and the SELFIES paper
(Krenn et al., 2020).
