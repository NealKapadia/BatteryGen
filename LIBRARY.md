# MolForge — run the generative model on your own machine

A small Python API over the pretrained SELFIES-VAE. Generate, encode/decode, and score
molecules on your own **CPU or GPU** — no server, no Azure, no API keys. ~100% valid SMILES.

## Install

```bash
# 1. get the code
git clone <your-repo-url> molforge && cd molforge/molvae       # the `molvae/` folder

# 2. install the package  (CPU torch by default; for a GPU install torch from pytorch.org first)
pip install .                 # editable dev install: pip install -e .
#   extras:  pip install ".[app]"   (local REST API)   ".[llm]"   ".[ce]"   ".[full]"

# 3. get the model weights + vocab (artifacts are ~0.5 GB; host best.pt on Hugging Face)
#    Layout expected under an artifacts dir:
#      artifacts/processed/{vocab.json, descriptor_stats.json, meta.json}
#      artifacts/checkpoints/best.pt
export MOLVAE_ART_DIR=/path/to/artifacts      # Windows: $env:MOLVAE_ART_DIR="..."
```

After `pip install .` the `molforge` console command and `from molforge import MolForge`
work from anywhere. (`requirements-lib.txt` still exists for a no-package, run-in-place setup.)

## Use it (Python)

```python
from molforge import MolForge

mf = MolForge(device="cpu")            # or device="cuda"
mf.generate(10)                        # 10 valid, novel SMILES
mf.generate(5, spec={"MolWt": 300, "QED": 0.8})        # property-targeted
mf.generate(5, with_properties=True)   # [{smiles, MolWt, MolLogP, ...}]

z = mf.encode("OCCN(CCO)CCO")          # 256-d latent
mf.decode(z)                           # back to a SMILES
mf.properties("CCO")                   # RDKit descriptors
mf.predict_properties("CCO")           # the VAE's own property-head prediction
```

## Use it (command line)

```bash
python molforge.py --n 20
python molforge.py --n 10 --spec '{"MolWt":250,"NumAromaticRings":1}'
python molforge.py --n 10 --device cuda --out molecules.csv
```

## Fine-tune it on your own molecules

`finetune.py` adapts the pretrained model to your own chemistry. It builds its training
set **directly from a SMILES file you supply** — it does *not* need the original ~6M Molport
token shards, so it runs from this `pip install` + the Hugging Face artifacts alone (CPU or GPU):

```bash
# MOLVAE_ART_DIR must point at a WRITABLE copy of the downloaded artifacts
python -m molforge.finetune --input my_molecules.smi --epochs 3 --device cuda
python -m molforge.finetune --input solvents.csv --smiles-col 0 --header --delim ,   # CSV
```

It writes `<artifacts>/checkpoints/finetuned.pt` (compatible with `MolForge`):

```python
from molforge import MolForge
mf = MolForge(device="cpu", ckpt="/path/to/artifacts/checkpoints/finetuned.pt")
mf.generate(20)
```

- **Runs on any laptop.** CPU is fine for a few thousand molecules / a few epochs (just slow);
  pass `--device cuda` if you have an NVIDIA GPU. It's a 42M-param model, so a GPU is faster.
- Molecules are filtered by the same rules as the base model (3–60 heavy atoms, organic
  elements, SELFIES length cap); the run prints how many survived.
- Descriptor normalization is reused from the base model so conditioning stays calibrated.

## Want a local REST API instead of importing?

The same engine powers a FastAPI server you can run on your own machine:

```bash
pip install fastapi "uvicorn[standard]" python-multipart pandas scikit-learn joblib
MOLVAE_DEVICE=cuda uvicorn server:app --host 0.0.0.0 --port 8000
# then POST to http://localhost:8000/generate , /train , /molecule , ...
```

## Notes
- **Library vs hosted API:** to run on *your own* hardware, use this library (fast on your GPU,
  free, reproducible). A hosted API runs on the *provider's* server — that's the public web app,
  meant for zero-install browser use.
- Conditional `spec` targeting is *soft* (the model nudges toward targets, not exact); over-
  generate and filter for hard constraints.
- `pip install .` works now (see above). Two small follow-ups remain for a one-liner public
  install: (1) publish to PyPI so `pip install molforge` works without cloning, and
  (2) auto-download weights from Hugging Face (via `huggingface_hub`) so step 3 is automatic.
