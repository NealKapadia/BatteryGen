"""Central configuration for the Molport SELFIES-VAE pipeline.

Every path, hyper-parameter and the canonical property list lives here so the rest
of the code stays declarative. Override most things with environment variables
(see ``.env.example``) or by editing this file.

Running ``python molvae/<script>.py`` puts this folder on ``sys.path`` so the
sibling modules can ``import config`` directly.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Dataset root = the folder that contains SMILES/ and the *.sdf.gz chunks.
# config.py lives in <DATA_DIR>/molvae/, so DATA_DIR is two parents up.
DATA_DIR = Path(os.getenv("MOLVAE_DATA_DIR", str(Path(__file__).resolve().parent.parent)))
SMILES_DIR = DATA_DIR / "SMILES"
SMILES_GLOB = "iis_smiles-*.txt.gz"

# All generated artifacts live here, separate from raw data + code.
ART_DIR = Path(os.getenv("MOLVAE_ART_DIR", str(DATA_DIR / "molvae_artifacts")))
PROC_DIR = ART_DIR / "processed"      # token shards, vocab, descriptors, meta
CKPT_DIR = ART_DIR / "checkpoints"    # training checkpoints
MEMBER_DIR = ART_DIR / "membership"   # Molport membership index
DFT_DIR = ART_DIR / "dft"             # xTB labels + fine-tune data
SAMPLE_DIR = ART_DIR / "samples"      # generated molecules


def ensure_dirs() -> None:
    for d in (ART_DIR, PROC_DIR, CKPT_DIR, MEMBER_DIR, DFT_DIR, SAMPLE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Processed-data artifacts
VOCAB_PATH = PROC_DIR / "vocab.json"
META_PATH = PROC_DIR / "meta.json"
TOKENS_PATH = PROC_DIR / "tokens.npy"            # int16 [N, max_len]
LENGTHS_PATH = PROC_DIR / "lengths.npy"          # int16 [N]  (incl. BOS/EOS)
DESC_PATH = PROC_DIR / "descriptors.npy"         # float32 [N, N_PROPS]
DESC_STATS_PATH = PROC_DIR / "descriptor_stats.json"

# Molport membership index
MOLPORT_DB = MEMBER_DIR / "molport.sqlite"
MOLPORT_BLOOM = MEMBER_DIR / "molport.bloom"

# xTB binary (already installed on this machine). Override with $XTB_EXE.
XTB_EXE = os.getenv(
    "XTB_EXE",
    r"C:\Users\nkapa\Downloads\xtb-6.7.1pre-windows-x86_64\xtb-6.7.1\bin\xtb.exe",
)

# --------------------------------------------------------------------------- #
# User-supplied datasets (e.g. the CE / predictive-model training CSV)
# --------------------------------------------------------------------------- #
# Put your CSV(s) in a "data/" folder where you run the commands and the CE tools
# find them automatically. Override the folder with $MOLVAE_USER_DATA_DIR, point at
# a specific file with $MOLVAE_CE_CSV, or pass --csv (highest priority).
USER_DATA_DIR = Path(os.getenv("MOLVAE_USER_DATA_DIR", "data"))


def resolve_ce_csv(explicit: str | None = None) -> Path:
    """Locate the CE dataset CSV robustly. Priority:

       1. an explicit path (e.g. the ``--csv`` flag)
       2. the ``$MOLVAE_CE_CSV`` environment variable
       3. a single ``*.csv`` inside ``USER_DATA_DIR`` (the ``data/`` folder)

    Raises ``SystemExit`` with guidance if nothing is found or the choice is ambiguous.
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"CE dataset not found: {p}")
        return p

    env = os.getenv("MOLVAE_CE_CSV")
    if env:
        p = Path(env)
        if not p.exists():
            raise SystemExit(f"$MOLVAE_CE_CSV points to a missing file: {p}")
        return p

    if USER_DATA_DIR.is_dir():
        csvs = sorted(USER_DATA_DIR.glob("*.csv"))
        if len(csvs) == 1:
            return csvs[0]
        if len(csvs) > 1:
            names = ", ".join(c.name for c in csvs)
            raise SystemExit(
                f"Multiple CSV files in {USER_DATA_DIR.resolve()} ({names}).\n"
                f"Choose one with  --csv <file>  or set $MOLVAE_CE_CSV."
            )

    raise SystemExit(
        "No CE dataset found. Do one of:\n"
        f"  - place a single .csv in a '{USER_DATA_DIR}' folder ({USER_DATA_DIR.resolve()})\n"
        "  - pass  --csv path/to/your.csv\n"
        "  - set the $MOLVAE_CE_CSV environment variable\n"
        "The CSV needs a SMILES column and a measured-CE column (see README, Workflow D)."
    )


def resolve_ce_csv_optional(explicit: str | None = None) -> Path | None:
    """Like :func:`resolve_ce_csv` but returns ``None`` when no dataset is configured.

    Still raises for a *named* file that is missing or an ambiguous ``data/`` folder, so
    user mistakes are not silently ignored; only the "nothing configured" case is soft.
    """
    if explicit or os.getenv("MOLVAE_CE_CSV"):
        return resolve_ce_csv(explicit)
    if USER_DATA_DIR.is_dir() and sorted(USER_DATA_DIR.glob("*.csv")):
        return resolve_ce_csv(None)  # one -> path, many -> ambiguity error
    return None

# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
SEED = int(os.getenv("MOLVAE_SEED", "42"))


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Properties used for conditioning + latent prediction (RDKit descriptors).
# The ORDER is fixed — it defines the conditioning-vector layout everywhere.
# --------------------------------------------------------------------------- #
PROPERTIES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "QED",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumRings",
    "FractionCSP3",
    "HeavyAtomCount",
]
N_PROPS = len(PROPERTIES)

# Human-friendly aliases accepted from the LLM / CLI when specifying a target.
PROPERTY_ALIASES = {
    "mw": "MolWt", "molwt": "MolWt", "molecular_weight": "MolWt", "weight": "MolWt",
    "logp": "MolLogP", "clogp": "MolLogP", "lipophilicity": "MolLogP",
    "tpsa": "TPSA", "psa": "TPSA",
    "qed": "QED", "druglikeness": "QED", "drug_likeness": "QED",
    "hbd": "NumHDonors", "donors": "NumHDonors", "h_donors": "NumHDonors",
    "hba": "NumHAcceptors", "acceptors": "NumHAcceptors", "h_acceptors": "NumHAcceptors",
    "rotatable": "NumRotatableBonds", "rotbonds": "NumRotatableBonds",
    "aromatic_rings": "NumAromaticRings", "aromatics": "NumAromaticRings",
    "rings": "NumRings", "ring_count": "NumRings",
    "fcsp3": "FractionCSP3", "fraction_csp3": "FractionCSP3", "sp3": "FractionCSP3",
    "heavy_atoms": "HeavyAtomCount", "heavy": "HeavyAtomCount", "size": "HeavyAtomCount",
}

# xTB / DFT-surrogate targets, added during fine-tuning.
#   homo/lumo/gap (eV), dipole (Debye), total_energy (Eh),
#   ip = vertical ionization potential (eV)  -> oxidative-stability proxy,
#   ea = vertical electron affinity (eV)      -> reductive-stability proxy,
#   window = ip - ea (eV)                     -> electrochemical-window proxy.
# (CE / overpotential / cycle-life are system/experimental properties — not computable
#  from one structure; supply them as electrolyte.py target columns to predict them.)
DFT_PROPERTIES = ["homo", "lumo", "gap", "dipole", "total_energy", "ip", "ea", "window"]

# --------------------------------------------------------------------------- #
# Preprocessing filters
# --------------------------------------------------------------------------- #
ALLOWED_ATOMS = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "H"}
MIN_HEAVY_ATOMS = 3
MAX_HEAVY_ATOMS = 60       # drug-like cap; bounds SELFIES length + reconstruction
MAX_SELFIES_LEN = 120      # hard token cap (incl. BOS/EOS); re-derived from data
N_WORKERS = int(os.getenv("MOLVAE_WORKERS", str(max(1, (os.cpu_count() or 4) - 1))))

# --------------------------------------------------------------------------- #
# Model hyper-parameters.
# "big" profile (default) is sized to use the RTX 3060 6 GB properly (~45M params)
# for an 8-10 h, higher-quality run. Set MOLVAE_PROFILE=small for the fast 4M model.
# --------------------------------------------------------------------------- #
_PROFILE = os.getenv("MOLVAE_PROFILE", "big").lower()
if _PROFILE == "small":
    EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, LATENT_DIM, ENC_LAYERS, DEC_LAYERS = 256, 512, 512, 128, 1, 1
else:  # "big"
    EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, LATENT_DIM, ENC_LAYERS, DEC_LAYERS = 512, 1024, 1024, 256, 2, 2
EMB_DIM = int(os.getenv("MOLVAE_EMB", EMB_DIM))
ENC_HIDDEN = int(os.getenv("MOLVAE_ENC_HIDDEN", ENC_HIDDEN))
DEC_HIDDEN = int(os.getenv("MOLVAE_DEC_HIDDEN", DEC_HIDDEN))
LATENT_DIM = int(os.getenv("MOLVAE_LATENT", LATENT_DIM))
ENC_LAYERS = int(os.getenv("MOLVAE_ENC_LAYERS", ENC_LAYERS))
DEC_LAYERS = int(os.getenv("MOLVAE_DEC_LAYERS", DEC_LAYERS))
DROPOUT = float(os.getenv("MOLVAE_DROPOUT", "0.2"))
# Decoder word/token dropout (Bowman et al. 2016): randomly replace teacher-forcing
# input tokens with <unk> so the decoder must rely on z -> a far more meaningful latent.
WORD_DROPOUT = float(os.getenv("MOLVAE_WORD_DROPOUT", "0.25"))
PROP_HEAD_HIDDEN = 256

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
# Batch 320 measured as the throughput/VRAM sweet spot for the 42M model on a
# 6 GB RTX 3060 Laptop: ~912 mol/s @ 3.14 GB. Going >=384 risks tipping over the
# VRAM-spill cliff (WDDM falls back to system RAM -> ~4x slowdown). Don't raise
# blindly — measure peak VRAM stays under ~3.5 GB first.
BATCH_SIZE = int(os.getenv("MOLVAE_BATCH", "320"))
LR = float(os.getenv("MOLVAE_LR", "5e-4"))
LR_MIN = float(os.getenv("MOLVAE_LR_MIN", "2e-5"))
LR_WARMUP_MOLECULES = int(os.getenv("MOLVAE_LR_WARMUP", "1000000"))
GRAD_CLIP = 5.0
# This model converges in ~2-4 epochs over 6M molecules; 6 + early-stopping (--patience 2)
# is plenty. More epochs were observed to over-train/destabilize, not improve.
EPOCHS = int(os.getenv("MOLVAE_EPOCHS", "6"))
CHECKPOINT_EVERY = int(os.getenv("MOLVAE_CKPT_EVERY", "500000"))  # molecules seen
VAL_FRACTION = 0.01
# DataLoader workers. Default 0 on Windows (spawn overhead; data is already an
# in-RAM/memmap int array so workers buy little).
NUM_WORKERS = int(os.getenv("MOLVAE_NUM_WORKERS", "0"))
USE_AMP = os.getenv("MOLVAE_AMP", "1") != "0"

# KL schedule: beta ramps 0->max over the first KL_RATIO of each of KL_CYCLES cycles,
# then holds. NOTE: a 12-epoch run with 4 cycles + max 0.15 was observed to DESTABILIZE
# an already-converged model (recon climbed, val acc fell after ~epoch 4). Defaults are
# now a single monotonic warm-up at a gentler max — stable for a model that converges
# in 2-4 epochs. Use train.py --patience 2 + best.pt to auto-catch any divergence.
KL_WEIGHT_MAX = float(os.getenv("MOLVAE_KL_MAX", "0.1"))
KL_CYCLES = int(os.getenv("MOLVAE_KL_CYCLES", "1"))
KL_RATIO = float(os.getenv("MOLVAE_KL_RATIO", "0.4"))
KL_ANNEAL_MOLECULES = int(os.getenv("MOLVAE_KL_ANNEAL", "2000000"))  # used only in small profile
PROP_LOSS_WEIGHT = float(os.getenv("MOLVAE_PROP_W", "1.0"))
FREE_BITS = float(os.getenv("MOLVAE_FREE_BITS", "0.03"))  # nats/dim, anti-collapse

# Special tokens (indices assigned in this order when the vocab is built).
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
