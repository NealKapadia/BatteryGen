"""Chemistry + data layer: canonicalization, RDKit descriptors, SELFIES vocab,
per-molecule processing (used by the multiprocessing preprocessor) and the torch
``Dataset`` over the preprocessed shards.

Importing this module pins the SELFIES semantic constraints so that encoding
(preprocessing) and decoding (generation) always agree.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from batterygen.core import config


# --------------------------------------------------------------------------- #
# SELFIES constraints (pinned, shared by encode + decode)
# --------------------------------------------------------------------------- #
import selfies as sf

# Default constraints guarantee every decoded molecule is RDKit-valid (the whole
# point of SELFIES). They already permit S=6 (sulfonyl) and P=5 (phosphate), so
# nothing is lost vs the old "hypervalent" preset — which additionally allowed
# pentavalent neutral nitrogen and made ~22% of prior samples RDKit-invalid.
# Switching to default took generation validity 0.77 -> 1.00 and recon 0.93 -> 0.98
# on the existing checkpoint (no retraining). See handoff §validity-fix.
CONSTRAINTS = "default"
sf.set_semantic_constraints()

# --------------------------------------------------------------------------- #
# RDKit (quiet)
# --------------------------------------------------------------------------- #
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


# --------------------------------------------------------------------------- #
# Canonicalization (the ONE definition used for training + membership)
# --------------------------------------------------------------------------- #
def canonical_smiles(smiles: str) -> Optional[str]:
    """Largest-fragment, RDKit-canonical isomeric SMILES, or None if unparseable."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if not frags:
            return None
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _atoms_ok(mol) -> bool:
    return all(a.GetSymbol() in config.ALLOWED_ATOMS for a in mol.GetAtoms())


def compute_descriptors(mol) -> List[float]:
    """RDKit descriptors in the fixed ``config.PROPERTIES`` order. NaNs -> 0."""
    out: List[float] = []
    for name in config.PROPERTIES:
        try:
            if name == "MolWt":
                v = Descriptors.MolWt(mol)
            elif name == "MolLogP":
                v = Crippen.MolLogP(mol)
            elif name == "TPSA":
                v = rdMolDescriptors.CalcTPSA(mol)
            elif name == "QED":
                v = QED.qed(mol)
            elif name == "NumHDonors":
                v = Lipinski.NumHDonors(mol)
            elif name == "NumHAcceptors":
                v = Lipinski.NumHAcceptors(mol)
            elif name == "NumRotatableBonds":
                v = Descriptors.NumRotatableBonds(mol)
            elif name == "NumAromaticRings":
                v = rdMolDescriptors.CalcNumAromaticRings(mol)
            elif name == "NumRings":
                v = rdMolDescriptors.CalcNumRings(mol)
            elif name == "FractionCSP3":
                v = rdMolDescriptors.CalcFractionCSP3(mol)
            elif name == "HeavyAtomCount":
                v = mol.GetNumHeavyAtoms()
            else:
                v = 0.0
        except Exception:
            v = 0.0
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            v = 0.0
        out.append(float(v))
    return out


def descriptors_for_smiles(smiles: str) -> Optional[List[float]]:
    canon = canonical_smiles(smiles)
    if canon is None:
        return None
    mol = Chem.MolFromSmiles(canon)
    return compute_descriptors(mol) if mol is not None else None


# --------------------------------------------------------------------------- #
# Vocabulary (fixed SELFIES robust alphabet + specials)
# --------------------------------------------------------------------------- #
class Vocab:
    def __init__(self, itos: List[str]):
        self.itos = list(itos)
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}
        self.pad = self.stoi[config.PAD]
        self.bos = self.stoi[config.BOS]
        self.eos = self.stoi[config.EOS]
        self.unk = self.stoi[config.UNK]

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls) -> "Vocab":
        symbols = sorted(sf.get_semantic_robust_alphabet())
        itos = list(config.SPECIAL_TOKENS) + symbols
        return cls(itos)

    def save(self, path: Path = config.VOCAB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"itos": self.itos, "constraints": CONSTRAINTS, "max_len": config.MAX_SELFIES_LEN},
                f,
            )

    @classmethod
    def load(cls, path: Path = config.VOCAB_PATH) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["itos"])

    # -- encode / decode -----------------------------------------------------
    def encode(self, selfies_str: str, max_len: int = config.MAX_SELFIES_LEN) -> Tuple[List[int], int]:
        """SELFIES string -> [BOS, ...symbols..., EOS, PAD...] of length max_len."""
        ids = [self.bos]
        for sym in sf.split_selfies(selfies_str):
            if len(ids) >= max_len - 1:  # leave room for EOS
                break
            ids.append(self.stoi.get(sym, self.unk))
        ids.append(self.eos)
        length = len(ids)
        if length < max_len:
            ids.extend([self.pad] * (max_len - length))
        return ids, length

    def decode(self, ids) -> str:
        """token ids -> SELFIES string (stops at EOS, drops specials)."""
        syms = []
        for i in ids:
            i = int(i)
            if i == self.eos:
                break
            if i in (self.pad, self.bos, self.unk):
                continue
            syms.append(self.itos[i])
        return "".join(syms)

    def decode_to_smiles(self, ids) -> Optional[str]:
        try:
            smi = sf.decoder(self.decode(ids))
        except Exception:
            return None
        return canonical_smiles(smi) if smi else None


# --------------------------------------------------------------------------- #
# Per-molecule processing for the preprocessor (runs inside worker processes)
# --------------------------------------------------------------------------- #
def process_record(smiles: str, molport_id: str, vocab: "Vocab"):
    """Full pipeline for one molecule. Returns (canon, molport_id, ids, length, desc)
    or None if the molecule is rejected by a filter."""
    canon = canonical_smiles(smiles)
    if canon is None:
        return None
    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        return None
    nheavy = mol.GetNumHeavyAtoms()
    if nheavy < config.MIN_HEAVY_ATOMS or nheavy > config.MAX_HEAVY_ATOMS:
        return None
    if not _atoms_ok(mol):
        return None
    try:
        s = sf.encoder(canon)
    except Exception:
        return None
    if not s:
        return None
    if sf.len_selfies(s) > config.MAX_SELFIES_LEN - 2:
        return None
    ids, length = vocab.encode(s)
    desc = compute_descriptors(mol)
    return canon, molport_id, ids, length, desc


# --------------------------------------------------------------------------- #
# Descriptor normalization stats
# --------------------------------------------------------------------------- #
def save_stats(mean: np.ndarray, std: np.ndarray, path: Path = config.DESC_STATS_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"properties": config.PROPERTIES, "mean": mean.tolist(), "std": std.tolist()}, f
        )


def load_stats(path: Path = config.DESC_STATS_PATH) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32)


def spec_to_condition(spec: Dict[str, float]) -> np.ndarray:
    """Map a {property: value} spec (raw units, human aliases ok) to a normalized
    condition vector of length N_PROPS. Unspecified properties -> 0 (the mean)."""
    mean, std = load_stats()
    cond = np.zeros(config.N_PROPS, np.float32)
    name_to_idx = {p: i for i, p in enumerate(config.PROPERTIES)}
    for key, val in spec.items():
        prop = config.PROPERTY_ALIASES.get(str(key).lower().strip(), key)
        if prop not in name_to_idx:
            continue
        # accept a scalar OR a [low, high] range (LLM specs) -> condition on the midpoint
        if isinstance(val, (list, tuple)):
            nums = [float(v) for v in val if v is not None]
            if not nums:
                continue
            val = sum(nums) / len(nums)
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        i = name_to_idx[prop]
        cond[i] = (val - mean[i]) / (std[i] if std[i] > 1e-8 else 1.0)
    return cond


# --------------------------------------------------------------------------- #
# Dataset over preprocessed shards
# --------------------------------------------------------------------------- #
def _load_meta() -> dict:
    with open(config.META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class MolDataset:
    """Lazy, memmap-backed dataset spanning all preprocessed shards.

    Returns (tokens LongTensor[max_len], length int, desc_norm FloatTensor[N_PROPS]).
    """

    def __init__(self, split: str = "train"):
        import torch  # local import so preprocessing doesn't need torch

        self.torch = torch
        meta = _load_meta()
        self.max_len = meta["max_len"]
        self.mean, self.std = load_stats()
        self.std = np.where(self.std > 1e-8, self.std, 1.0).astype(np.float32)

        self._tok, self._desc, self._len = [], [], []
        sizes = []
        for shard in meta["shards"]:
            base = config.PROC_DIR / shard
            self._tok.append(np.load(f"{base}_tokens.npy", mmap_mode="r"))
            self._desc.append(np.load(f"{base}_desc.npy", mmap_mode="r"))
            self._len.append(np.load(f"{base}_len.npy", mmap_mode="r"))
            sizes.append(self._tok[-1].shape[0])
        self.cum = np.cumsum([0] + sizes)
        total = int(self.cum[-1])

        # Prefer a held-out-scaffold split (make_split.py) if available; else a
        # deterministic random ~1% split.
        split_path = config.PROC_DIR / "split_val.npy"
        if split_path.exists():
            is_val = np.load(split_path)
            if len(is_val) != total:  # stale split -> ignore rather than misalign
                is_val = (np.arange(total) % 100) == 0
        else:
            is_val = (np.arange(total) % 100) == 0
        self.index = np.where(~is_val if split == "train" else is_val)[0]

    def __len__(self) -> int:
        return len(self.index)

    def _locate(self, gidx: int) -> Tuple[int, int]:
        shard = int(np.searchsorted(self.cum, gidx, side="right") - 1)
        return shard, gidx - int(self.cum[shard])

    def __getitem__(self, i: int):
        gidx = int(self.index[i])
        s, j = self._locate(gidx)
        tokens = np.asarray(self._tok[s][j], dtype=np.int64)
        length = int(self._len[s][j])
        desc = np.asarray(self._desc[s][j], dtype=np.float32)
        desc = (desc - self.mean) / self.std
        return (
            self.torch.from_numpy(tokens),
            length,
            self.torch.from_numpy(desc),
        )


def make_collate(pad_idx: int):
    """Collate that trims the batch to its longest sequence (speed)."""
    import torch

    def collate(batch):
        toks, lens, descs = zip(*batch)
        maxlen = max(lens)
        toks = torch.stack(toks)[:, :maxlen].contiguous()
        lens = torch.tensor(lens, dtype=torch.long)
        descs = torch.stack(descs)
        return toks, lens, descs

    return collate
