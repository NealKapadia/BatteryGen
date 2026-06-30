"""Core regression tests for the MolForge library.

Goal: prove the public path (import -> build model -> generate/encode/decode/score ->
fine-tune) keeps working through refactors. The weight-dependent tests are gated so the
suite is green on a fresh clone too.

Run:  python -m pytest tests/ -q          (from the molvae/ folder)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import requires_weights, requires_stats

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Lightweight tests (no weights needed)
# --------------------------------------------------------------------------- #
def test_modules_import():
    import config, data, model, infer, finetune, molforge  # noqa: F401
    from molforge import MolForge  # noqa: F401


def test_vocab_roundtrip(vocab):
    import selfies as sf
    assert len(vocab) > 4  # specials + alphabet
    sel = sf.encoder("CCO")
    ids, length = vocab.encode(sel)
    assert ids[0] == vocab.bos
    assert length <= len(ids)
    assert vocab.decode_to_smiles(ids) == "CCO"


def test_canonical_smiles():
    import data
    assert data.canonical_smiles("OCC") == "CCO"          # canonicalized
    assert data.canonical_smiles("c1ccccc1") is not None   # aromatic
    assert data.canonical_smiles("this is not a molecule") is None


def test_descriptors():
    import config, data
    vals = data.descriptors_for_smiles("CCO")
    assert vals is not None and len(vals) == config.N_PROPS
    mw = vals[config.PROPERTIES.index("MolWt")]
    assert 40 < mw < 60  # ethanol ~46


def test_model_forward_sample_and_loss(vocab):
    import model as M
    torch.manual_seed(0)
    net = M.SelfiesVAE(
        vocab_size=len(vocab), pad_idx=vocab.pad, unk_idx=vocab.unk,
        emb_dim=32, enc_hidden=32, dec_hidden=32, latent_dim=16,
        enc_layers=1, dec_layers=1, n_props=11,
    )
    net.train()
    B, T = 4, 12
    toks = torch.randint(4, len(vocab), (B, T))
    toks[:, 0] = vocab.bos
    lens = torch.full((B,), T, dtype=torch.long)
    cond = torch.zeros(B, 11)

    logits, mu, logvar, prop = net(toks, lens, cond)
    assert logits.shape == (B, T - 1, len(vocab))
    assert mu.shape == (B, 16)

    total, comps = M.vae_loss(logits, toks[:, 1:], mu, logvar, prop, cond, vocab.pad)
    assert torch.isfinite(total)
    assert {"recon", "kl", "prop"} <= set(comps)

    seqs = net.sample(20, torch.zeros(20, 11), vocab.bos, vocab.eos, device="cpu")
    assert seqs.shape[0] == 20
    valid = [vocab.decode_to_smiles(s.tolist()) for s in seqs]
    assert any(v for v in valid)  # SELFIES decoding yields valid molecules


def test_finetune_build_dataset(vocab):
    import config, finetune
    toks, lens, desc = finetune.build_dataset(["CCO", "c1ccccc1", "CC(=O)O"], vocab)
    assert toks.shape[0] == 3
    assert lens.shape[0] == 3
    assert desc.shape == (3, config.N_PROPS)


def test_finetune_rejects_junk(vocab):
    import finetune
    with pytest.raises(SystemExit):
        finetune.build_dataset(["not_a_molecule", "@@@@"], vocab)


def test_resolve_ce_csv(tmp_path, monkeypatch):
    import config
    # explicit path that exists -> returned; missing -> error
    f = tmp_path / "my.csv"
    f.write_text("Additive_SMILES,CE_aver. (%)\nCCO,99\n")
    assert config.resolve_ce_csv(str(f)) == f
    with pytest.raises(SystemExit):
        config.resolve_ce_csv(str(tmp_path / "nope.csv"))

    # environment variable
    monkeypatch.setenv("MOLVAE_CE_CSV", str(f))
    assert config.resolve_ce_csv(None) == f
    monkeypatch.delenv("MOLVAE_CE_CSV", raising=False)

    # data/ folder: single CSV -> auto; multiple -> error; none -> error
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setattr(config, "USER_DATA_DIR", d)
    with pytest.raises(SystemExit):           # empty folder
        config.resolve_ce_csv(None)
    assert config.resolve_ce_csv_optional(None) is None
    (d / "a.csv").write_text("x")
    assert config.resolve_ce_csv(None) == d / "a.csv"
    (d / "b.csv").write_text("x")
    with pytest.raises(SystemExit):           # ambiguous
        config.resolve_ce_csv(None)


@requires_stats
def test_spec_to_condition():
    import config, data
    cond = data.spec_to_condition({"MolWt": 300, "qed": 0.8})  # alias 'qed' accepted
    assert cond.shape == (config.N_PROPS,)
    assert np.isfinite(cond).all()


# --------------------------------------------------------------------------- #
# Weight-dependent tests (skipped unless artifacts are present)
# --------------------------------------------------------------------------- #
@requires_weights
def test_molforge_generate():
    from rdkit import Chem
    from molforge import MolForge
    mf = MolForge(device="cpu")
    out = mf.generate(3)
    assert len(out) == 3
    assert all(Chem.MolFromSmiles(s) is not None for s in out)


@requires_weights
def test_molforge_encode_decode_and_props():
    from molforge import MolForge
    mf = MolForge(device="cpu")
    z = mf.encode("OCCN(CCO)CCO")
    assert z is not None and z.ndim == 1
    smi = mf.decode(z)
    assert isinstance(smi, str) and smi
    props = mf.properties("CCO")
    assert props and "MolWt" in props


@requires_weights
def test_finetune_cli_end_to_end(tmp_path):
    """Fine-tune for 1 epoch on a tiny file and confirm the checkpoint reloads."""
    import infer
    smi = tmp_path / "mini.smi"
    smi.write_text("\n".join(["CCO", "OCCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC",
                              "OCCN(CCO)CCO", "CCOC(C)=O", "O=C1OCCO1"]) + "\n")
    out = tmp_path / "ft.pt"
    r = subprocess.run(
        [sys.executable, "finetune.py", "--input", str(smi),
         "--epochs", "1", "--batch", "4", "--device", "cpu", "--out", str(out)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.exists()
    net, vocab, ck, device = infer.load_model(str(out), device="cpu")
    assert "hparams" in ck
