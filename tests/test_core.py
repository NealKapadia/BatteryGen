"""Core regression tests for the BatteryGen library.

Goal: prove the public path (import → build model → generate/encode/decode/score →
fine-tune) and the config-driven predictive pipeline keep working through refactors. The
weight- and dependency-gated tests are skipped so the suite is green on a fresh clone too.

Run:  python -m pytest tests/ -q          (from the batterygen/ folder)
"""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import requires_weights, requires_stats

ROOT = Path(__file__).resolve().parent.parent          # .../batterygen
PARENT = ROOT.parent                                    # dir where `batterygen` is importable


def _have(*mods) -> bool:
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception:
            return False
    return True


# --------------------------------------------------------------------------- #
# Lightweight tests (no weights needed)
# --------------------------------------------------------------------------- #
def test_public_api_import():
    from batterygen.core import config, data, model, infer, finetune, api  # noqa: F401
    from batterygen import BatteryGen  # noqa: F401


# Every module, fully qualified, to prove the package-absolute reorg didn't break a
# cross-module import. Optional third-party deps (pandas, xgboost, ...) are skipped.
_SUBMODULES = {
    "core": ["config", "data", "model", "infer", "api", "finetune", "membership", "llm"],
    "generative": ["preprocess", "train", "add_data", "make_split", "generate", "search",
                   "evaluate", "report", "pipeline"],
    "grounding": ["xtb_label", "qm9", "finetune_dft", "openqdc_data", "hf_data"],
    "electrolyte": ["electrolyte", "electrolyte_data", "solvent_lib"],
    "predictive": ["target", "transform", "paths", "features", "select", "tune", "train",
                   "design", "design_llm", "rag", "sampling"],
}
LOCAL_MODULES = sorted(f"batterygen.{sub}.{m}" for sub, mods in _SUBMODULES.items() for m in mods)


@pytest.mark.parametrize("mod", LOCAL_MODULES)
def test_local_module_imports(mod):
    try:
        importlib.import_module(mod)
    except ModuleNotFoundError as e:
        missing = (e.name or "").split(".")[0]
        if missing == "batterygen":
            raise  # a genuine cross-module breakage from the reorg
        pytest.skip(f"optional dependency '{missing}' not installed")


def test_vocab_roundtrip(vocab):
    import selfies as sf
    assert len(vocab) > 4
    sel = sf.encoder("CCO")
    ids, length = vocab.encode(sel)
    assert ids[0] == vocab.bos
    assert length <= len(ids)
    assert vocab.decode_to_smiles(ids) == "CCO"


def test_canonical_smiles():
    from batterygen.core import data
    assert data.canonical_smiles("OCC") == "CCO"
    assert data.canonical_smiles("c1ccccc1") is not None
    assert data.canonical_smiles("this is not a molecule") is None


def test_descriptors():
    from batterygen.core import config, data
    vals = data.descriptors_for_smiles("CCO")
    assert vals is not None and len(vals) == config.N_PROPS
    mw = vals[config.PROPERTIES.index("MolWt")]
    assert 40 < mw < 60


def test_model_forward_sample_and_loss(vocab):
    from batterygen.core import model as M
    torch.manual_seed(0)
    net = M.SelfiesVAE(
        vocab_size=len(vocab), pad_idx=vocab.pad, unk_idx=vocab.unk,
        emb_dim=32, enc_hidden=32, dec_hidden=32, latent_dim=16,
        enc_layers=1, dec_layers=1, n_props=11,
    )
    net.train()
    B, T = 4, 12
    toks = torch.randint(4, len(vocab), (B, T)); toks[:, 0] = vocab.bos
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
    assert any(vocab.decode_to_smiles(s.tolist()) for s in seqs)


def test_finetune_build_dataset(vocab):
    from batterygen.core import config, finetune
    toks, lens, desc = finetune.build_dataset(["CCO", "c1ccccc1", "CC(=O)O"], vocab)
    assert toks.shape[0] == 3 and lens.shape[0] == 3
    assert desc.shape == (3, config.N_PROPS)


def test_finetune_rejects_junk(vocab):
    from batterygen.core import finetune
    with pytest.raises(SystemExit):
        finetune.build_dataset(["not_a_molecule", "@@@@"], vocab)


def test_resolve_ce_csv(tmp_path, monkeypatch):
    from batterygen.core import config
    f = tmp_path / "my.csv"
    f.write_text("Additive_SMILES,CE_aver. (%)\nCCO,99\n")
    assert config.resolve_ce_csv(str(f)) == f
    with pytest.raises(SystemExit):
        config.resolve_ce_csv(str(tmp_path / "nope.csv"))
    monkeypatch.setenv("BATTERYGEN_CE_CSV", str(f))
    assert config.resolve_ce_csv(None) == f
    monkeypatch.delenv("BATTERYGEN_CE_CSV", raising=False)
    d = tmp_path / "data"; d.mkdir()
    monkeypatch.setattr(config, "USER_DATA_DIR", d)
    with pytest.raises(SystemExit):
        config.resolve_ce_csv(None)
    assert config.resolve_ce_csv_optional(None) is None
    (d / "a.csv").write_text("x")
    assert config.resolve_ce_csv(None) == d / "a.csv"
    (d / "b.csv").write_text("x")
    with pytest.raises(SystemExit):
        config.resolve_ce_csv(None)


# --------------------------------------------------------------------------- #
# Config-driven predictive pipeline (target.py generalization)
# --------------------------------------------------------------------------- #
def test_target_presets_switch_framing():
    from batterygen.predictive.target import PRESETS
    assert {"Zn", "Li", "Na", "K"} <= set(PRESETS)
    assert PRESETS["Li"].cation == "Li+" and "lithium" in PRESETS["Li"].system
    assert PRESETS["Na"].cation == "Na+" and PRESETS["K"].cation == "K+"
    # switching a preset changes only the framing, not the schema knobs
    assert PRESETS["Li"].smiles_col == PRESETS["Zn"].smiles_col


def test_feature_order_responds_to_config():
    from batterygen.predictive import features
    from batterygen.predictive.target import TARGET
    with_xtb = features.feature_order(True)
    without = features.feature_order(False)
    assert all(c in with_xtb for c in features.XTB_COLS)
    assert not any(c in without for c in features.XTB_COLS)
    for c in TARGET.context_cols:                       # context columns always present
        assert c in with_xtb and c in without


def test_transform_roundtrip(monkeypatch):
    import batterygen.predictive.transform as T
    from batterygen.predictive import target
    monkeypatch.setattr(target.TARGET, "target_transform", "lce")
    assert np.allclose(T.inverse(T.forward(np.array([90.0, 99.0]))), [90.0, 99.0], atol=1e-6)
    monkeypatch.setattr(target.TARGET, "target_transform", "none")
    assert np.allclose(T.inverse(T.forward(np.array([1.5, -2.0]))), [1.5, -2.0])


@pytest.mark.skipif(not _have("pandas", "sklearn", "xgboost", "rdkit", "joblib"),
                    reason="predictive extras (pandas/sklearn/xgboost/rdkit) not installed")
def test_predictive_pipeline_end_to_end(tmp_path):
    """features → select → train on a tiny synthetic dataset, then predict on new SMILES.
    Proves the config-driven predictive stack runs without xTB, LLM, or the generator."""
    smis = ["CCO", "OCCO", "CC(=O)O", "CCN(CC)CC", "OCCN(CCO)CCO", "CCOC(C)=O",
            "O=C1OCCO1", "CCS", "CCOCC", "NCCO", "OCCOCC", "CC(N)=O"]
    ces = [95, 96, 91, 97, 99, 93, 90, 92, 94, 98, 96, 91]
    lines = ["Additive_SMILES,CE_1 (%),CE_2 (%),CE_3 (%),CE_aver. (%),LogMolarRatio"]
    for s, c in zip(smis, ces):
        lines.append(f"{s},{c},{c},{c},{c},-1.5")
    csv = tmp_path / "mini.csv"; csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Override rather than splat, so an inherited BATTERYGEN_ART_DIR doesn't collide.
    env = dict(_os_environ())
    env["BATTERYGEN_ART_DIR"] = str(tmp_path / "art")
    env["BATTERYGEN_CE_CSV"] = str(csv)

    def run(mod, *a):
        r = subprocess.run([sys.executable, "-m", f"batterygen.predictive.{mod}", *a],
                           cwd=str(PARENT), env=env, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, f"{mod} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
        return r

    run("features", "--no-xtb")
    run("select")
    run("train")

    model = tmp_path / "art" / "predictive" / "production_model.pkl"
    assert model.exists(), "production model bundle was not written"

    # predict() works on unseen SMILES and returns the expected shape
    code = (
        "import joblib; from batterygen.predictive import train;"
        f"b=joblib.load(r'{model}');"
        "o=train.predict(b, ['CCCO','OCCCO'], compute_xtb=False);"
        "assert o and all('pred' in v and 'unc' in v for v in o.values()); print('PREDICT_OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(PARENT), env=env,
                       capture_output=True, text=True, timeout=300)
    assert "PREDICT_OK" in r.stdout, f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}"


def _os_environ():
    import os
    return os.environ


# --------------------------------------------------------------------------- #
# Weight-dependent tests (skipped unless artifacts are present)
# --------------------------------------------------------------------------- #
@requires_stats
def test_spec_to_condition():
    from batterygen.core import config, data
    cond = data.spec_to_condition({"MolWt": 300, "qed": 0.8})
    assert cond.shape == (config.N_PROPS,)
    assert np.isfinite(cond).all()


@requires_weights
def test_batterygen_generate():
    from rdkit import Chem
    from batterygen import BatteryGen
    bg = BatteryGen(device="cpu")
    out = bg.generate(3)
    assert len(out) == 3
    assert all(Chem.MolFromSmiles(s) is not None for s in out)


@requires_weights
def test_batterygen_encode_decode_and_props():
    from batterygen import BatteryGen
    bg = BatteryGen(device="cpu")
    z = bg.encode("OCCN(CCO)CCO")
    assert z is not None and z.ndim == 1
    assert isinstance(bg.decode(z), str) and bg.decode(z)
    props = bg.properties("CCO")
    assert props and "MolWt" in props


@requires_weights
def test_finetune_cli_end_to_end(tmp_path):
    """Fine-tune for 1 epoch on a tiny file and confirm the checkpoint reloads."""
    from batterygen.core import infer
    smi = tmp_path / "mini.smi"
    smi.write_text("\n".join(["CCO", "OCCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC",
                              "OCCN(CCO)CCO", "CCOC(C)=O", "O=C1OCCO1"]) + "\n")
    out = tmp_path / "ft.pt"
    r = subprocess.run(
        [sys.executable, "-m", "batterygen.core.finetune", "--input", str(smi),
         "--epochs", "1", "--batch", "4", "--device", "cpu", "--out", str(out)],
        cwd=str(PARENT), capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.exists()
    net, vocab, ck, device = infer.load_model(str(out), device="cpu")
    assert "hparams" in ck
