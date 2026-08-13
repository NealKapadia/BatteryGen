"""Naming guard tests: the project is `batterygen` / `BatteryGen`, everywhere.

The package was previously published under two older names, and the rename touches
every file in the tree — package name, class name, console scripts, environment
variables, docs. These tests are the executable definition of "the rename is done":
they fail loudly if any legacy token survives, or comes back in a later edit.

The legacy tokens are assembled from fragments so this file can scan the *whole*
tracked tree without excluding itself.

Run:  python -m pytest tests/test_naming.py -q
"""
import configparser
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent          # .../batterygen
PARENT = ROOT.parent                                   # dir where `batterygen` is importable

# Assembled at runtime so the literal strings never appear in this file's source.
_OLD = "mol" + "forge"                                 # old package / class stem
_OLDER = "mol" + "vae"                                 # older-still stem, still in env vars
LEGACY_TOKENS = (_OLD, _OLDER)
LEGACY_ENV_PREFIXES = (_OLD.upper() + "_", _OLDER.upper() + "_")

NEW_PKG = "batterygen"
NEW_CLASS = "BatteryGen"
NEW_ENV_PREFIX = "BATTERYGEN_"


def _tracked_files() -> list[Path]:
    r = subprocess.run(["git", "ls-files"], cwd=str(ROOT),
                       capture_output=True, text=True, check=True)
    return [ROOT / line for line in r.stdout.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# 1. No legacy token survives anywhere in the tracked tree
# --------------------------------------------------------------------------- #
def test_no_legacy_tokens_in_tracked_files():
    """Every tracked file — code, README, LICENSE, .env.example, .gitignore — is clean."""
    offenders: list[str] = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            for token in LEGACY_TOKENS:
                if token in low:
                    rel = path.relative_to(ROOT).as_posix()
                    offenders.append(f"{rel}:{lineno}: {line.strip()[:110]}")
    assert not offenders, (
        f"{len(offenders)} legacy name reference(s) remain:\n" + "\n".join(offenders[:60])
    )


def test_repo_folder_is_named_for_the_package():
    """`import batterygen` from the parent dir only resolves if the folder matches."""
    assert ROOT.name == NEW_PKG, (
        f"repo folder is {ROOT.name!r}; it must be {NEW_PKG!r} so the package imports in place"
    )


def test_git_remote_points_at_the_renamed_repo():
    r = subprocess.run(["git", "remote", "get-url", "origin"], cwd=str(ROOT),
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("no `origin` remote configured")
    url = r.stdout.strip().lower()
    assert "batterygen" in url, f"origin still points at {r.stdout.strip()}"
    for token in LEGACY_TOKENS:
        assert token not in url, f"origin still points at {r.stdout.strip()}"


# --------------------------------------------------------------------------- #
# 2. The public API is the new name
# --------------------------------------------------------------------------- #
def test_public_import_is_batterygen():
    import importlib
    pkg = importlib.import_module(NEW_PKG)
    cls = getattr(pkg, NEW_CLASS)
    assert cls.__name__ == NEW_CLASS
    assert pkg.__all__ == [NEW_CLASS]
    assert re.fullmatch(r"\d+\.\d+\.\d+", pkg.__version__)


def test_no_legacy_module_is_importable_from_the_tree():
    """A stale `molforge` copy in site-packages must not shadow or survive the rename."""
    import importlib
    for token in LEGACY_TOKENS:
        try:
            mod = importlib.import_module(token)
        except ImportError:
            continue
        origin = Path(getattr(mod, "__file__", "") or "").resolve()
        assert PARENT not in origin.parents, (
            f"legacy package {token!r} is still importable from the repo tree at {origin}"
        )


# --------------------------------------------------------------------------- #
# 3. Packaging metadata
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_project_name_and_urls(pyproject):
    assert pyproject["project"]["name"] == NEW_PKG
    for url in pyproject["project"].get("urls", {}).values():
        assert "batterygen" in url.lower()
    for author in pyproject["project"].get("authors", []):
        assert NEW_CLASS in author.get("name", "") or "Kapadia" in author.get("name", "")


def test_setuptools_packages_and_package_dir(pyproject):
    packages = pyproject["tool"]["setuptools"]["packages"]
    assert NEW_PKG in packages
    assert all(p == NEW_PKG or p.startswith(NEW_PKG + ".") for p in packages), packages
    # every purpose subpackage on disk is declared
    on_disk = {d.name for d in ROOT.iterdir()
               if d.is_dir() and (d / "__init__.py").exists() and not d.name.startswith(".")}
    assert {f"{NEW_PKG}.{d}" for d in on_disk} <= set(packages)
    assert pyproject["tool"]["setuptools"]["package-dir"] == {NEW_PKG: "."}


def test_console_scripts_are_renamed(pyproject):
    scripts = pyproject["project"]["scripts"]
    assert scripts, "no console scripts declared"
    assert NEW_PKG in scripts, f"bare `{NEW_PKG}` command is missing"
    for name, target in scripts.items():
        assert name == NEW_PKG or name.startswith(NEW_PKG + "-"), name
        assert target.startswith(NEW_PKG + "."), f"{name} -> {target}"


@pytest.mark.parametrize("name,target", sorted(
    tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"].items()
))
def test_every_console_script_target_resolves(name, target):
    """`batterygen-train = "batterygen.generative.train:main"` must actually point at a callable."""
    import importlib
    mod_path, _, func = target.partition(":")
    try:
        mod = importlib.import_module(mod_path)
    except ModuleNotFoundError as e:
        missing = (e.name or "").split(".")[0]
        if missing in (NEW_PKG, *LEGACY_TOKENS):
            raise
        pytest.skip(f"optional dependency {missing!r} not installed")
    assert callable(getattr(mod, func)), f"{target} is not callable"


# --------------------------------------------------------------------------- #
# 4. Environment variables
# --------------------------------------------------------------------------- #
def test_all_env_vars_use_the_new_prefix():
    """No `os.getenv("MOLVAE_...")` / `MOLFORGE_...` left in the tracked tree."""
    pattern = re.compile(r"\b(" + "|".join(LEGACY_ENV_PREFIXES) + r")[A-Z_]+")
    offenders: list[str] = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{lineno}: {line.strip()[:110]}")
    assert not offenders, "legacy env vars remain:\n" + "\n".join(offenders)


def test_config_reads_the_new_env_vars(monkeypatch, tmp_path):
    """The knobs users actually set are wired to BATTERYGEN_* and take effect."""
    import importlib
    monkeypatch.setenv(NEW_ENV_PREFIX + "ART_DIR", str(tmp_path / "art"))
    monkeypatch.setenv(NEW_ENV_PREFIX + "SEED", "1234")
    monkeypatch.setenv(NEW_ENV_PREFIX + "BATCH", "64")
    config = importlib.reload(importlib.import_module(f"{NEW_PKG}.core.config"))
    try:
        assert config.ART_DIR == tmp_path / "art"
        assert config.SEED == 1234
        assert config.BATCH_SIZE == 64
    finally:
        for k in ("ART_DIR", "SEED", "BATCH"):
            monkeypatch.delenv(NEW_ENV_PREFIX + k, raising=False)
        importlib.reload(config)


def test_ce_csv_env_var_is_renamed(tmp_path, monkeypatch):
    import importlib
    config = importlib.import_module(f"{NEW_PKG}.core.config")
    f = tmp_path / "my.csv"
    f.write_text("Additive_SMILES,CE_aver. (%)\nCCO,99\n")
    monkeypatch.setenv(NEW_ENV_PREFIX + "CE_CSV", str(f))
    assert config.resolve_ce_csv(None) == f


def test_env_example_documents_only_new_vars():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    declared = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, flags=re.M))
    legacy = {v for v in declared if v.startswith(LEGACY_ENV_PREFIXES)}
    assert not legacy, f".env.example still documents {sorted(legacy)}"
    assert any(v.startswith(NEW_ENV_PREFIX) for v in declared), \
        f".env.example documents no {NEW_ENV_PREFIX}* vars (found {sorted(declared)})"


def test_default_artifacts_dir_is_renamed(monkeypatch):
    """With no override set, the artifacts folder carries the new brand, not an old one."""
    import importlib
    monkeypatch.delenv(NEW_ENV_PREFIX + "ART_DIR", raising=False)
    config = importlib.reload(importlib.import_module(f"{NEW_PKG}.core.config"))
    try:
        assert config.ART_DIR.name == f"{NEW_PKG}_artifacts", config.ART_DIR
    finally:
        monkeypatch.undo()                       # restore the real env before reloading back
        importlib.reload(config)


# --------------------------------------------------------------------------- #
# 5. Docs match the shipped code
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_usage_snippet_matches_the_real_api(readme):
    assert f"from {NEW_PKG} import {NEW_CLASS}" in readme
    assert f"github.com/NealKapadia/BatteryGen" in readme


def test_readme_pip_install_url_is_the_renamed_repo(readme):
    installs = re.findall(r"pip install[^\n`]*", readme)
    assert installs, "README documents no pip install line"
    assert any("BatteryGen" in line for line in installs), installs


def test_readme_commands_exist_as_console_scripts(readme, pyproject):
    """Every `batterygen-foo` command the README documents is a real entry point.

    Only inline code spans count, so prose, markdown anchors and the suggested
    `batterygen-env` virtualenv name are not mistaken for commands. Brace forms like
    `batterygen-{generate,search,report}` are expanded first.
    """
    declared = set(pyproject["project"]["scripts"])
    mentioned: set[str] = set()
    for span in re.findall(r"`([^`\n]+)`", readme):
        span = span.strip()
        brace = re.fullmatch(rf"{NEW_PKG}-\{{([a-z0-9,-]+)\}}", span)
        if brace:                                # batterygen-{generate,search,report}
            mentioned.update(f"{NEW_PKG}-{v.strip()}" for v in brace.group(1).split(","))
            continue
        for piece in span.split(","):            # `batterygen-qm9, batterygen-electrolyte`
            piece = piece.strip()
            if re.fullmatch(rf"{NEW_PKG}(-[a-z0-9-]+)?", piece):
                mentioned.add(piece)
    assert mentioned, "README documents no console commands at all"
    unknown = mentioned - declared
    assert not unknown, f"README documents commands that do not exist: {sorted(unknown)}"


def test_license_carries_the_new_name():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert NEW_CLASS in text
