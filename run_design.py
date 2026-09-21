"""
BatteryGen aligned inverse-design launcher.

Place this file directly in:
    <BatteryGen-main>/run_design.py

Run from Jupyter, for example:

    %run run_design.py --n 100 --rounds 1 --shortlist 20 --top 10 --out candidates_test.csv

or the full run:

    %run run_design.py --n 1200 --rounds 3 --shortlist 60 --top 30 --out candidates.csv

What this launcher does automatically:
- forces ALL BatteryGen artifacts into <BatteryGen-main>/batterygen_artifacts/
- recovers an already-trained production_model.pkl if it exists in an older nearby folder
- recovers existing pretrained generator assets if they exist nearby
- if generator assets are still missing, downloads processed/* and checkpoints/*
  from SuLabUTD/BatteryGen on Hugging Face
- routes relative --out files into batterygen_artifacts/design/
- selects CUDA automatically when available unless --device was explicitly supplied
- runs the ORIGINAL batterygen.predictive.design.main() without rewriting its
  recursive generation, xTB refinement, or disagreement-aware ranking
- prints a compact preview of the resulting CSV
"""

from pathlib import Path
import os
import shutil
import sys

# ============================================================================
# ONE PROJECT-LOCAL ARTIFACT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = PROJECT_ROOT / "batterygen_artifacts"
PREDICTIVE_ROOT = ARTIFACT_ROOT / "predictive"
DESIGN_ROOT = ARTIFACT_ROOT / "design"
PROCESSED_ROOT = ARTIFACT_ROOT / "processed"
CHECKPOINT_ROOT = ARTIFACT_ROOT / "checkpoints"

# Must be set BEFORE importing BatteryGen modules.
os.environ["BATTERYGEN_ART_DIR"] = str(ARTIFACT_ROOT)

for folder in (
    ARTIFACT_ROOT,
    PREDICTIVE_ROOT,
    DESIGN_ROOT,
    PROCESSED_ROOT,
    CHECKPOINT_ROOT,
):
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ARTIFACT RECOVERY HELPERS
# ============================================================================

def _newest_nearby(filename, exclude=None):
    """Find the newest matching file in this BatteryGen folder or its parent."""
    exclude = Path(exclude).resolve() if exclude is not None else None
    matches = []

    for root in (PROJECT_ROOT, PROJECT_ROOT.parent):
        if not root.exists():
            continue

        try:
            for p in root.rglob(filename):
                try:
                    if not p.is_file():
                        continue
                    rp = p.resolve()
                    if exclude is not None and rp == exclude:
                        continue
                    matches.append(p)
                except OSError:
                    pass
        except (OSError, PermissionError):
            pass

    if not matches:
        return None

    return max(matches, key=lambda p: p.stat().st_mtime)


def _recover_file(target):
    """Copy an already-existing nearby artifact into the aligned folder."""
    target = Path(target)

    if target.exists():
        return True

    source = _newest_nearby(target.name, exclude=target)

    if source is None:
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    print(f"[artifact alignment] recovered {target.name}")
    print(f"    from: {source}")
    print(f"      to: {target}")

    return True


def _ensure_predictive_model():
    target = PREDICTIVE_ROOT / "production_model.pkl"

    if not _recover_file(target):
        raise FileNotFoundError(
            "\nNo production_model.pkl was found.\n"
            f"Expected aligned location:\n  {target}\n\n"
            "Run your final predictive training first:\n"
            "  %run predictive/train.py\n"
            "or\n"
            "  !python -m batterygen.predictive.train"
        )

    return target


def _ensure_generator_assets():
    """
    Ensure the pretrained generator assets are inside the SAME artifact folder.

    The two strictly essential locations observed by BatteryGen are:
      processed/vocab.json
      checkpoints/best.pt

    We also recover common processed metadata when available.
    """
    targets = [
        PROCESSED_ROOT / "vocab.json",
        CHECKPOINT_ROOT / "best.pt",
        PROCESSED_ROOT / "descriptor_stats.json",
        PROCESSED_ROOT / "meta.json",
    ]

    for target in targets:
        _recover_file(target)

    essential = [
        PROCESSED_ROOT / "vocab.json",
        CHECKPOINT_ROOT / "best.pt",
    ]

    if all(p.exists() for p in essential):
        return

    print("\nPretrained generator assets are not complete.")
    print("Downloading BatteryGen processed/checkpoint assets from Hugging Face...")
    print(f"Destination: {ARTIFACT_ROOT}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed.\n"
            "Install it once with:\n"
            "  %pip install huggingface_hub"
        ) from exc

    try:
        snapshot_download(
            repo_id="NealKapadia/BatteryGen",
            local_dir=str(ARTIFACT_ROOT),
            allow_patterns=[
                "processed/*",
                "checkpoints/*",
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            "\nCould not download the pretrained BatteryGen generator.\n"
            "If the repository requires authentication, run once in Jupyter:\n\n"
            "    from huggingface_hub import login\n"
            "    login()\n\n"
            "Then rerun run_design.py.\n\n"
            f"Original error:\n{exc}"
        ) from exc

    missing = [p for p in essential if not p.exists()]

    if missing:
        formatted = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Generator download completed, but essential files are still missing:\n"
            + formatted
        )


# ============================================================================
# COMMAND-LINE NORMALIZATION
# ============================================================================

def _set_or_localize_out(args):
    """
    If --out is relative, save it inside batterygen_artifacts/design/.
    If --out is omitted, use candidates.csv there.
    """
    args = list(args)

    if "--out" in args:
        idx = args.index("--out")

        if idx + 1 >= len(args):
            raise ValueError("--out was supplied without a filename.")

        requested = Path(args[idx + 1])

        if not requested.is_absolute():
            requested = DESIGN_ROOT / requested

        requested.parent.mkdir(parents=True, exist_ok=True)
        args[idx + 1] = str(requested)

    else:
        args.extend(["--out", str(DESIGN_ROOT / "candidates.csv")])

    return args


def _set_device_if_missing(args):
    """Use CUDA automatically when available, but respect an explicit --device."""
    args = list(args)

    if "--device" in args:
        return args

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    args.extend(["--device", device])
    return args


def _extract_out(args):
    idx = args.index("--out")
    return Path(args[idx + 1]).resolve()


# ============================================================================
# RUN
# ============================================================================

def main():
    print("=" * 96)
    print("BATTERYGEN — ALIGNED RECURSIVE INVERSE DESIGN")
    print("=" * 96)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Artifact root      : {ARTIFACT_ROOT}")
    print(f"Predictive model   : {PREDICTIVE_ROOT / 'production_model.pkl'}")
    print(f"Generator vocab    : {PROCESSED_ROOT / 'vocab.json'}")
    print(f"Generator checkpoint: {CHECKPOINT_ROOT / 'best.pt'}")

    model_path = _ensure_predictive_model()
    _ensure_generator_assets()

    args = sys.argv[1:]
    args = _set_or_localize_out(args)
    args = _set_device_if_missing(args)

    out_path = _extract_out(args)

    print("-" * 96)
    print("Inputs ready")
    print(f"Production model exists : {model_path.exists()}")
    print(f"Generator vocab exists  : {(PROCESSED_ROOT / 'vocab.json').exists()}")
    print(f"Generator model exists  : {(CHECKPOINT_ROOT / 'best.pt').exists()}")
    print(f"Design output           : {out_path}")
    print("-" * 96)

    # Import only AFTER artifact root has been fixed and assets are in place.
    from batterygen.predictive import design

    old_argv = sys.argv[:]

    try:
        sys.argv = [old_argv[0]] + args
        design.main()
    finally:
        sys.argv = old_argv

    print("\n" + "=" * 96)
    print("INVERSE DESIGN COMPLETE")
    print("=" * 96)
    print(f"Output file: {out_path}")
    print(f"Exists     : {out_path.exists()}")

    if out_path.exists():
        try:
            import pandas as pd

            df = pd.read_csv(out_path)

            print(f"Candidates : {len(df)}")
            print("\nColumns:")
            print(list(df.columns))

            print("\nTop candidates:")
            with pd.option_context(
                "display.max_columns", 20,
                "display.width", 160,
                "display.max_colwidth", 40,
            ):
                print(df.head(10).to_string(index=False))

        except Exception as exc:
            print(f"\nOutput exists, but preview could not be displayed: {exc}")

    print("=" * 96)


if __name__ == "__main__":
    main()
