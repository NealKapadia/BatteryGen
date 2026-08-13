"""ONE command to finish the model after base pre-training.

Runs, in order (each stage resumable — re-running skips finished stages):

  1. electrolyte_data : CALiSol-23 + OEDB -> electrolyte_train.csv
  2. add_data         : continued pre-training corpus = ZINC-250k + ChEMBL(sample)
                        + electrolyte solvents (+ OpenQDC if fetched) -> new shards
  3. continued_train  : train.py --resume on the expanded dataset
  4. ground           : QM9 real-DFT labels -> finetune_dft property head
  5. electrolyte_train: multi-target conductivity/coordination model (cation-aware)
  6. eval             : benchmark vs ElectrolyteGPT + HTML report

Run AFTER the base run finishes (it needs latest.pt). GPU is free by then, so no
contention.

  python -m batterygen.generative.pipeline                       # do everything
  python -m batterygen.generative.pipeline --dry-run             # print the plan
  python -m batterygen.generative.pipeline --only electrolyte_train
  python -m batterygen.generative.pipeline --from ground         # resume from a stage
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from batterygen.core import config


BATTERYGEN = Path(__file__).resolve().parent
PY = sys.executable
STATE = config.ART_DIR / "pipeline_state.json"


def _script(name):
    return str(BATTERYGEN / name)


def _run(parts, desc, dry):
    cmd = [PY] + [str(p) for p in parts]
    print(f"\n===== {desc} =====")
    print("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if dry:
        return
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit(f"\n[pipeline] stage FAILED: {desc}. Fix and re-run "
                         f"`python -m batterygen.generative.pipeline --from <stage>`.")


def _need_vae():
    if not (config.CKPT_DIR / "latest.pt").exists():
        raise SystemExit("No latest.pt — finish base training first "
                         "(python -m batterygen.generative.train --epochs 12 --batch 320).")


# --------------------------------------------------------------------------- #
# helpers used by stages
# --------------------------------------------------------------------------- #
def _extract_zinc(zinc_csv: Path, out: Path) -> int:
    n = 0
    with open(zinc_csv, encoding="utf-8", errors="ignore", newline="") as f, \
         open(out, "w", encoding="utf-8") as o:
        r = csv.reader(f); next(r, None)
        for parts in r:
            if parts:
                smi = parts[0].strip().strip('"').strip()
                if smi:
                    o.write(smi + "\n"); n += 1
    return n


def _extract_solvents(elec_csv: Path, out: Path) -> int:
    seen = set()
    if elec_csv.exists():
        with open(elec_csv, encoding="utf-8") as f:
            for d in csv.DictReader(f):
                for tok in (d.get("mix", "") or "").split(";"):
                    smi = tok.rpartition(":")[0] or tok
                    if smi.strip():
                        seen.add(smi.strip())
                a = (d.get("anion_smiles", "") or "").strip()
                if a:
                    seen.add(a)
    out.write_text("\n".join(sorted(seen)), encoding="utf-8")
    return len(seen)


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_electrolyte_data(a, dry):
    _run([_script("electrolyte_data.py"), "--calisol", a.calisol, "--oedb", a.oedb],
         "1. CALiSol-23 + OEDB -> electrolyte_train.csv", dry)


def stage_add_data(a, dry):
    elec = config.ART_DIR / "electrolyte_train.csv"
    zinc_txt = config.DFT_DIR / "zinc_smiles.txt"
    solv_txt = config.DFT_DIR / "electrolyte_solvents.txt"
    if not dry:
        config.ensure_dirs()
        print(f"  ZINC SMILES: {_extract_zinc(Path(a.zinc), zinc_txt)}")
        print(f"  electrolyte solvents: {_extract_solvents(elec, solv_txt)}")
    inputs = []
    if Path(a.chembl).exists():
        inputs.append(a.chembl)
    inputs += [str(zinc_txt), str(solv_txt)]
    oq = config.DFT_DIR / "openqdc_smiles.txt"
    if oq.exists():
        inputs.append(str(oq))
    _run([_script("add_data.py"), "--input", *inputs, "--tag", "ext",
          "--limit", a.chembl_limit, "--dedup"],
         "2. continued-pretrain corpus: ZINC + ChEMBL + solvents (deduped)", dry)


def stage_continued_train(a, dry):
    if not dry:
        _need_vae()
    _run([_script("train.py"), "--resume", "--reset-schedule", "--patience", "2",
          "--batch", a.batch, "--epochs", a.epochs],
         "3. continued pre-training (clean schedule + early stop)", dry)


def stage_ground(a, dry):
    if not dry:
        _need_vae()
    _run([_script("qm9.py"), "--max", a.qm9_max], "4a. fetch QM9 real-DFT labels", dry)
    _run([_script("finetune_dft.py"), "--target", "homo,lumo,gap,dipole",
          "--labels", str(config.DFT_DIR / "qm9_labels.csv"), "--epochs", "300"],
         "4b. ground property head on QM9", dry)


def stage_electrolyte_train(a, dry):
    if not dry:
        _need_vae()
    elec = config.ART_DIR / "electrolyte_train.csv"
    _run([_script("electrolyte.py"), "--mode", "train", "--csv", str(elec),
          "--mix-col", "mix", "--cation-col", "cation", "--anion-smiles-col", "anion_smiles",
          "--conc-col", "conc", "--temp-col", "temp", "--source-col", "source",
          "--target-cols", "conductivity,coord_cat_anion,coord_cat_solvent",
          "--log-target", "--epochs", "500"],
         "5. electrolyte conductivity + coordination model (cation-aware)", dry)


def stage_eval(a, dry):
    if not dry:
        _need_vae()
    _run([_script("evaluate.py"), "--gen", "5000"], "6a. benchmark vs ElectrolyteGPT", dry)
    _run([_script("report.py")], "6b. HTML report", dry)


STAGES = [
    ("electrolyte_data", stage_electrolyte_data),
    ("add_data", stage_add_data),
    ("continued_train", stage_continued_train),
    ("ground", stage_ground),
    ("electrolyte_train", stage_electrolyte_train),
    ("eval", stage_eval),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zinc", default=str(config.DATA_DIR / "250k_rndm_zinc_drugs_clean_3.csv"))
    ap.add_argument("--chembl", default=str(config.DATA_DIR / "chembl_37.sdf"))
    ap.add_argument("--chembl-limit", default="800000", help="ChEMBL molecules to sample (0=all 2.4M)")
    ap.add_argument("--calisol", default=str(config.DATA_DIR / "calisol23_DOI_10.11583DTU.c.6929599.csv"))
    ap.add_argument("--oedb", default=str(config.DATA_DIR / "oedb-electrolytes-v2026-05-11.csv"))
    ap.add_argument("--batch", default="320")
    ap.add_argument("--epochs", default="3", help="continued-training epochs (fresh schedule; converges fast)")
    ap.add_argument("--qm9-max", default="0", help="QM9 molecules (0=all 134k)")
    ap.add_argument("--from", dest="from_stage", default=None, help="start at this stage")
    ap.add_argument("--only", default=None, help="run only this stage")
    ap.add_argument("--force", action="store_true", help="re-run completed stages")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    config.ensure_dirs()

    state = json.loads(STATE.read_text()) if STATE.exists() else {"done": []}
    names = [n for n, _ in STAGES]
    start = names.index(a.from_stage) if a.from_stage in names else 0
    print("Pipeline stages:", " -> ".join(names))
    print("Data: ZINC, ChEMBL(%s), CALiSol-23, OEDB | continued epochs=%s\n" % (a.chembl_limit, a.epochs))

    for i, (name, fn) in enumerate(STAGES):
        if a.only and name != a.only:
            continue
        if not a.only and i < start:
            continue
        if name in state["done"] and not a.force and not a.only:
            print(f"[skip] {name} (already done)")
            continue
        fn(a, a.dry_run)
        if not a.dry_run:
            state["done"] = sorted(set(state["done"]) | {name})
            STATE.write_text(json.dumps(state, indent=2))

    if not a.dry_run:
        print("\n✅ pipeline complete. Try the designer:  python -m batterygen.predictive.design")
        print("   screen electrolytes:  python -m batterygen.electrolyte.electrolyte --mode screen --cation Li --source oedb")
        print("   (conductivity model has data for Li/Na/K; add data for Mg/Zn/etc. to screen those.)")


if __name__ == "__main__":
    main()
