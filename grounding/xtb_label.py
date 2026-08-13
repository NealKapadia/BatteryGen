"""Label a subset of molecules with GFN2-xTB (a fast semi-empirical DFT surrogate).

For each molecule: RDKit 3D-embed (ETKDG) -> MMFF clean-up -> xyz -> run xtb.exe,
then parse HOMO / LUMO / HOMO-LUMO gap (eV), dipole (Debye) and total energy (Eh).
Results are appended (de-duplicated) to batterygen_artifacts/dft/labels.csv.

  python -m batterygen.grounding.xtb_label --n 300 --source dataset
  python -m batterygen.grounding.xtb_label --input my_smiles.txt          # one SMILES per line / csv
  python -m batterygen.grounding.xtb_label --n 500 --opt                  # geometry-optimize first
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from batterygen.core import config

from batterygen.core import data


_RE_GAP = re.compile(r"HOMO-LUMO gap\s+(-?\d+\.\d+)", re.IGNORECASE)
_RE_HOMO = re.compile(r"(-?\d+\.\d+)\s+\(HOMO\)")
_RE_LUMO = re.compile(r"(-?\d+\.\d+)\s+\(LUMO\)")
_RE_ETOT = re.compile(r"total energy\s+(-?\d+\.\d+)", re.IGNORECASE)
_RE_DIP = re.compile(r"full:\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+(-?\d+\.\d+)")


def parse_xtb(out: str) -> Dict[str, Optional[float]]:
    res: Dict[str, Optional[float]] = {k: None for k in config.DFT_PROPERTIES}
    m = _RE_GAP.search(out)
    if m:
        res["gap"] = float(m.group(1))
    m = _RE_HOMO.search(out)
    if m:
        res["homo"] = float(m.group(1))
    m = _RE_LUMO.search(out)
    if m:
        res["lumo"] = float(m.group(1))
    etots = _RE_ETOT.findall(out)
    if etots:
        res["total_energy"] = float(etots[-1])
    # dipole total: restrict to the molecular-dipole section
    lo = out.lower().find("molecular dipole")
    hi = out.lower().find("molecular quadrupole")
    section = out[lo: hi if hi > lo else lo + 600] if lo >= 0 else out
    m = _RE_DIP.search(section)
    if m:
        res["dipole"] = float(m.group(1))
    if res["gap"] is None and res["homo"] is not None and res["lumo"] is not None:
        res["gap"] = res["lumo"] - res["homo"]
    return res


def run_xtb(smiles: str, scratch: Path, *, opt: bool, charge: int, timeout: int,
            env: dict) -> Optional[Dict]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    canon = data.canonical_smiles(smiles)
    if canon is None:
        return None
    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, p) != 0:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    (scratch / "mol.xyz").write_text(Chem.MolToXYZBlock(mol))
    cmd = [config.XTB_EXE, "mol.xyz", "--gfn", "2", "--chrg", str(charge)]
    cmd.append("--opt" if opt else "--sp")
    try:
        r = subprocess.run(cmd, cwd=scratch, capture_output=True, text=True,
                          encoding="latin-1", timeout=timeout, env=env)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    res = parse_xtb(r.stdout)
    if res["gap"] is None:
        return None
    res["smiles"] = canon
    return res


def _reservoir_dataset(n: int, seed: int = 0) -> List[str]:
    import json

    rng = random.Random(seed)
    meta = json.load(open(config.META_PATH, encoding="utf-8"))
    sample: List[str] = []
    seen = 0
    for shard in meta["shards"]:
        with open(config.PROC_DIR / f"{shard}_ids.txt", encoding="utf-8") as f:
            for line in f:
                canon = line.partition("\t")[0]
                if not canon:
                    continue
                seen += 1
                if len(sample) < n:
                    sample.append(canon)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < n:
                        sample[j] = canon
    return sample


def _read_smiles_file(path: Path) -> List[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tok = line.strip().split(",")[0].split("\t")[0]
            if tok and tok.lower() not in ("smiles", "smiles_canonical"):
                out.append(tok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="how many molecules to label")
    ap.add_argument("--source", choices=["dataset", "file"], default="dataset")
    ap.add_argument("--input", type=str, default=None, help="SMILES file (with --source file)")
    ap.add_argument("--opt", action="store_true", help="geometry-optimize (slower, more accurate)")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--threads", type=int, default=0, help="OMP threads per xtb call (0=default)")
    ap.add_argument("--out", type=str, default=str(config.DFT_DIR / "labels.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config.ensure_dirs()
    if not Path(config.XTB_EXE).exists():
        raise SystemExit(f"xtb not found at {config.XTB_EXE} (set $XTB_EXE).")

    if args.source == "file":
        if not args.input:
            raise SystemExit("--source file requires --input")
        mols = _read_smiles_file(Path(args.input))[: args.n] if args.n else _read_smiles_file(Path(args.input))
    else:
        mols = _reservoir_dataset(args.n, args.seed)
    print(f"Labeling {len(mols)} molecules with GFN2-xTB ({'opt' if args.opt else 'single-point'}) ...")

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {r["smiles"] for r in csv.DictReader(f)}
    fields = ["smiles"] + config.DFT_PROPERTIES
    write_header = not out_path.exists()

    env = dict(os.environ)
    if args.threads > 0:
        env["OMP_NUM_THREADS"] = str(args.threads)
        env["XTB_NUM_THREADS"] = str(args.threads)

    scratch = Path(tempfile.mkdtemp(prefix="xtb_"))
    ok = fail = 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for smi in tqdm(mols, unit="mol", dynamic_ncols=True):
            canon = data.canonical_smiles(smi)
            if canon is None or canon in done:
                continue
            res = run_xtb(smi, scratch, opt=args.opt, charge=args.charge,
                          timeout=args.timeout, env=env)
            if res is None:
                fail += 1
                continue
            done.add(res["smiles"])
            w.writerow({k: res.get(k) for k in fields})
            f.flush()
            ok += 1
    print(f"Done. labeled {ok}, failed {fail}. -> {out_path}")


if __name__ == "__main__":
    main()
