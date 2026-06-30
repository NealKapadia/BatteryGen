"""Domain dictionaries for electrolyte datasets: solvent abbreviation -> SMILES and
salt name -> (cation, anion SMILES). Used to normalize CALiSol-23 (which encodes
solvents as named fraction columns and salts as names) into SMILES the VAE can encode.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

# Solvent abbreviation -> SMILES. Keys match the CALiSol-23 column names.
SOLVENT_SMILES = {
    "EC": "C1COC(=O)O1", "PC": "CC1COC(=O)O1", "DMC": "COC(=O)OC",
    "EMC": "CCOC(=O)OC", "DEC": "CCOC(=O)OCC", "DME": "COCCOC",
    "DMSO": "CS(C)=O", "AN": "CC#N", "MOEMC": "COCCOC(=O)OC",
    "EA": "CCOC(C)=O", "MA": "COC(C)=O", "FEC": "O=C1OCC(F)O1",
    "DOL": "C1COCO1", "2-MeTHF": "CC1CCCO1", "DMM": "COCOC",
    "Freon 11": "FC(Cl)(Cl)Cl", "Methylene chloride": "ClCCl", "THF": "C1CCOC1",
    "Toluene": "Cc1ccccc1", "Sulfolane": "O=S1(=O)CCCC1",
    "2-Glyme": "COCCOCCOC", "3-Glyme": "COCCOCCOCCOC", "4-Glyme": "COCCOCCOCCOCCOC",
    "3-Me-2-Oxazolidinone": "CN1CCOC1=O", "3-MeSulfolane": "CC1CCS(=O)(=O)C1",
    "Ethyldiglyme": "CCOCCOCCOCC", "DMF": "CN(C)C=O", "Ethylbenzene": "CCc1ccccc1",
    "Ethylmonoglyme": "CCOCCOCC", "Benzene": "c1ccccc1", "g-Butyrolactone": "O=C1CCCO1",
    "Cumene": "CC(C)c1ccccc1", "Propylsulfone": "CCCS(=O)(=O)CCC",
    "Pseudocumeme": "Cc1ccc(C)c(C)c1", "TEOS": "CCO[Si](OCC)(OCC)OCC",
    "m-Xylene": "Cc1cccc(C)c1", "o-Xylene": "Cc1ccccc1C",
    # "TFP" is ambiguous in the source -> intentionally omitted (column skipped).
}

# Anion abbreviation -> SMILES (the charged conjugate base).
ANION_SMILES = {
    "PF6": "F[P-](F)(F)(F)(F)F", "BF4": "[B-](F)(F)(F)F",
    "CLO4": "[O-]Cl(=O)(=O)=O", "ASF6": "F[As-](F)(F)(F)(F)F",
    "TFSI": "FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F",
    "NTF2": "FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F",
    "FSI": "O=S(=O)(F)[N-]S(=O)(=O)F",
    "OTF": "[O-]S(=O)(=O)C(F)(F)F", "TF": "[O-]S(=O)(=O)C(F)(F)F",
    "CF3SO3": "[O-]S(=O)(=O)C(F)(F)F", "TFO": "[O-]S(=O)(=O)C(F)(F)F",
    "BETI": "FC(F)(F)C(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)C(F)(F)F",
    "NO3": "[O-][N+](=O)[O-]", "BR": "[Br-]", "CL": "[Cl-]", "I": "[I-]",
    "DFOB": "[B-]1(F)(F)OC(=O)C(=O)O1", "FNFSI": "O=S(=O)(F)[N-]S(=O)(=O)F",
}

_CATIONS = ["Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Zn", "Al", "Fe"]
# normalize common salt-name spellings to an ANION_SMILES key
_ANION_ALIASES = {
    "tfsi": "TFSI", "ntf2": "TFSI", "n(so2cf3)2": "TFSI", "bis(trifluoromethanesulfonyl)imide": "TFSI",
    "fsi": "FSI", "n(so2f)2": "FSI", "otf": "OTF", "triflate": "OTF", "cf3so3": "OTF",
    "pf6": "PF6", "bf4": "BF4", "clo4": "CLO4", "asf6": "ASF6", "beti": "BETI",
    "no3": "NO3", "dfob": "DFOB", "br": "BR", "cl": "CL", "i": "I", "bob": "DFOB",
}


def parse_salt(name: str) -> Tuple[Optional[str], Optional[str]]:
    """'LiPF6' -> ('Li', 'F[P-](F)...'); returns (cation, anion_smiles), either may be None."""
    if not name:
        return None, None
    s = name.strip()
    cation = next((c for c in _CATIONS if s.startswith(c)), None)
    rest = s[len(cation):] if cation else s
    key = _ANION_ALIASES.get(rest.lower().strip(), rest.upper().replace("LI", "").strip())
    anion = ANION_SMILES.get(key)
    return cation, anion
