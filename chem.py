from __future__ import annotations
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

_uncharger = rdMolStandardize.Uncharger()
_lfc = rdMolStandardize.LargestFragmentChooser()

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def canonical_smiles(smiles: str, salt_strip: bool = True) -> str | None:
  
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if salt_strip:
        try:
            mol = _lfc.choose(mol)
            mol = _uncharger.uncharge(mol)
        except Exception:
            pass
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def murcko_scaffold(smiles: str) -> str | None:
   
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        s = Chem.MolToSmiles(core, canonical=True)
        return s if s else "__acyclic__"
    except Exception:
        return None


def protein_key(sequence: str) -> str:
    """Exact-sequence key for cold-target exclusion (Sec. 4.1)."""
    return str(sequence).strip().upper()


def clean_seq(seq: str, max_len: int) -> str:
    s = "".join(c for c in str(seq).strip().upper() if c in VALID_AA)
    return s[:max_len]
