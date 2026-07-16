"""
Dataset loading, column normalisation, and PyTorch data plumbing.

Expects He et al.'s CSV schema
    compound_iso_smiles, target_key, target_sequence, affinity
Legacy `smiles,sequence,label` headers are auto-mapped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .chem import canonical_smiles, murcko_scaffold, protein_key
from .config import CFG, COL_AFF, COL_SMILES, COL_TKEY, COL_TSEQ


@dataclass
class Triple:
    """One drug-target-affinity record with its precomputed entity keys."""
    smiles: str
    sequence: str
    label: float
    drug_key: str
    prot_key: str
    scaffold: str | None = None


def _normalise_columns(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Accept He et al. column names, or map legacy smiles/sequence/label."""
    lower = {c.lower().strip(): c for c in df.columns}
    smi = lower.get(COL_SMILES) or lower.get("smiles") or lower.get("compound_smiles")
    seq = lower.get(COL_TSEQ) or lower.get("sequence") or lower.get("protein") or lower.get("target")
    aff = lower.get(COL_AFF) or lower.get("label") or lower.get("y")
    key = lower.get(COL_TKEY) or lower.get("target_name") or lower.get("target_id")
    if smi is None or seq is None:
        raise ValueError(
            f"{path}: need drug + protein columns ('{COL_SMILES}'/'{COL_TSEQ}' "
            f"or 'smiles'/'sequence'); got {list(df.columns)}")
    ren = {smi: COL_SMILES, seq: COL_TSEQ}
    if aff is not None:
        ren[aff] = COL_AFF
    if key is not None:
        ren[key] = COL_TKEY
    df = df.rename(columns=ren)
    if COL_TKEY not in df.columns:
        # He et al. key cold-target on target_key; fall back to the sequence
        # itself when no key column exists so the split stays entity-disjoint.
        df[COL_TKEY] = df[COL_TSEQ].astype(str).str.strip().str.upper()
        print(f"[data] {os.path.basename(path)}: no '{COL_TKEY}' column; "
              f"derived it from '{COL_TSEQ}'")
    return df


def load_dataframe(path: str, label_agnostic: bool = False) -> pd.DataFrame:
    """
    Load a dataset CSV.  `label_agnostic=True` is used for the BindingDB
    pre-training corpus, where Stages 1-3 never read an affinity value.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"data file not found: {path}")
    df = _normalise_columns(pd.read_csv(path), path)
    if COL_AFF not in df.columns:
        if not label_agnostic:
            raise ValueError(f"{path}: no affinity/label column")
        df[COL_AFF] = 0.0
    before = len(df)
    df = df.dropna(subset=[COL_SMILES, COL_TSEQ]).reset_index(drop=True)
    if before != len(df):
        print(f"[data] {os.path.basename(path)}: dropped {before - len(df)} null rows")
    print(f"[data] {os.path.basename(path)}: {len(df)} rows, "
          f"{df[COL_SMILES].nunique()} drugs, {df[COL_TKEY].nunique()} targets")
    return df


def df_to_triples(df: pd.DataFrame, need_scaffold: bool = False) -> List[Triple]:
    """Convert a dataframe to Triples, dropping unparsable SMILES."""
    rows: List[Triple] = []
    dropped = 0
    for r in df.itertuples(index=False):
        smi = str(getattr(r, COL_SMILES)).strip()
        seq = str(getattr(r, COL_TSEQ)).strip()
        dk = canonical_smiles(smi)
        if dk is None or not seq:
            dropped += 1
            continue
        sc = murcko_scaffold(smi) if need_scaffold else None
        rows.append(Triple(smi, seq, float(getattr(r, COL_AFF)), dk, protein_key(seq), sc))
    if dropped:
        print(f"[data] dropped {dropped} rows with unparsable SMILES")
    return rows


class DTBADataset(Dataset):
    def __init__(self, rows: List[Triple]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Triple:
        return self.rows[i]


def collate(batch: List[Triple]):
    return ([b.smiles for b in batch],
            [b.sequence for b in batch],
            torch.tensor([b.label for b in batch], dtype=torch.float32))


def make_loader(rows: List[Triple], shuffle: bool, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(DTBADataset(rows), batch_size=CFG["batch_size"],
                      shuffle=shuffle, collate_fn=collate,
                      generator=g if shuffle else None, drop_last=False)
