"""
Dataset splitting protocols.

`create_fold` and `create_fold_setting_cold` are faithful reimplementations of
He et al. (2023) NHGNN-DTA `Code/split.py`
(https://github.com/hehh77/NHGNN-DTA), itself derived from the Therapeutics
Data Commons splitters (https://tdc.readthedocs.io).

They are reproduced EXACTLY - including frac = [0.8, 0.1, 0.1] and the
hardcoded random_state=1 for the validation draw in the random fold - so that
SimCF-DTBA is evaluated on the SAME partitions as the published baseline
values cited in Tables 4-5.

DO NOT "clean up" these two functions.  Any deviation silently breaks
comparability with the cited baselines, which is precisely the defect the
reviewers flagged.

Also provides the stricter protocols of Sec. 4.8:
  * scaffold_drug  - Bemis-Murcko scaffold-disjoint
  * seqid_target   - CD-HIT sequence-identity-disjoint (40%)
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .chem import canonical_smiles, murcko_scaffold, protein_key
from .config import COL_AFF, COL_SMILES, COL_TKEY, COL_TSEQ
from .similarity import cluster_sequences, sequence_identity


# ---------------------------------------------------------------------------
# He et al. (2023) protocol - verbatim ports
# ---------------------------------------------------------------------------

def create_fold(df: pd.DataFrame, fold_seed: int, frac: List[float]) -> Dict[str, pd.DataFrame]:
    """Random split.  Verbatim port of He et al. split.py::create_fold."""
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
    train_val = df[~df.index.isin(test.index)]
    # NOTE: random_state=1 is hardcoded upstream, NOT fold_seed.  Preserved
    # deliberately so our random-split partitions match He et al.'s exactly.
    val = train_val.sample(frac=val_frac / (1 - test_frac), replace=False, random_state=1)
    train = train_val[~train_val.index.isin(val.index)]
    return {"train": train.reset_index(drop=True),
            "valid": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def create_fold_setting_cold(df: pd.DataFrame, fold_seed: int, frac: List[float],
                             entities) -> Dict[str, pd.DataFrame]:
    """
    Entity-level cold split.  Verbatim port of He et al.
    split.py::create_fold_setting_cold.

    cold-drug   -> entities = ['compound_iso_smiles']
    cold-target -> entities = ['target_key']
    """
    if isinstance(entities, str):
        entities = [entities]
    train_frac, val_frac, test_frac = frac

    test_entity_instances = [
        df[e].drop_duplicates().sample(frac=test_frac, replace=False,
                                       random_state=fold_seed).values
        for e in entities
    ]
    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]
    if len(test) == 0:
        raise ValueError("No test samples found. Try another seed, increasing the "
                         "test frac, or a less stringent splitting strategy.")

    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [
        train_val[e].drop_duplicates().sample(frac=val_frac / (1 - test_frac),
                                              replace=False,
                                              random_state=fold_seed).values
        for e in entities
    ]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]
    if len(val) == 0:
        raise ValueError("No validation samples found. Try another seed, increasing "
                         "the val frac, or a less stringent splitting strategy.")

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return {"train": train.reset_index(drop=True),
            "valid": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def create_fold_group(df: pd.DataFrame, fold_seed: int, frac: List[float],
                      group_col: str) -> Dict[str, pd.DataFrame]:
    """
    Group-disjoint split used by both stricter protocols (Sec. 4.8).

    Whole groups (scaffolds, or sequence-identity clusters) are assigned
    entirely to train, valid, or test.  Structurally identical to
    create_fold_setting_cold but keyed on a precomputed group column.
    """
    return create_fold_setting_cold(df, fold_seed, frac, [group_col])


# ---------------------------------------------------------------------------
# Dispatch + verification
# ---------------------------------------------------------------------------

SPLITS = ("cold_drug", "cold_target", "cold_drug_target",
          "random", "scaffold_drug", "seqid_target")


def build_splits(df: pd.DataFrame, split: str, seed: int, cfg: Dict):
    """
    Dispatch to the correct protocol.

    Returns
        (train_df, valid_df, test_df, held_drug_keys, held_prot_seqs, meta)

    cold_drug / cold_target / random  -> He et al. protocol (Tables 4-6)
    scaffold_drug / seqid_target      -> stricter protocols (Table 8)

    Every protocol is asserted after the fact: if the cold property does not
    hold, this raises rather than silently reporting an inflated score.
    """
    frac = cfg["frac"]
    meta: Dict = {"split": split, "seed": seed}

    if split == "random":
        folds = create_fold(df, seed, frac)

    elif split == "cold_drug":
        folds = create_fold_setting_cold(df, seed, frac, [COL_SMILES])

    elif split == "cold_target":
        folds = create_fold_setting_cold(df, seed, frac, [COL_TKEY])

    elif split == "cold_drug_target":
        folds = create_fold_setting_cold(df, seed, frac, [COL_TKEY, COL_SMILES])

    elif split == "scaffold_drug":
        # Sec. 4.8: group drugs by Bemis-Murcko scaffold; whole scaffolds go to
        # one partition, so no test scaffold is ever seen during training.
        work = df.copy()
        uniq = work[COL_SMILES].drop_duplicates()
        smi2sc = {s: (murcko_scaffold(s) or "__unparsable__") for s in uniq}
        work["_scaffold"] = work[COL_SMILES].map(smi2sc)
        n_sc = work["_scaffold"].nunique()
        print(f"[scaffold] {len(uniq)} drugs -> {n_sc} Bemis-Murcko scaffolds")
        meta["n_scaffolds"] = int(n_sc)
        folds = create_fold_group(work, seed, frac, "_scaffold")

    elif split == "seqid_target":
        # Sec. 4.8: cluster targets by sequence identity; whole clusters go to
        # one partition, so no test protein has a >=40% identity homolog in
        # training.
        work = df.copy()
        uniq_seqs = work[COL_TSEQ].drop_duplicates().tolist()
        cl = cluster_sequences(uniq_seqs, cfg["seqid_threshold"], cfg)
        seq2cl = {s: f"c{c}" for s, c in zip(uniq_seqs, cl)}
        work["_seqcluster"] = work[COL_TSEQ].map(seq2cl)
        meta["n_clusters"] = int(len(set(cl)))
        meta["seqid_threshold"] = cfg["seqid_threshold"]
        folds = create_fold_group(work, seed, frac, "_seqcluster")

    else:
        raise ValueError(f"unknown split: {split} (expected one of {SPLITS})")

    tr, va, te = folds["train"], folds["valid"], folds["test"]

    # ---- verify the protocol actually holds ------------------------------
    if split in ("cold_drug", "scaffold_drug", "cold_drug_target"):
        assert not (set(tr[COL_SMILES]) & set(te[COL_SMILES])), "cold-drug leak: train/test"
        assert not (set(va[COL_SMILES]) & set(te[COL_SMILES])), "cold-drug leak: val/test"
    if split in ("cold_target", "seqid_target", "cold_drug_target"):
        assert not (set(tr[COL_TKEY]) & set(te[COL_TKEY])), "cold-target leak: train/test"
        assert not (set(va[COL_TKEY]) & set(te[COL_TKEY])), "cold-target leak: val/test"
    if split == "scaffold_drug":
        assert not (set(tr["_scaffold"]) & set(te["_scaffold"])), "scaffold leak"
    if split == "seqid_target":
        assert not (set(tr["_seqcluster"]) & set(te["_seqcluster"])), "seq-identity cluster leak"

    # ---- entities to purge from the BindingDB pre-training corpus ---------
    held_drugs: set = set()
    held_prots: set = set()
    if split in ("cold_drug", "scaffold_drug", "cold_drug_target"):
        held_drugs = {canonical_smiles(s) for s in te[COL_SMILES].unique()}
        held_drugs.discard(None)
    if split in ("cold_target", "seqid_target", "cold_drug_target"):
        held_prots = {protein_key(s) for s in te[COL_TSEQ].unique()}

    meta.update({"n_train": len(tr), "n_valid": len(va), "n_test": len(te)})
    print(f"[split] {split} seed={seed}: train={len(tr)} valid={len(va)} test={len(te)}")
    return tr, va, te, held_drugs, held_prots, meta


def filter_pretrain_corpus(df: pd.DataFrame, split: str, test_df: pd.DataFrame,
                           held_drugs: set, held_prots: set, cfg: Dict) -> pd.DataFrame:
    """
    Remove every BindingDB pair whose entity is held out for testing.

    Sec. 4.1 requires test entities to be excluded from pre-training AND
    supervised training.  Without this the cold-split claim is void: the model
    would have seen the "novel" entity during contrastive pre-training.

    Under the stricter protocols (Sec. 4.8) the exclusion is correspondingly
    stricter - scaffold-level for drugs and >=40% sequence identity for
    proteins - not merely exact matches.
    """
    before = len(df)
    out = df

    if split == "scaffold_drug":
        test_scaffolds = {murcko_scaffold(s) for s in test_df[COL_SMILES].unique()}
        test_scaffolds.discard(None)
        out = out[out[COL_SMILES].map(lambda s: murcko_scaffold(s) not in test_scaffolds)]
        print(f"[pretrain] scaffold-level exclusion vs {len(test_scaffolds)} test scaffolds")

    elif split == "seqid_target":
        test_seqs = [str(s) for s in test_df[COL_TSEQ].unique()]
        thr = cfg["seqid_threshold"]
        uniq = out[COL_TSEQ].drop_duplicates().tolist()
        print(f"[pretrain] sequence-identity exclusion: screening {len(uniq)} corpus "
              f"proteins vs {len(test_seqs)} test proteins @ {thr:.0%} "
              f"(cached alignments; this is the slow step)")
        drop: set = set()
        for s in uniq:
            for t in test_seqs:
                if sequence_identity(s, t) >= thr:
                    drop.add(s)
                    break
        out = out[~out[COL_TSEQ].isin(drop)]
        print(f"[pretrain] {len(drop)} corpus proteins removed as homologs")

    else:
        if held_drugs:
            out = out[out[COL_SMILES].map(lambda s: canonical_smiles(s) not in held_drugs)]
        if held_prots:
            out = out[~out[COL_TSEQ].map(protein_key).isin(held_prots)]

    out = out.reset_index(drop=True)
    print(f"[pretrain] leakage filter: {before} -> {len(out)} "
          f"({before - len(out)} pairs removed)")
    if len(out) == 0:
        raise RuntimeError("pre-training corpus is empty after leakage filtering")
    return out
