"""
Global configuration, device selection, column names, and seeding.

All hyperparameters live here so a run can be reproduced from this file alone.
Section numbers in comments refer to the manuscript.
"""

from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# He et al. (2023) CSV schema (NHGNN-DTA Data/davis.csv, Data/kiba.csv)
# ---------------------------------------------------------------------------
COL_SMILES = "compound_iso_smiles"
COL_TKEY = "target_key"
COL_TSEQ = "target_sequence"
COL_AFF = "affinity"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG: Dict = {
    # ---- Backbones (Sec. 4.4) -------------------------------------------
    # Drug backbone.  DrugEncoder reads the hidden size from the checkpoint at
    # load time, so d_drug always matches the model named here.
    #   DeepChem/ChemBERTa-77M-MLM        -> ChemBERTa-2, 77M, d_drug = 384
    #   seyonec/ChemBERTa-zinc-base-v1    -> ChemBERTa v1,      d_drug = 768
    "chemberta_name": "DeepChem/ChemBERTa-77M-MLM",
    "esm_name": "esm2_t33_650M_UR50D",   # 650M / 33 layers / d_prot = 1280
    "esm_repr_layer": 33,

    # ---- Adapter (Sec. 3.2.2) -------------------------------------------
    "adapter_m": 128,
    "alpha_init": 1e-2,

    # ---- Perceiver IO (Sec. 3.4) ----------------------------------------
    "perceiver_M": 128,
    "perceiver_L": 6,
    "perceiver_heads": 8,
    "d_shared": 768,

    # ---- MLP head -------------------------------------------------------
    "mlp_hidden": 256,
    "mlp_dropout": 0.2,

    # ---- Fingerprints, ECFP4 (Sec. 3.5) ---------------------------------
    "fp_radius": 2,
    "fp_bits": 2048,
    "fp_chirality": False,

    # ---- Contrastive objectives (Sec. 3.5, Table 3) ---------------------
    "temperature": 0.07,
    "top_k": 5,
    "hard_neg_thr": 0.3,        # delta in |S_d - S_p| > delta      (Step 4)
    "hard_random_ratio": 3,     # random:hard = 3:1                 (Step 4)
    "lambda_hyb": 0.5,
    "margin": 0.5,
    "sim_threshold": 0.4,
    "agreement": "product",     # product | geometric | harmonic

    # ---- BLOSUM62 alignment (Sec. 3.3.3) --------------------------------
    "blosum_gap_open": -11.0,   # NCBI BLASTP defaults for BLOSUM62
    "blosum_gap_extend": -1.0,
    "blosum_mode": "global",

    # ---- Optimisation (Table 3) -----------------------------------------
    "lr": 1e-4,
    "adapter_lr_ratio": 0.01,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,
    "batch_size": 32,
    "max_epochs": 200,
    "patience": 10,
    "min_delta": 1e-5,
    "cosine_tmax": 50,
    "cosine_eta_min": 1e-6,

    # ---- Sequence limits (Sec. 4.4) -------------------------------------
    "max_drug_len": 128,
    "max_prot_len": 1022,       # ESM-2 positional limit (1024 - BOS/EOS)

    # ---- Protocol (Sec. 4.1) --------------------------------------------
    "seeds": [42, 123, 456],
    "frac": [0.8, 0.1, 0.1],    # He et al. train/valid/test fractions
    "pretrain_val_frac": 0.05,

    # ---- Stricter protocols (Sec. 4.8) ----------------------------------
    "seqid_threshold": 0.40,    # CD-HIT identity cutoff
    "cdhit_bin": "cd-hit",      # falls back to internal clustering if absent

    # ---- Efficiency -----------------------------------------------------
    "cache_embeddings": True,
    "max_cache_entries": 20000,

    # ---- Data paths  <-- EDIT THESE -------------------------------------
    "bindingdb_path": "data/bindingdb_pretrain.csv",
    "davis_path": "data/davis.csv",
    "kiba_path": "data/kiba.csv",
    "out_dir": "./simcf_runs",
}


def set_seed(seed: int) -> None:
    """Full determinism for a single run (Sec. 4.1)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
