"""
Similarity oracles and sequence clustering.

  * Tanimoto over ECFP4                      
  * Alignment-based BLOSUM62 similarity      
  * Agreement function A(S_d, S_p)           
  * Sequence identity + CD-HIT clustering  

All oracles are label-agnostic: they never see an affinity value.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from Bio import Align
from Bio.Align import substitution_matrices
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

from .chem import clean_seq
from .config import CFG

# ---------------------------------------------------------------------------
# Tanimoto / ECFP4  
# ---------------------------------------------------------------------------

_mfp_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=CFG["fp_radius"],
    fpSize=CFG["fp_bits"],
    includeChirality=CFG["fp_chirality"],
)
_FP_CACHE: Dict[str, np.ndarray] = {}


def smiles_to_fp(smiles: str) -> np.ndarray:
    """ECFP4 bit vector, cached by SMILES string."""
    if smiles in _FP_CACHE:
        return _FP_CACHE[smiles]
    mol = Chem.MolFromSmiles(smiles)
    arr = np.zeros(CFG["fp_bits"], dtype=np.float32)
    if mol is not None:
        DataStructs.ConvertToNumpyArray(_mfp_gen.GetFingerprint(mol), arr)
    else:
        print(f"[warn] unparsable SMILES -> zero fingerprint: {smiles[:60]}")
    _FP_CACHE[smiles] = arr
    return arr


def tanimoto_matrix(smiles_list: Sequence[str]) -> torch.Tensor:
    """Pairwise Tanimoto similarity over ECFP4.  Returns (N, N) in [0, 1]."""
    fps = np.stack([smiles_to_fp(s) for s in smiles_list])
    t = torch.from_numpy(fps).float()
    inter = t @ t.T
    rs = t.sum(1, keepdim=True)
    union = rs + rs.T - inter
    return (inter / union.clamp(min=1e-8)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Alignment-based BLOSUM62  
# ---------------------------------------------------------------------------

_aligner = Align.PairwiseAligner()
_aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
_aligner.open_gap_score = CFG["blosum_gap_open"]
_aligner.extend_gap_score = CFG["blosum_gap_extend"]
_aligner.mode = CFG["blosum_mode"]

_ALIGN_CACHE: Dict[Tuple[str, str], float] = {}
_SELF_CACHE: Dict[str, float] = {}


def _raw_align_score(a: str, b: str) -> float:
    key = (a, b) if a <= b else (b, a)
    if key in _ALIGN_CACHE:
        return _ALIGN_CACHE[key]
    try:
        score = float(_aligner.score(a, b))
    except Exception:
        score = 0.0
    _ALIGN_CACHE[key] = score
    return score


def _self_score(a: str) -> float:
    if a not in _SELF_CACHE:
        _SELF_CACHE[a] = max(_raw_align_score(a, a), 1e-8)
    return _SELF_CACHE[a]


def blosum62_matrix(sequences: Sequence[str]) -> torch.Tensor:
    
    seqs = [clean_seq(s, CFG["max_prot_len"]) for s in sequences]
    n = len(seqs)
    raw = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            raw[i, j] = raw[j, i] = _raw_align_score(seqs[i], seqs[j])

    self_scores = np.array([_self_score(s) for s in seqs], dtype=np.float64)
    norm = raw / np.maximum(np.sqrt(np.outer(self_scores, self_scores)), 1e-8)

    row_min = norm.min(axis=1, keepdims=True)
    row_max = norm.max(axis=1, keepdims=True)
    span = np.where(row_max - row_min > 1e-8, row_max - row_min, 1.0)
    out = np.clip((norm - row_min) / span, 0.0, 1.0)
    return torch.from_numpy(out.astype(np.float32))


def agreement_weights(S_d: torch.Tensor, S_p: torch.Tensor, rule: str) -> torch.Tensor:
    """
    Agreement function A(S_d, S_p)  
    The paper's default is the product rule; the geometric and harmonic
    variants are provided for the sensitivity analysis  
    """
    if rule == "product":
        return (S_d * S_p).clamp(min=0.0)
    if rule == "geometric":
        return torch.sqrt((S_d.clamp(min=0.0) * S_p.clamp(min=0.0)).clamp(min=0.0))
    if rule == "harmonic":
        return (2.0 * S_d * S_p / (S_d + S_p).clamp(min=1e-8)).clamp(min=0.0)
    raise ValueError(f"unknown agreement rule: {rule}")


# ---------------------------------------------------------------------------
# Sequence identity + clustering  
# ---------------------------------------------------------------------------

def sequence_identity(a: str, b: str) -> float:
    
    a, b = clean_seq(a, CFG["max_prot_len"]), clean_seq(b, CFG["max_prot_len"])
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    try:
        aln = _aligner.align(a, b)[0]
    except Exception:
        return 0.0
    sa, sb = aln[0], aln[1]
    matches = sum(1 for x, y in zip(sa, sb) if x == y and x != "-")
    return matches / max(1, min(len(a), len(b)))


def cluster_sequences_cdhit(seqs: List[str], threshold: float,
                            cdhit_bin: str) -> List[int] | None:
    
    if shutil.which(cdhit_bin) is None:
        return None
    n_word = 5 if threshold >= 0.7 else 4 if threshold >= 0.6 else 3 if threshold >= 0.5 else 2
    tmp = tempfile.mkdtemp(prefix="cdhit_")
    fa, out = os.path.join(tmp, "in.fa"), os.path.join(tmp, "out")
    try:
        with open(fa, "w") as f:
            for i, s in enumerate(seqs):
                f.write(f">s{i}\n{clean_seq(s, CFG['max_prot_len'])}\n")
        subprocess.run(
            [cdhit_bin, "-i", fa, "-o", out, "-c", str(threshold),
             "-n", str(n_word), "-G", "1", "-d", "0", "-M", "0", "-T", "0"],
            check=True, capture_output=True,
        )
        assign: Dict[int, int] = {}
        cid = -1
        with open(out + ".clstr") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">Cluster"):
                    cid += 1
                elif ">s" in line:
                    idx = int(line.split(">s")[1].split("...")[0])
                    assign[idx] = cid
        print(f"[cdhit] {len(seqs)} sequences -> {cid + 1} clusters @ {threshold:.0%}")
        return [assign[i] for i in range(len(seqs))]
    except Exception as e:
        print(f"[cdhit] failed ({e}); falling back to internal clustering")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cluster_sequences_greedy(seqs: List[str], threshold: float) -> List[int]:
   
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    assign = [-1] * len(seqs)
    reps: List[int] = []
    for i in order:
        placed = False
        for c, r in enumerate(reps):
            if sequence_identity(seqs[i], seqs[r]) >= threshold:
                assign[i] = c
                placed = True
                break
        if not placed:
            assign[i] = len(reps)
            reps.append(i)
    print(f"[cluster] {len(seqs)} sequences -> {len(reps)} clusters "
          f"@ {threshold:.0%} (greedy fallback)")
    return assign


def cluster_sequences(seqs: List[str], threshold: float, cfg: Dict) -> List[int]:
    
    got = cluster_sequences_cdhit(seqs, threshold, cfg["cdhit_bin"])
    return got if got is not None else cluster_sequences_greedy(seqs, threshold)
