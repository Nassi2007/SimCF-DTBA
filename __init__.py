"""
SimCF-DTBA
==========
Similarity-Grounded Contrastive Fusion for Cold-Split Drug-Target Binding
Affinity Prediction.

Reference implementation accompanying the manuscript.

Modules
-------
config       hyperparameters, device, CSV schema, seeding
chem         canonical SMILES keys, Bemis-Murcko scaffolds, sequence cleaning
similarity   Tanimoto / BLOSUM62 oracles, agreement rule, CD-HIT clustering
splits       He et al. (2023) cold-split protocol + stricter protocols (Sec. 4.8)
data         CSV loading, column normalisation, Dataset / DataLoader
models       adapters, frozen encoders, Perceiver IO, SimCF-DTBA, VanillaLM+MLP
losses       hybrid intra-modal loss, soft-SupCon cross-modal loss, Huber
train        four-stage sequential freeze curriculum
evaluate     Concordance Index, MSE, Pearson r
experiments  single runs, architectural ablation, full table reproduction
cli          command-line interface
"""

__version__ = "1.0.0"

from .config import CFG, DEVICE, set_seed
from .evaluate import concordance_index, evaluate
from .experiments import run_ablation, run_all, run_single_seed
from .models import SimCFDTBA, VanillaLMMLP, build_model
from .splits import build_splits, create_fold, create_fold_setting_cold

__all__ = [
    "CFG", "DEVICE", "set_seed",
    "SimCFDTBA", "VanillaLMMLP", "build_model",
    "create_fold", "create_fold_setting_cold", "build_splits",
    "concordance_index", "evaluate",
    "run_single_seed", "run_ablation", "run_all",
]
