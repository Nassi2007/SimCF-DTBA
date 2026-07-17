"""Evaluation metrics: Concordance Index, MSE, Pearson r  ."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .config import DEVICE


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    i, j = np.triu_indices(len(y_true), k=1)
    yt_i, yt_j, yp_i, yp_j = y_true[i], y_true[j], y_pred[i], y_pred[j]

    differ = yt_i != yt_j
    if not differ.any():
        return 0.5
    yt_i, yt_j = yt_i[differ], yt_j[differ]
    yp_i, yp_j = yp_i[differ], yp_j[differ]

    swap = yt_i < yt_j                      # normalise so yt_i > yt_j
    hi = np.where(swap, yp_j, yp_i)
    lo = np.where(swap, yp_i, yp_j)
    return float((hi > lo).sum() + 0.5 * (hi == lo).sum()) / float(len(hi))


@torch.no_grad()
def evaluate(model, loader, label: str = "test") -> Dict[str, float]:
    
    from scipy.stats import pearsonr

    model.eval()
    preds, trues = [], []
    for smiles, seqs, labels in loader:
        preds.append(model(smiles, seqs).cpu().numpy())
        trues.append(labels.numpy())
    y_pred, y_true = np.concatenate(preds), np.concatenate(trues)

    mse = float(np.mean((y_pred - y_true) ** 2))
    ci = concordance_index(y_true, y_pred)
    r = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 2 else float("nan")
    print(f"\n[{label}] CI={ci:.4f}  MSE={mse:.4f}  r={r:.4f}  (n={len(y_true)})")
    return {"ci": ci, "mse": mse, "pearson_r": r, "n": int(len(y_true))}
