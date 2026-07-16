"""
Training objectives.

  HybridLoss      intra-modal manifold calibration        (Sec. 3.5.1)
  SoftSupConLoss  cross-modal alignment + Steps 3-4       (Sec. 3.5, 3.5.2)
  huber_loss      supervised regression objective
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .similarity import agreement_weights


class HybridLoss(nn.Module):
    """
    Intra-modal manifold calibration (Sec. 3.5.1):

        L = lambda * L_margin + (1 - lambda) * L_InfoNCE

    Label-agnostic: supervision comes only from the similarity oracle S, never
    from binding labels.  Used for Stage 1 (drug, Tanimoto) and Stage 2
    (protein, BLOSUM62).

    The InfoNCE term masks the diagonal with a large finite negative rather
    than -inf, and gates positives with torch.where, so that empty-positive
    rows cannot produce -inf * 0 = NaN.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.tau = cfg["temperature"]
        self.top_k = cfg["top_k"]
        self.margin = cfg["margin"]
        self.sim_thr = cfg["sim_threshold"]
        self.lam = cfg["lambda_hyb"]

    @staticmethod
    def pool(H: torch.Tensor) -> torch.Tensor:
        """Mean-pool then safe L2-normalise (avoids 0/0)."""
        h = H.mean(dim=1)
        return h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def forward(self, H: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        h = self.pool(H)
        N, dev = h.size(0), h.device
        cos = (h @ h.T).clamp(-1.0, 1.0)
        dist = (1.0 - cos).clamp(min=0.0)

        # ---- margin term -------------------------------------------------
        y = (S > self.sim_thr).float()
        L_m = (y * dist.pow(2) + (1.0 - y) * F.relu(self.margin - dist).pow(2)).mean()
        if N < 2:
            return self.lam * L_m

        # ---- InfoNCE term ------------------------------------------------
        diag = torch.eye(N, device=dev, dtype=torch.bool)
        sim = (cos / self.tau).masked_fill(diag, -1e9)
        log_sm = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        k = min(self.top_k, N - 1)
        thr = S.topk(k, dim=1).values[:, -1:] - 1e-8
        pos = (S >= thr) & ~diag

        pos_logprob = torch.where(pos, log_sm, torch.zeros_like(log_sm))
        pos_sum = pos.float().sum(dim=1)
        valid = pos_sum > 0
        if not valid.any():
            return self.lam * L_m
        L_i = (-pos_logprob.sum(dim=1) / pos_sum.clamp(min=1e-8))[valid].mean()
        return self.lam * L_m + (1.0 - self.lam) * L_i


class SoftSupConLoss(nn.Module):
    """
    Cross-modal alignment with explicit negative construction
    (Sec. 3.5.2 together with Sec. 3.5 Steps 3-4).  Used for Stage 3.

    Step 3 - positives:
        P(i) = TopK_{j != i} w_ij,  where w = A(S_d, S_p)          (Eq. 14)

    Step 4 - negatives:
        Hard negatives are pairs with |S_d,ij - S_p,ij| > delta, which have
        w_ij ~ 0 by construction and force the model to distinguish
        cross-modal binding specificity from unimodal structural similarity.
        Random negatives are sampled uniformly from the remainder of the
        batch, at random:hard = `hard_random_ratio` (paper: 1:3).

    Only P(i) and the sampled negatives enter the partition function; all
    other batch entries are masked out.  If a row has no hard negatives in a
    given batch, the full random pool is used so the anchor still has a valid
    partition function.  Setting hard_neg_thr >= 1 disables Step 4 and
    recovers a plain full-batch denominator.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.tau = cfg["temperature"]
        self.rule = cfg["agreement"]
        self.top_k = cfg["top_k"]
        self.delta = cfg["hard_neg_thr"]
        self.ratio = cfg["hard_random_ratio"]

    def _build_masks(self, S_d: torch.Tensor, S_p: torch.Tensor,
                     w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (positive_mask, selected_negative_mask) for the batch."""
        N, dev = w.size(0), w.device
        diag = torch.eye(N, device=dev, dtype=torch.bool)

        # ---- Step 3: Top-K positives by agreement weight (Eq. 14) --------
        k = min(self.top_k, N - 1)
        thr = w.masked_fill(diag, -1.0).topk(k, dim=1).values[:, -1:]
        pos = (w >= thr.clamp(min=1e-12)) & ~diag

        # ---- Step 4: hard + random negative construction -----------------
        cand = (~pos) & (~diag)
        hard = ((S_d - S_p).abs() > self.delta) & cand
        rand_pool = cand & (~hard)

        sel = torch.zeros_like(cand)
        for i in range(N):
            h_idx = hard[i].nonzero(as_tuple=True)[0]
            r_idx = rand_pool[i].nonzero(as_tuple=True)[0]
            n_hard = int(h_idx.numel())
            if n_hard > 0:
                sel[i, h_idx] = True
                n_rand = min(self.ratio * n_hard, int(r_idx.numel()))
            else:
                n_rand = int(r_idx.numel())
            if n_rand > 0:
                perm = torch.randperm(int(r_idx.numel()), device=dev)[:n_rand]
                sel[i, r_idx[perm]] = True
        return pos, sel

    def forward(self, z: torch.Tensor, S_d: torch.Tensor, S_p: torch.Tensor) -> torch.Tensor:
        N = z.size(0)
        if N < 2:
            return z.sum() * 0.0
        dev = z.device
        diag = torch.eye(N, device=dev, dtype=torch.bool)

        w = agreement_weights(S_d, S_p, self.rule).masked_fill(diag, 0.0)
        pos, neg = self._build_masks(S_d, S_p, w)

        keep = pos | neg                        # partition support P(i) u N(i)
        w_pos = w * pos.float()
        row = w_pos.sum(1, keepdim=True)
        valid = (row.squeeze(1) > 1e-8) & keep.any(dim=1)
        if not valid.any():
            return z.sum() * 0.0
        w_hat = w_pos / row.clamp(min=1e-8)     # row-normalised soft targets

        z_n = F.normalize(z, dim=-1)
        sim = (z_n @ z_n.T) / self.tau
        sim = sim.masked_fill(~keep, -1e9)      # restrict the denominator
        log_d = torch.logsumexp(sim, dim=1, keepdim=True)
        num = (w_hat * torch.exp(sim - log_d)).sum(1)
        return (-torch.log(num.clamp(min=1e-8)))[valid].mean()


def huber_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Supervised regression objective for Stage 4 (delta = 1.0)."""
    return F.huber_loss(y_hat, y, delta=1.0)
