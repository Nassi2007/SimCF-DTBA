from __future__ import annotations
import copy
import math
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from .config import DEVICE
from .losses import HybridLoss, SoftSupConLoss, huber_loss
from .models import MLPHead
from .similarity import blosum62_matrix, tanimoto_matrix

def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(True)


def make_opt(param_groups, cfg: Dict):
    opt = torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["cosine_tmax"], eta_min=cfg["cosine_eta_min"])
    return opt, sch


def train_stage(name, model, opt, sch, train_loader, val_loader, loss_fn, cfg) -> float:
     
    best_val, wait, best_state = float("inf"), 0, None
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        t_losses = []
        for i, (smiles, seqs, labels) in enumerate(train_loader):
            labels = labels.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, smiles, seqs, labels)
            if not torch.isfinite(loss):
                print(f"[{name}] epoch {epoch} batch {i}: non-finite loss, skipped")
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            opt.step()
            t_losses.append(loss.item())
        sch.step()

        model.eval()
        v_losses = []
        with torch.no_grad():
            for smiles, seqs, labels in val_loader:
                labels = labels.to(DEVICE)
                loss = loss_fn(model, smiles, seqs, labels)
                if torch.isfinite(loss):
                    v_losses.append(loss.item())

        t = float(np.mean(t_losses)) if t_losses else float("nan")
        v = float(np.mean(v_losses)) if v_losses else float("nan")
        print(f"[{name}] epoch {epoch:3d} | train={t:.4f}  val={v:.4f}")

        if math.isfinite(v) and v < best_val - cfg["min_delta"]:
            best_val, wait = v, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"[{name}] converged (early stop) at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def build_loss_fns(cfg: Dict):
    hybrid = HybridLoss(cfg).to(DEVICE)
    soft = SoftSupConLoss(cfg).to(DEVICE)

    def drug_loss(model, smiles, seqs, labels):
        return hybrid(model.drug(smiles), tanimoto_matrix(smiles).to(DEVICE))

    def prot_loss(model, smiles, seqs, labels):
        return hybrid(model.prot(seqs), blosum62_matrix(seqs).to(DEVICE))

    def fusion_loss(model, smiles, seqs, labels):
        S_d = tanimoto_matrix(smiles).to(DEVICE)
        S_p = blosum62_matrix(seqs).to(DEVICE)
        z = model.fusion(model.drug(smiles), model.prot(seqs))
        return soft(z, S_d, S_p)

    def pred_loss(model, smiles, seqs, labels):
        return huber_loss(model(smiles, seqs), labels)

    return drug_loss, prot_loss, fusion_loss, pred_loss


def run_curriculum(model, pre_train_loader, pre_val_loader,
                   sup_train_loader, sup_val_loader, cfg) -> None:
    drug_loss, prot_loss, fusion_loss, pred_loss = build_loss_fns(cfg)
    alr = cfg["lr"] * cfg["adapter_lr_ratio"]
    freeze(model)
    unfreeze(model.drug.adapter)
    opt, sch = make_opt([{"params": model.drug.adapter.parameters(), "lr": cfg["lr"]}], cfg)
    train_stage("S1-Drug", model, opt, sch, pre_train_loader, pre_val_loader, drug_loss, cfg)
    freeze(model.drug.adapter)
    unfreeze(model.prot.adapter)
    opt, sch = make_opt([{"params": model.prot.adapter.parameters(), "lr": cfg["lr"]}], cfg)
    train_stage("S2-Prot", model, opt, sch, pre_train_loader, pre_val_loader, prot_loss, cfg)
    freeze(model.prot.adapter)
    unfreeze(model.fusion)
    unfreeze(model.drug.adapter)
    unfreeze(model.prot.adapter)
    opt, sch = make_opt([
        {"params": model.fusion.parameters(), "lr": cfg["lr"]},
        {"params": model.drug.adapter.parameters(), "lr": alr},
        {"params": model.prot.adapter.parameters(), "lr": alr},
    ], cfg)
    train_stage("S3-Fusion", model, opt, sch, pre_train_loader, pre_val_loader, fusion_loss, cfg)
    freeze(model.fusion)
    freeze(model.drug.adapter)
    freeze(model.prot.adapter)
    model.head = MLPHead(cfg).to(DEVICE)      
    unfreeze(model.head)
    unfreeze(model.drug.adapter)
    unfreeze(model.prot.adapter)
    opt, sch = make_opt([
        {"params": model.head.parameters(), "lr": cfg["lr"]},
        {"params": model.drug.adapter.parameters(), "lr": alr},
        {"params": model.prot.adapter.parameters(), "lr": alr},
    ], cfg)
    train_stage("S4-Pred", model, opt, sch, sup_train_loader, sup_val_loader, pred_loss, cfg)
    print("[curriculum] all four stages complete.")


def run_vanilla_training(model, sup_train_loader, sup_val_loader, cfg) -> None:
     
    _, _, _, pred_loss = build_loss_fns(cfg)
    freeze(model)
    unfreeze(model.head)
    opt, sch = make_opt([{"params": model.head.parameters(), "lr": cfg["lr"]}], cfg)
    train_stage("Vanilla-Pred", model, opt, sch, sup_train_loader, sup_val_loader, pred_loss, cfg)
    print("[vanilla] training complete.")
