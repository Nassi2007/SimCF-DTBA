from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch

from .config import CFG, DEVICE, set_seed
from .data import df_to_triples, load_dataframe, make_loader
from .evaluate import evaluate
from .models import build_model
from .splits import build_splits, filter_pretrain_corpus
from .train import run_curriculum, run_vanilla_training


def run_single_seed(dataset: str, split: str, seed: int, cfg: Dict = CFG,
                    variant: str = "simcf") -> Dict:
   
    set_seed(seed)

    target_path = cfg["davis_path"] if dataset == "davis" else cfg["kiba_path"]
    df = load_dataframe(target_path)

    tr_df, va_df, te_df, held_d, held_p, meta = build_splits(df, split, seed, cfg)

    pre_df = load_dataframe(cfg["bindingdb_path"], label_agnostic=True)
    if split == "random":
        pre_use = pre_df
    else:
        pre_use = filter_pretrain_corpus(pre_df, split, te_df, held_d, held_p, cfg)

    need_sc = (split == "scaffold_drug")
    tr = df_to_triples(tr_df, need_sc)
    va = df_to_triples(va_df, need_sc)
    te = df_to_triples(te_df, need_sc)
    pre = df_to_triples(pre_use)

    rng = random.Random(seed)
    idx = list(range(len(pre)))
    rng.shuffle(idx)
    n_pv = max(1, int(cfg["pretrain_val_frac"] * len(pre)))
    pre_val = [pre[i] for i in idx[:n_pv]]
    pre_train = [pre[i] for i in idx[n_pv:]]

    model = build_model(variant, cfg)
    if variant == "simcf":
        run_curriculum(model,
                       make_loader(pre_train, True, seed),
                       make_loader(pre_val, False, seed),
                       make_loader(tr, True, seed),
                       make_loader(va, False, seed), cfg)
    else:
        run_vanilla_training(model,
                             make_loader(tr, True, seed),
                             make_loader(va, False, seed), cfg)

    metrics = evaluate(model, make_loader(te, False, seed),
                       label=f"{variant}/{dataset}/{split}/seed{seed}")
    metrics["meta"] = meta

    os.makedirs(cfg["out_dir"], exist_ok=True)
    ck = os.path.join(cfg["out_dir"], f"{variant}_{dataset}_{split}_seed{seed}.pt")
    torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                "metrics": metrics, "variant": variant}, ck)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return metrics


def summarise(results: List[Dict], keys=("ci", "mse", "pearson_r")) -> Dict:
     
    out = {}
    for k in keys:
        v = np.array([r[k] for r in results], dtype=np.float64)
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=0)), "values": v.tolist()}
    return out


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000,
                        alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n < 2:
        return (float("nan"), float("nan"))
    means = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def run_ablation(dataset: str, split: str, seeds: List[int], cfg: Dict = CFG) -> Dict:
    
    rows = {"vanilla": [], "simcf": []}
    for seed in seeds:
        for variant in ("vanilla", "simcf"):
            rows[variant].append(run_single_seed(dataset, split, seed, cfg, variant))

    out: Dict = {"dataset": dataset, "split": split, "seeds": seeds,
                 "per_seed": rows, "metrics": {}}

     
    print(f"{'Metric':7s} {'VanillaLM+MLP':>18s} {'SimCF-DTBA':>18s} "
          f"{'Delta':>11s} {'paired 95% CI':>22s}")
     
    for key in ("ci", "mse"):
        v = np.array([r[key] for r in rows["vanilla"]], dtype=np.float64)
        s = np.array([r[key] for r in rows["simcf"]], dtype=np.float64)
        d = s - v
        lo, hi = paired_bootstrap_ci(d, seed=abs(hash((dataset, split, key))) % (2 ** 31))
        out["metrics"][key] = {
            "vanilla_mean": float(v.mean()), "vanilla_std": float(v.std(ddof=0)),
            "simcf_mean": float(s.mean()), "simcf_std": float(s.std(ddof=0)),
            "delta_mean": float(d.mean()), "delta_std": float(d.std(ddof=0)),
            "delta_per_seed": d.tolist(), "paired_bootstrap_ci95": [lo, hi],
        }
        print(f"{key.upper():7s} {v.mean():>10.4f}+/-{v.std(ddof=0):<6.4f} "
              f"{s.mean():>10.4f}+/-{s.std(ddof=0):<6.4f} "
              f"{d.mean():>+11.4f} [{lo:+.4f}, {hi:+.4f}]")
    
     
    os.makedirs(cfg["out_dir"], exist_ok=True)
    p = os.path.join(cfg["out_dir"], f"ablation_{dataset}_{split}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=str)
     
    return out


def run_all(cfg: Dict = CFG, seeds: List[int] | None = None) -> Dict:
    
    seeds = seeds or cfg["seeds"]
    plan = [
        ("Tables 4-5 (cold-split, He et al. protocol)",
         [(d, s, False) for d in ("davis", "kiba") for s in ("cold_drug", "cold_target")]),
        ("Table 6 (random split)",
         [(d, "random", False) for d in ("davis", "kiba")]),
        ("Table 7 (architectural ablation)",
         [(d, s, True) for d in ("davis", "kiba")
          for s in ("cold_drug", "cold_target", "random")]),
        ("Table 8 (stricter protocols)",
         [(d, s, False) for d in ("davis", "kiba")
          for s in ("scaffold_drug", "seqid_target")]),
    ]

    everything: Dict = {}
    for title, jobs in plan:
        print("\n\n" + "#" * 80)
        print(f"### {title}")
        print("#" * 80)
        for dataset, split, is_abl in jobs:
            tag = f"{dataset}/{split}" + ("/ablation" if is_abl else "")
            try:
                if is_abl:
                    everything[tag] = run_ablation(dataset, split, seeds, cfg)
                else:
                    res = [run_single_seed(dataset, split, s, cfg, "simcf") for s in seeds]
                    everything[tag] = {"per_seed": res, "summary": summarise(res)}
                    sm = everything[tag]["summary"]
                    print(f"\n[{tag}] " + "  ".join(
                        f"{k.upper()}={sm[k]['mean']:.4f}+/-{sm[k]['std']:.4f}"
                        for k in ("ci", "mse", "pearson_r")))
            except Exception as e:
                print(f"[ERROR] {tag} failed: {type(e).__name__}: {e}")
                everything[tag] = {"error": f"{type(e).__name__}: {e}"}

    os.makedirs(cfg["out_dir"], exist_ok=True)
    p = os.path.join(cfg["out_dir"], "all_results.json")
    with open(p, "w") as f:
        json.dump(everything, f, indent=2, default=str)
    print(f"\n[save] complete results -> {p}")
    print("\nThese are the numbers to report. Where they differ from the current")
    print("Tables 4-8, update the tables and every claim that depends on them.")
    return everything
