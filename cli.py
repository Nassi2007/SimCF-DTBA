"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .config import CFG, DEVICE
from .experiments import run_ablation, run_all, run_single_seed, summarise
from .splits import SPLITS


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="simcf-dtba",
        description="SimCF-DTBA: Similarity-Grounded Contrastive Fusion for "
                    "Cold-Split Drug-Target Binding Affinity Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # cold-split results (Tables 4-5)
  simcf-dtba --dataset davis --split cold_drug
  simcf-dtba --dataset kiba  --split cold_target

  # random split (Table 6)
  simcf-dtba --dataset davis --split random

  # architectural ablation (Table 7): both arms, identical splits
  simcf-dtba --dataset davis --split cold_drug --ablation

  # stricter protocols (Table 8)
  simcf-dtba --dataset davis --split scaffold_drug
  simcf-dtba --dataset davis --split seqid_target

  # everything
  simcf-dtba --run_all
""")
    ap.add_argument("--dataset", choices=["davis", "kiba"])
    ap.add_argument("--split", choices=list(SPLITS))
    ap.add_argument("--variant", choices=["simcf", "vanilla"], default="simcf",
                    help="'vanilla' = VanillaLM+MLP ablation baseline (Sec. 4.6)")
    ap.add_argument("--ablation", action="store_true",
                    help="run BOTH arms on identical splits and report the paired "
                         "Delta table (Sec. 4.6, Table 7)")
    ap.add_argument("--run_all", action="store_true",
                    help="reproduce Tables 4, 5, 6, 7 and 8 end to end")
    ap.add_argument("--seeds", type=int, nargs="+", default=CFG["seeds"])
    ap.add_argument("--out_dir", default=CFG["out_dir"])
    ap.add_argument("--davis_path", default=None)
    ap.add_argument("--kiba_path", default=None)
    ap.add_argument("--bindingdb_path", default=None)
    ap.add_argument("--no_cache", action="store_true", help="disable embedding cache")
    return ap


def main(argv=None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)

    CFG["out_dir"] = args.out_dir
    for k in ("davis_path", "kiba_path", "bindingdb_path"):
        v = getattr(args, k)
        if v:
            CFG[k] = v
    if args.no_cache:
        CFG["cache_embeddings"] = False

    print(f"[env] device={DEVICE}  torch={torch.__version__}  seeds={args.seeds}")

    if args.run_all:
        run_all(CFG, args.seeds)
        return

    if not args.dataset or not args.split:
        ap.error("--dataset and --split are required unless --run_all is given")

    if args.ablation:
        run_ablation(args.dataset, args.split, args.seeds, CFG)
        return

    results = [run_single_seed(args.dataset, args.split, s, CFG, args.variant)
               for s in args.seeds]
    summary = {"variant": args.variant, "dataset": args.dataset, "split": args.split,
               "seeds": args.seeds, "per_seed": results}
    summary.update(summarise(results))

    print("\n" + "=" * 74)
    print(f"FINAL  variant={args.variant}  dataset={args.dataset}  split={args.split}")
    print("=" * 74)
    for k in ("ci", "mse", "pearson_r"):
        print(f"  {k.upper():9s} = {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}   "
              f"{np.round(summary[k]['values'], 4).tolist()}")

    os.makedirs(CFG["out_dir"], exist_ok=True)
    out = os.path.join(CFG["out_dir"],
                       f"summary_{args.variant}_{args.dataset}_{args.split}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[save] summary -> {out}")
    print("\nReport these numbers. If they differ from the current tables, update")
    print("the tables and the surrounding claims.")


if __name__ == "__main__":
    main()
