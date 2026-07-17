# SimCF-DTBA

Similarity-Grounded Contrastive Fusion for Cold-Split Drug–Target Binding
Affinity Prediction — reference implementation for the paper "Artificial
Intelligence in Drug Discovery: Accelerating Precision Medicine for Novel
Compounds and Targets via Contrastive Learning", *Artificial Intelligence in
Health*.

## Install

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and a CUDA GPU (~16 GB, for frozen ESM-2 650M).
Backbones (ChemBERTa-2, ESM-2) download automatically on first run.

Optional: [CD-HIT](https://github.com/weizhongli/cdhit) for `--split
seqid_target`. Without it, an equivalent clustering in `similarity.py` is
used automatically.

## Data

Put three CSVs in `data/`, each with columns
`compound_iso_smiles,target_key,target_sequence,affinity`
(`target_key`/`affinity` optional for the BindingDB file):

| File | Source |
|---|---|
| `data/davis.csv` | [NHGNN-DTA](https://github.com/hehh77/NHGNN-DTA) → `Data/davis.7z` |
| `data/kiba.csv` | [NHGNN-DTA](https://github.com/hehh77/NHGNN-DTA) → `Data/kiba.7z` |
| `data/bindingdb_pretrain.csv` | [BindingDB](https://www.bindingdb.org) |

Use the NHGNN-DTA Davis/KIBA files specifically: `splits.py` reproduces their
split protocol exactly, so the same files give the same train/val/test
partitions as the baseline values reported in the paper.

## Run

```bash
python run.py --run_all          # every table, all seeds
```

Or per table:

| Table | Command |
|---|---|
| 4–5 — cold-split | `python run.py --dataset davis --split cold_drug`<br>`python run.py --dataset davis --split cold_target` |
| 6 — random split | `python run.py --dataset davis --split random` |
| 7 — ablation | `python run.py --dataset davis --split cold_drug --ablation` |
| 8 — stricter protocols | `python run.py --dataset davis --split scaffold_drug`<br>`python run.py --dataset davis --split seqid_target` |

(repeat with `--dataset kiba`)

Uses seeds 42, 123, 456 by default; reports mean ± std. Results and
checkpoints are written to `./simcf_runs`. If you hit CUDA out-of-memory,
lower `batch_size` or `max_prot_len` in `simcf_dtba/config.py`.

## License

MIT
