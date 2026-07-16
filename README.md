# SimCF-DTBA

Similarity-Grounded Contrastive Fusion for Cold-Split Drug–Target Binding
Affinity Prediction — reference implementation for the paper.

## Install

Requires Python ≥ 3.9 and a CUDA GPU with ≈ 16 GB memory (frozen ESM-2 650M).

```bash
git clone https://github.com/<user>/simcf-dtba.git
cd simcf-dtba
pip install -r requirements.txt
```

Backbones (ChemBERTa-2, ESM-2 650M) download automatically on first run.

Optional: [CD-HIT](https://github.com/weizhongli/cdhit) for `--split
seqid_target`. Without it, an equivalent clustering in `similarity.py` is used.

## Data

Put three CSVs in `data/`:

| File | Source |
|---|---|
| `davis.csv` | [NHGNN-DTA](https://github.com/hehh77/NHGNN-DTA) → `Data/davis.7z` |
| `kiba.csv` | [NHGNN-DTA](https://github.com/hehh77/NHGNN-DTA) → `Data/kiba.7z` |
| `bindingdb_pretrain.csv` | [BindingDB](https://www.bindingdb.org) |

Format:

```csv
compound_iso_smiles,target_key,target_sequence,affinity
CC(=O)Oc1ccccc1C(=O)O,AAK1,MKKFFDSRREQGGSGLGSGSSGGGG...,5.0
```

BindingDB needs only `compound_iso_smiles` and `target_sequence` — pre-training
is label-agnostic. Paths are set in `simcf_dtba/config.py`.

Use the Davis/KIBA files from NHGNN-DTA: `splits.py` reproduces He et al.'s
split protocol exactly, so the same files give the same partitions as the
published baselines.

## Run

```bash
python run.py --run_all                              # every table
```

Or individually:

| Table | Command |
|---|---|
| 4 — Davis cold-split | `python run.py --dataset davis --split cold_drug`<br>`python run.py --dataset davis --split cold_target` |
| 5 — KIBA cold-split | `python run.py --dataset kiba --split cold_drug`<br>`python run.py --dataset kiba --split cold_target` |
| 6 — random split | `python run.py --dataset davis --split random` |
| 7 — ablation | `python run.py --dataset davis --split cold_drug --ablation` |
| 8 — stricter protocols | `python run.py --dataset davis --split scaffold_drug`<br>`python run.py --dataset davis --split seqid_target` |

(repeat with `--dataset kiba` where applicable)

Runs use seeds 42, 123, 456 and report mean ± std. Results and checkpoints are
written to `./simcf_runs`.

If you hit CUDA OOM, lower `batch_size` or `max_prot_len` in `config.py`.

## Code

```
simcf_dtba/
├── config.py       hyperparameters and paths
├── chem.py         canonical SMILES, Murcko scaffolds
├── similarity.py   Tanimoto and BLOSUM62 oracles, CD-HIT
├── splits.py       cold / random / scaffold / sequence-identity splits
├── data.py         CSV loading, DataLoader
├── models.py       adapters, encoders, Perceiver IO, SimCF-DTBA, VanillaLM+MLP
├── losses.py       hybrid intra-modal loss, soft-SupCon, Huber
├── train.py        four-stage freeze curriculum
├── evaluate.py     CI, MSE, Pearson r
├── experiments.py  runs, ablation, full reproduction
└── cli.py          argument parsing
```

## Citation

```bibtex
@article{aleb_simcf_dtba,
  title  = {Similarity-Grounded Contrastive Fusion for Cold-Split
            Drug--Target Binding Affinity Prediction},
  author = {Aleb, Nassima},
  year   = {2026}
}
```

Split protocol and baseline values: He et al., *NHGNN-DTA*, Bioinformatics
39(6):btad355, 2023.

## License

MIT
