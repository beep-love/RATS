# RATS: RL-Adaptive Triplet Sampling for Deep Metric Learning

Official code for **"RATS: RL-Adaptive Triplet Sampling for Deep Metric Learning"**
(B. S. Regmi, C. Anutariya, C. Kingkan), *IEEE Access*.

[Paper](PAPER_LINK_HERE) · [Checkpoints](CHECKPOINTS_LINK_HERE)

RATS trains a small actor-critic agent that, at the end of each observation window,
decides how to split the sampling weight between the **Hard** and **Semi-Hard**
triplet regions, defined by the triplet hardness `h = d_an - d_ap`. The agent acts
through a one-dimensional "seesaw" on two region logits. Its reward is computed
entirely from in-batch geometric statistics, so no validation split is held out and
the encoder trains on 100% of the training data. The triplet loss itself is unchanged.
An optional **Joint** variant also adapts the margin.

## Repository layout

```
configs/                 one config per benchmark (RATS Sampling-Only by default)
scripts/
  run_train.sh           example: run one training experiment from a dataset config
  run_eval.sh            example: evaluate one trained model
train.py                 training entry point
rl_miner.py              hardness regions, region weighting, actor-critic agent, reward
losses.py  metrics.py  sampler.py  model.py
*_dataset.py             dataset loaders
evaluate.py  evaluate_veriwild.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch should match your CUDA version; see <https://pytorch.org> if the pinned build
does not install on your machine. The experiments in the paper used a single NVIDIA
A6000 GPU.

## Data

Download the datasets from their official sources (they are not redistributed here)
and place them under `data/`:

```
data/
├── CUB_200/CUB_200_2011/          # CUB-200-2011
├── CAR196/                        # CARS-196
├── Stanford_Online_Products/      # SOP
└── VeRi-Wild/                     # VeRi-Wild (includes train_test_split/)
```

| Dataset | Source |
|---|---|
| CUB-200-2011 | <https://www.vision.caltech.edu/datasets/cub_200_2011/> |
| CARS-196 | <https://web.archive.org/web/20230405013536/https://ai.stanford.edu/~jkrause/cars/car_dataset.html> |
| Stanford Online Products | <https://cvgl.stanford.edu/projects/lifted_struct/> |
| VeRi-Wild | <https://github.com/PKU-IMRE/VERI-Wild> |

Class splits follow the paper: first half of the classes for training and second half
for testing on CUB-200 / CARS-196 / SOP. On VeRi-Wild, 20% of the training identities
are held out as the evaluation split (`train_val_split: 0.8`) and the official small,
medium, and large test splits are used for final results only.

## Training

All settings from the paper are fixed in `configs/<dataset>.yaml` (`cub200`, `cars196`,
`sop`, `veriwild`). The default config is RATS (Sampling-Only). To run one experiment:

```bash
bash scripts/run_train.sh cub200        # dataset, optional seed (default 1)
bash scripts/run_train.sh sop 2         # seed 2
```

This calls `train.py` with the dataset config. The direct equivalent is:

```bash
python train.py --config configs/cub200.yaml --set seed=1
```

Any config value can be changed for a single run with `--set KEY=VALUE` (nested keys use
a dot), or permanently by editing the YAML:

```bash
# a different RL algorithm or region step
bash scripts/run_train.sh cars196 1 --set rl.method=A2C
bash scripts/run_train.sh cub200 1 --set rl.region_step=0.15 --set epochs=60

# Joint variant (adaptive margin)
EXP_NAME=rats_joint bash scripts/run_train.sh sop 1 --set rl.adaptive_margin=True

# a static baseline instead of RATS (batch_hard | semi_hard | distance_weighted)
EXP_NAME=batch_hard bash scripts/run_train.sh sop 1 \
    --set rl.enabled=False --set sampling_strategy=batch_hard
```

Logs are written to `runs/<dataset>/<EXP_NAME>/seed<n>` (TensorBoard) and checkpoints to
`checkpoints/<dataset>/<EXP_NAME>/seed<n>`, with `EXP_NAME` defaulting to `rats`. The
paper reports the mean over seeds 1, 2 and 3.

## Evaluation

```bash
bash scripts/run_eval.sh cub200 1        # dataset, optional seed (default 1)
bash scripts/run_eval.sh veriwild 1      # small / medium / large test splits
```

The script evaluates the newest `best_model_*.pth` in `checkpoints/<dataset>/<EXP_NAME>/seed<n>`.
To evaluate another run or a downloaded checkpoint:

```bash
EXP_NAME=batch_hard bash scripts/run_eval.sh sop 1
CKPT=path/to/best_model.pth bash scripts/run_eval.sh cars196
```

Under the hood it calls `python evaluate.py --dataset {CUB200,CARS196,SOP} --checkpoint PATH`
or `python evaluate_veriwild.py --data_root data/VeRi-Wild --checkpoint PATH`.

Reported values for CUB-200, CARS-196 and SOP are those of the epoch with the highest
test R@1, applied to all methods alike. For VeRi-Wild the checkpoint is selected on the
held-out evaluation split and results are reported on the three test splits.

## Results

R@1 (%), mean over three seeds. Full tables (R@K, NMI, mAP, standard deviations) are in
the paper.

| Method | CUB-200 | CARS-196 | SOP | VeRi-Wild S | VeRi-Wild M | VeRi-Wild L |
|---|---|---|---|---|---|---|
| Distance-Weighted | 59.8 | 83.8 | 77.9 | 64.9 | 56.7 | 45.9 |
| PADS (published) | 64.0 | 79.9 | 74.8 | n/a | n/a | n/a |
| **RATS (Sampling-Only)** | 57.8 | 80.8 | 77.0 | **66.8** | **59.3** | **48.1** |

RATS is ahead of the static samplers on VeRi-Wild, ahead of PADS on SOP, and
comparable to PADS on CARS-196. A well-tuned static prior (Distance-Weighted) is
stronger on CUB-200 and CARS-196, where the reward's margin term never activates; see
the discussion in the paper.

## Implementation notes

- The reported experiments use **deterministic region weighting** (a weighted
  batch-all triplet loss, `rl.adaptive_sampling_strategy: weighting`), not random
  sampling of triplets. The `sampling` option in `rl_miner.py` is an unused alternative.
- The reward `R_t` is computed from the window that has just ended, before the new
  action takes effect, so each action is credited with a one-window delay (Sec. IV-E of
  the paper). This matches the results reported.
- Hyperparameters for CUB-200, CARS-196, and SOP were explored on the test set, so
  results on these three benchmarks are not independent held-out estimates; only the
  VeRi-Wild test splits are (Sec. V-D).
- Runs use fixed seeds and deterministic cuDNN kernels; small numerical differences
  across GPUs and library versions are expected.


## Citation

```bibtex
@article{regmi2026rats,
  title   = {{RATS}: {RL}-Adaptive Triplet Sampling for Deep Metric Learning},
  author  = {Regmi, Biplav S. and Anutariya, Chutiporn and Kingkan, Cherdsak},
  journal = {IEEE Access},
  year    = {2026},
  doi     = {DOI_HERE}
}
```

## License

Released under the MIT License (see `LICENSE`). The datasets are subject to their own
licenses and terms of use.