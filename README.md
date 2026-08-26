# DePriV Artifact

This repository contains the implementation and evaluation drivers for
**DePriV**, a bidirectional differentially private training framework for
heterogeneous multiparty vertical federated learning (VFL). DePriV protects
the forward representations sent to the active party and the backward
gradients returned to passive parties under a composed privacy budget.

The artifact supports MNIST, Fashion-MNIST, CIFAR-10, and Criteo; heterogeneous
bottom models; label and feature inference attacks; defense baselines; and the
ablation, sensitivity, dynamics, and runtime experiments used in the paper.

## Repository layout

```text
.
├── main.py                         # Main training and attack entry point
├── attackers/                      # VFL protocol, DePriV controller, and attacks
├── utils/                          # Models, datasets, DP, accounting, and priors
├── adavfed/                        # AdaVFed baseline implementation
├── run_*.py                        # Paper experiment and aggregation drivers
├── prepare_criteo_private.py
├── prepare_criteo_1tb_aux.py
├── process_criteo_1tb_aux.py
└── environment.yml
```

Experiment drivers preserve each command, complete console log, parsed
metrics, and aggregate summaries below their output directory. Existing valid
runs are reused unless a driver's rerun or force option is supplied.

## Environment

The experiments were run on Linux with Python 3.10, PyTorch 2.8, CUDA, and
Opacus 1.5. Create the Conda environment with:

```bash
conda env create -f environment.yml
conda activate vfl
```

## Datasets

Image datasets are downloaded by TorchVision into `dataset/`. By default,
DePriV uses the following disjoint public auxiliary datasets:

| Private task | Public auxiliary data |
| --- | --- |
| MNIST | EMNIST Letters |
| Fashion-MNIST | KMNIST |
| CIFAR-10 | CIFAR-100 |
| Criteo Display Advertising Challenge | Criteo 1TB Click Logs |

The public data are used only for structural calibration. Dataset files are
not distributed in this repository.

For Criteo, place the official Display Advertising Challenge archive at
`dataset/criteo_raw/dac.tar.gz`, then prepare disjoint labeled splits:

```bash
python prepare_criteo_private.py
```

Prepare the public auxiliary subset and its deterministic encoding with:

```bash
python prepare_criteo_1tb_aux.py
python process_criteo_1tb_aux.py
```

Use `--local_source PATH` with `prepare_criteo_1tb_aux.py` when the selected
official Parquet shard has already been downloaded.

## Quick start

`--num_passive` is the number of passive parties, so `--num_passive 4`
corresponds to M=5 total parties. The following command runs DePriV on MNIST
with the paper's fixed risk threshold:

```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --dataset mnist \
  --num_passive 4 \
  --attack sign \
  --attack_id 1 \
  --epochs 10 \
  --batch_size 128 \
  --epsilon 1.0 \
  --rl --hg \
  --rl_proxy_threshold 40 \
  --run_seeds 42
```

Replace `mnist` with `fashionmnist` or `cifar10` to run another image task.
Multiple trials can be launched in one command with, for example,
`--run_seeds 42 43 44 45 46`. Use `python main.py --help` for the complete set
of model, privacy, attack, and controller options.

## Attacks and baselines

The main entry point supports Sign, Cluster, PMC, AMC, and feature
reconstruction attacks. For example:

```bash
# Cluster attack on released embeddings
CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --dataset mnist --num_passive 4 --attack cluster --use_emb \
  --attack_id 1 --epochs 10 --batch_size 128 --run_seeds 42

# AMC stress test with 10 randomly sampled auxiliary labels
CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --dataset mnist --num_passive 4 --attack amc --attack_id 1 \
  --completion_aux_num_labels 10 --amc_stratified_aux 0 \
  --amc_pseudo_warmup_epochs 0 --epsilon 8.0 --rl --hg \
  --run_seeds 42
```

Defense modes are selected with the following flags:

| Method | Flag |
| --- | --- |
| DePriV | `--rl --hg` |
| DP-TPSL | `--defense_all` |
| DP-PPDL | `--ppdl` |
| ACVFL | `--acvfl --acvfl_bidirectional 1` |
| AdaVFed | `--adavfed` |

All methods use the same VFL interface. The final privacy budget is reported
by the RDP accountant at the end of each run.

## Reproducing the main evaluations

The drivers below run all configured dataset, seed, and method combinations
and save raw logs plus CSV and JSON summaries.

```bash
# AMC evaluation: three datasets, five methods, seeds 42--44, M=5, epsilon=8
python -u run_amc_attack.py --gpu 0

# Fixed risk-threshold sweep with AMC
python -u run_risk_threshold_amc_sensitivity.py --gpu 0

# Training dynamics at the default threshold
python -u run_rl_control_convergence.py \
  --gpu 0 --risk_threshold 40 --display_risk_threshold 40

# Adaptive and static backward control
python -u run_adaptive_fixed_all.py --gpu 0

# Runtime and peak-memory measurements
python -u run_runtime_overhead.py --gpu 0
```

Sensitivity drivers expose their evaluated values through command-line
arguments. Examples include:

```bash
python -u run_budget_split_sensitivity.py --gpu 0 --seeds 42 43 44
python -u run_clip_quantile_sensitivity.py --seeds 42 43 44
```

Pass `--help` to any driver before a long run. The drivers are resumable and
store outputs below `results/` unless another output directory is supplied.

## Reproducibility notes

- Paper experiments use the seeds stated by their corresponding driver.
- Each run records the exact command and complete log in its run directory.
- Generated datasets, logs, model outputs, and temporary files should not be
  committed. Curated aggregate CSV/JSON files may be included under a small
  `artifact_results/` directory when needed for figure reproduction.
- Do not expose a personally identifying repository URL in an anonymous
  submission. Use the conference's anonymous artifact mechanism during
  double-blind review.

## License

See [LICENSE](LICENSE). Third-party code and datasets remain subject to their
respective licenses and terms of use.
