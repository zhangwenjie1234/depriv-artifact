# Ada-VFed Paper-Literal Path

This folder is the only implementation area for `--adavfed`. It implements the
paper's Algorithm 2, 3, and 4 training structure without mixing with the
repository's `defense_all`, `ppdl`, `rl`, or `acvfl` branches.

## Implemented

- Input-layer clipped Gaussian stochastic gates:
  `Z = clamp(mu + Normal(0, tau^2), 0, 1)`.
- Paper Eq. (6) regularization:
  - `lambda_1 / 2 * sum(max(0, ||h(x * Z)||_2 - C))`
  - `lambda_2 / 2 * sum(Phi(mu / tau))`
- Algorithm 2 dynamic forward noise:
  - Laplacian Score per release dimension.
  - `epsilon_d = D * w_d * epsilon`.
  - `sigma_d = 2 * C * sqrt(2 * ln(1.25 / delta)) / epsilon_d`.
- Algorithm 4 communication boundary:
  - The active party receives detached noisy embeddings.
  - Passive bottom models receive only noisy returned gradients plus the
    regularization gradient.
  - Ada-VFed parameters are updated once per step.
- Paper defaults are intentionally kept in `adavfed/config.py`, not exposed
  as legacy CLI knobs.

## Not Yet Claimed

This is a paper-literal experimental mechanism path. It does not yet implement
a strict, independently verified heterogeneous Gaussian RDP accountant for the
paper's Theorem 1. In particular, Algorithm 2 applies the paper epsilon to both
the forward and backward pseudocode releases. The paper does not specify the
missing calibration from a final composed epsilon to `sigma_*`, `sigma_g`, and
the per-dimension `sigma_d` values. Do not interpret `--epsilon` in this mode
as a verified final composed privacy guarantee.

## Run

```bash
python main.py --attack sign --dataset mnist --num_passive 1 --attack_id 0 --epochs 10 --batch_size 128 --lr_active 0.05 --lr_passive 0.05 --epsilon 1.0 --adavfed --run_seeds 42
```

For an ablation, edit the three module switches and the two Eq. (6)
coefficients directly in `adavfed/config.py`; no AdaVFed experiment parameters
are added to `main.py`.

For CIFAR-10, use the existing dataset/model optimization settings and add
`--adavfed`. Ada-VFed-specific paper defaults remain isolated in
`adavfed/config.py` until the mechanism is validated.
