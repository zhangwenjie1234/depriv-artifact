from __future__ import annotations

import math

import torch


def apply_paper_stochastic_gate(
    inputs: torch.Tensor,
    mu: torch.Tensor,
    tau: float,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the clipped Gaussian input gate from Ada-VFed Eq. (6)."""
    if tuple(mu.shape) != tuple(inputs.shape[1:]):
        raise ValueError(
            "Gate shape {} does not match input feature shape {}.".format(
                tuple(mu.shape),
                tuple(inputs.shape[1:]),
            )
        )
    if noise is None:
        noise = torch.randn_like(mu) * float(tau)
    if tuple(noise.shape) != tuple(mu.shape):
        raise ValueError("Gate noise must have the same shape as mu.")

    gate = torch.clamp(mu + noise, min=0.0, max=1.0)
    return inputs * gate.unsqueeze(0), gate


def paper_gate_regularizer(mu: torch.Tensor, tau: float, coefficient: float) -> torch.Tensor:
    """Return lambda_2 / 2 * sum_j Phi(mu_j / tau) from Eq. (6)."""
    if float(coefficient) <= 0.0:
        return mu.new_zeros(())
    safe_tau = max(float(tau), 1e-12)
    normal_cdf = 0.5 * (1.0 + torch.erf(mu / (safe_tau * math.sqrt(2.0))))
    return 0.5 * float(coefficient) * normal_cdf.sum()
