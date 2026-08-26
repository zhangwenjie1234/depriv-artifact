from __future__ import annotations

import torch


def build_completion_cache_payload(
    released_embeddings: list[torch.Tensor],
    observed_gradients: list[torch.Tensor],
    labels: torch.Tensor,
) -> list:
    """Return the communication transcript consumed by completion attacks."""
    return [
        [embedding.detach().cpu() for embedding in released_embeddings],
        [gradient.detach().cpu() for gradient in observed_gradients],
        labels.detach().cpu(),
    ]


def completion_attack_mode(
    epoch: int,
    set_attack_epoch: bool,
    attack_epoch: int,
) -> str | None:
    """Choose whether a completion attack initializes, runs, or waits this epoch."""
    if not set_attack_epoch:
        return "init" if epoch == 0 else "run"
    current_epoch = epoch + 1
    if current_epoch < attack_epoch:
        return None
    return "init" if current_epoch == attack_epoch else "run"
