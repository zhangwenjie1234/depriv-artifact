from __future__ import annotations

import torch


def paper_constraint_regularizer(
    embeddings: list[torch.Tensor],
    clip_C: float,
    coefficient: float,
) -> torch.Tensor:
    """Return lambda_1 / 2 * sum_i max(0, ||h(x_i * Z)||_2 - C)."""
    if not embeddings:
        return torch.tensor(0.0)

    penalties = []
    for embedding in embeddings:
        norms = torch.linalg.vector_norm(embedding.reshape(embedding.shape[0], -1), ord=2, dim=1)
        penalties.append(torch.clamp(norms - float(clip_C), min=0.0).sum())
    return 0.5 * float(coefficient) * torch.stack(penalties).sum()
