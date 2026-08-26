from __future__ import annotations

import math

import torch


def theorem_one_joint_noise_multipliers(
    epsilon: float,
    delta: float,
    training_steps: int,
    forward_privacy_fraction: float = 0.5,
) -> tuple[float, float, float]:
    """Calibrate normalized forward/backward noise from Theorem 1.

    The theorem bounds the sum of forward and backward RDP precisions.
    The returned multipliers are normalized by the Algorithm 1 sensitivity
    ``2C``; callers convert them to coordinate standard deviations.
    """
    if int(training_steps) <= 0:
        raise ValueError("training_steps must be positive.")
    if not 0.0 < float(forward_privacy_fraction) < 1.0:
        raise ValueError("forward_privacy_fraction must be in (0, 1).")

    safe_epsilon = max(float(epsilon), 1e-12)
    privacy_mass = float(training_steps) * (
        safe_epsilon + 2.0 * math.log(1.0 / float(delta))
    ) / safe_epsilon**2
    precision_budget = 1.0 / privacy_mass
    forward_precision = float(forward_privacy_fraction) * precision_budget
    backward_precision = (1.0 - float(forward_privacy_fraction)) * precision_budget
    return (
        1.0 / math.sqrt(forward_precision),
        1.0 / math.sqrt(backward_precision),
        precision_budget,
    )


def theorem_one_dimension_sigmas(
    weights: torch.Tensor,
    clip_C: float,
    equivalent_multiplier: float,
) -> torch.Tensor:
    """Map dynamic weights to coordinate noise under Lemma 5.

    A dimension with weight ``w_d`` receives ``w_d`` of the allowed
    precision. Consequently, sum_d (2C / sigma_d)^2 equals
    ``1 / equivalent_multiplier^2`` exactly.
    """
    if weights.dim() != 1 or weights.numel() == 0:
        raise ValueError("weights must be a non-empty 1D tensor.")
    if torch.any(weights <= 0):
        raise ValueError("weights must be strictly positive.")
    normalized_weights = weights / weights.sum()
    sensitivity = 2.0 * float(clip_C)
    multiplier = max(float(equivalent_multiplier), 1e-12)
    return sensitivity * multiplier / torch.sqrt(normalized_weights)


def paper_gaussian_scale(clip_C: float, epsilon: float, delta: float) -> float:
    """Algorithm 1 standard deviation for a scalar privacy budget."""
    safe_epsilon = max(float(epsilon), 1e-12)
    return (2.0 * float(clip_C) * math.sqrt(2.0 * math.log(1.25 / float(delta)))) / safe_epsilon


def paper_dimension_privacy_parameters(
    scores: torch.Tensor,
    clip_C: float,
    epsilon: float,
    delta: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Algorithm 2 weights, epsilon_d, and sigma_d."""
    if scores.dim() != 1 or scores.numel() == 0:
        raise ValueError("Laplacian scores must be a non-empty 1D tensor.")
    safe_temperature = max(float(temperature), 1e-12)
    weights = torch.softmax(-scores / safe_temperature, dim=0)
    epsilons = float(scores.numel()) * weights * float(epsilon)
    numerator = paper_gaussian_scale(clip_C=clip_C, epsilon=1.0, delta=delta)
    sigmas = numerator / torch.clamp(epsilons, min=1e-12)
    return weights, epsilons, sigmas


def clip_per_sample(tensor: torch.Tensor, clip_C: float) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.dim() < 2:
        raise ValueError("Expected a batched tensor for per-sample clipping.")
    flat = tensor.reshape(tensor.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True)
    coefficients = torch.clamp(float(clip_C) / (norms + 1e-12), max=1.0)
    clipped = tensor * coefficients.reshape([tensor.shape[0]] + [1] * (tensor.dim() - 1))
    return clipped, norms.squeeze(1)


def laplacian_scores(tensor: torch.Tensor, knn_k: int) -> torch.Tensor:
    """Label-free Laplacian Score from Eq. (7)-(8), evaluated per release batch."""
    flat = tensor.detach().reshape(tensor.shape[0], -1)
    batch_size, dimension = flat.shape
    if batch_size <= 1:
        return torch.zeros(dimension, device=flat.device, dtype=flat.dtype)

    k = min(max(int(knn_k), 1), batch_size - 1)
    pairwise = torch.cdist(flat, flat, p=2)
    pairwise.fill_diagonal_(float("inf"))
    distances, neighbors = torch.topk(pairwise, k=k, dim=1, largest=False)
    kernel_scale = torch.clamp(distances.mean().pow(2), min=1e-12)
    similarities = torch.exp(-distances.pow(2) / kernel_scale)
    adjacency = torch.zeros_like(pairwise)
    adjacency.scatter_(1, neighbors, similarities)
    adjacency = torch.maximum(adjacency, adjacency.transpose(0, 1))

    degree = adjacency.sum(dim=1)
    degree_sum = torch.clamp(degree.sum(), min=1e-12)
    centered = flat - ((degree.unsqueeze(1) * flat).sum(dim=0, keepdim=True) / degree_sum)
    laplacian_feature = degree.unsqueeze(1) * centered - adjacency @ centered
    numerator = (centered * laplacian_feature).sum(dim=0)
    denominator = torch.clamp((degree.unsqueeze(1) * centered.pow(2)).sum(dim=0), min=1e-12)
    return numerator / denominator


def add_paper_dynamic_noise(
    tensor: torch.Tensor,
    clip_C: float,
    epsilon: float,
    delta: float,
    temperature: float,
    knn_k: int,
    noise_injection_ratio: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    clipped, raw_norms = clip_per_sample(tensor, clip_C=clip_C)
    flat = clipped.reshape(clipped.shape[0], -1)
    scores = laplacian_scores(clipped, knn_k=knn_k)
    weights, epsilons, sigmas = paper_dimension_privacy_parameters(
        scores,
        clip_C=clip_C,
        epsilon=epsilon,
        delta=delta,
        temperature=temperature,
    )
    if not 0.0 < float(noise_injection_ratio) <= 1.0:
        raise ValueError("noise_injection_ratio must be in (0, 1].")
    sigmas = sigmas.to(device=flat.device, dtype=flat.dtype) * float(noise_injection_ratio)
    noise = torch.randn_like(flat) * sigmas.unsqueeze(0)
    released = (flat + noise).reshape_as(clipped)
    clipped_norms = torch.linalg.vector_norm(flat, ord=2, dim=1)
    return released, {
        "raw_norm_mean": float(raw_norms.mean().item()),
        "clipped_norm_mean": float(clipped_norms.mean().item()),
        "clip_hit_rate": float((raw_norms > float(clip_C)).float().mean().item()),
        "weight_min": float(weights.min().item()),
        "weight_max": float(weights.max().item()),
        "epsilon_min": float(epsilons.min().item()),
        "epsilon_max": float(epsilons.max().item()),
        "sigma_min": float(sigmas.min().item()),
        "sigma_mean": float(sigmas.mean().item()),
        "sigma_max": float(sigmas.max().item()),
        "reduction_max": float((epsilons.max()).item()),
        "noise_std": float(noise.std().item()),
    }


def add_theorem_one_dynamic_noise(
    tensor: torch.Tensor,
    clip_C: float,
    equivalent_multiplier: float,
    temperature: float,
    knn_k: int,
) -> tuple[torch.Tensor, dict]:
    """Release a dynamic forward vector under Theorem 1 and Lemma 5."""
    clipped, raw_norms = clip_per_sample(tensor, clip_C=clip_C)
    flat = clipped.reshape(clipped.shape[0], -1)
    scores = laplacian_scores(clipped, knn_k=knn_k)
    weights = torch.softmax(-scores / max(float(temperature), 1e-12), dim=0)
    sigmas = theorem_one_dimension_sigmas(
        weights=weights,
        clip_C=clip_C,
        equivalent_multiplier=equivalent_multiplier,
    ).to(device=flat.device, dtype=flat.dtype)
    noise = torch.randn_like(flat) * sigmas.unsqueeze(0)
    released = (flat + noise).reshape_as(clipped)
    clipped_norms = torch.linalg.vector_norm(flat, ord=2, dim=1)
    return released, {
        "raw_norm_mean": float(raw_norms.mean().item()),
        "clipped_norm_mean": float(clipped_norms.mean().item()),
        "clip_hit_rate": float((raw_norms > float(clip_C)).float().mean().item()),
        "weight_min": float(weights.min().item()),
        "weight_max": float(weights.max().item()),
        "epsilon_min": 0.0,
        "epsilon_max": 0.0,
        "sigma_min": float(sigmas.min().item()),
        "sigma_mean": float(sigmas.mean().item()),
        "sigma_max": float(sigmas.max().item()),
        "equivalent_multiplier": float(equivalent_multiplier),
        "noise_std": float(noise.std().item()),
    }


def add_gaussian_noise(
    tensor: torch.Tensor,
    clip_C: float,
    epsilon: float,
    delta: float,
    noise_injection_ratio: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    clipped, raw_norms = clip_per_sample(tensor, clip_C=clip_C)
    if not 0.0 < float(noise_injection_ratio) <= 1.0:
        raise ValueError("noise_injection_ratio must be in (0, 1].")
    sigma = paper_gaussian_scale(clip_C=clip_C, epsilon=epsilon, delta=delta) * float(noise_injection_ratio)
    noise = torch.randn_like(clipped) * sigma
    clipped_norms = torch.linalg.vector_norm(clipped.reshape(clipped.shape[0], -1), ord=2, dim=1)
    return clipped + noise, {
        "raw_norm_mean": float(raw_norms.mean().item()),
        "clipped_norm_mean": float(clipped_norms.mean().item()),
        "clip_hit_rate": float((raw_norms > float(clip_C)).float().mean().item()),
        "sigma": float(sigma),
        "sigma_min": float(sigma),
        "sigma_mean": float(sigma),
        "sigma_max": float(sigma),
        "noise_std": float(noise.std().item()),
    }


def add_theorem_one_gaussian_noise(
    tensor: torch.Tensor,
    clip_C: float,
    multiplier: float,
) -> tuple[torch.Tensor, dict]:
    """Release a backward gradient using the Theorem 1 noise multiplier."""
    clipped, raw_norms = clip_per_sample(tensor, clip_C=clip_C)
    sigma = 2.0 * float(clip_C) * float(multiplier)
    noise = torch.randn_like(clipped) * sigma
    clipped_norms = torch.linalg.vector_norm(clipped.reshape(clipped.shape[0], -1), ord=2, dim=1)
    return clipped + noise, {
        "raw_norm_mean": float(raw_norms.mean().item()),
        "clipped_norm_mean": float(clipped_norms.mean().item()),
        "clip_hit_rate": float((raw_norms > float(clip_C)).float().mean().item()),
        "sigma": float(sigma),
        "equivalent_multiplier": float(multiplier),
        "noise_std": float(noise.std().item()),
    }
