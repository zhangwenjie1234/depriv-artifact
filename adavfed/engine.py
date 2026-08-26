from __future__ import annotations

import torch

from .config import AdaVFedConfig
from .gates import apply_paper_stochastic_gate, paper_gate_regularizer
from .noise import (
    add_gaussian_noise,
    add_paper_dynamic_noise,
    add_theorem_one_dynamic_noise,
    add_theorem_one_gaussian_noise,
    theorem_one_joint_noise_multipliers,
)
from .regularizer import paper_constraint_regularizer


class AdaVFedEngine(torch.nn.Module):
    """Paper Algorithm 2/3/4 state owned by the isolated Ada-VFed path."""

    def __init__(self, config: AdaVFedConfig, num_passive: int, device: torch.device):
        super().__init__()
        self.config = config
        self.num_passive = int(num_passive)
        self.device = device
        self.gate_means = torch.nn.ParameterList()
        self._cached_embeddings: list[torch.Tensor] = []
        self.last_gate_stats: list[dict] = []
        self.last_forward_stats: list[dict] = []
        self.last_backward_stats: list[dict] = []
        self.last_regularization_stats: dict = {}

    def configure_training_steps(self, training_steps: int) -> None:
        if int(training_steps) <= 0:
            raise ValueError("AdaVFed training_steps must be positive.")
        self.config.total_training_steps = int(training_steps)

    def configure_rdp_step_epsilons(
        self,
        forward_step_epsilon: float,
        backward_step_epsilon: float,
    ) -> None:
        if float(forward_step_epsilon) <= 0.0 or float(backward_step_epsilon) <= 0.0:
            raise ValueError("RDP-calibrated AdaVFed requires positive forward/backward step epsilons.")
        self.config.forward_step_epsilon = float(forward_step_epsilon)
        self.config.backward_step_epsilon = float(backward_step_epsilon)

    def _theorem_one_noise_multipliers(self) -> tuple[float, float, float]:
        return theorem_one_joint_noise_multipliers(
            epsilon=self.config.total_epsilon,
            delta=self.config.delta,
            training_steps=self.config.total_training_steps,
            forward_privacy_fraction=self.config.forward_privacy_fraction,
        )

    def _noise_delta(self) -> float:
        if self.config.privacy_calibration == "rdp_subsampled_algorithm2":
            return float(self.config.rdp_accounting_delta)
        return float(self.config.delta)

    def prepare_active_embedding(self, released: torch.Tensor) -> torch.Tensor:
        if self.config.training_mode == "defense_all_aligned":
            return released
        return released.detach().requires_grad_(True)

    def ensure_input_gates(self, inputs: list[torch.Tensor]) -> None:
        if not self.config.enable_gates:
            return
        if len(inputs) != self.num_passive:
            raise ValueError("Expected one input tensor per passive party.")
        for party_id, tensor in enumerate(inputs):
            feature_shape = tuple(tensor.shape[1:])
            if party_id < len(self.gate_means):
                if tuple(self.gate_means[party_id].shape) != feature_shape:
                    raise ValueError("Ada-VFed input shape changed after gate initialization.")
                continue
            self.gate_means.append(
                torch.nn.Parameter(
                    torch.full(
                        feature_shape,
                        float(self.config.gate_initial_mu),
                        device=tensor.device,
                        dtype=tensor.dtype,
                    )
                )
            )

    def apply_input_gates(
        self,
        inputs: list[torch.Tensor],
        gate_noises: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        if not self.config.enable_gates:
            self.last_gate_stats = [
                {"mean": 1.0, "zero_ratio": 0.0, "one_ratio": 1.0}
                for _ in inputs
            ]
            return inputs

        self.ensure_input_gates(inputs)
        if gate_noises is not None and len(gate_noises) != self.num_passive:
            raise ValueError("Expected one gate-noise tensor per passive party.")

        gated_inputs = []
        gate_stats = []
        for party_id, tensor in enumerate(inputs):
            noise = None if gate_noises is None else gate_noises[party_id]
            if not self.training and noise is None:
                noise = torch.zeros_like(self.gate_means[party_id])
            gated, gate = apply_paper_stochastic_gate(
                tensor,
                self.gate_means[party_id],
                tau=self.config.gate_tau,
                noise=noise,
            )
            gated_inputs.append(gated)
            gate_stats.append(
                {
                    "mean": float(gate.mean().item()),
                    "zero_ratio": float((gate == 0.0).float().mean().item()),
                    "one_ratio": float((gate == 1.0).float().mean().item()),
                }
            )
        self.last_gate_stats = gate_stats
        return gated_inputs

    def record_gated_embeddings(self, embeddings: list[torch.Tensor]) -> None:
        self._cached_embeddings = list(embeddings)

    def build_regularization(self) -> torch.Tensor:
        if not self._cached_embeddings:
            return torch.zeros((), device=self.device)
        zero = sum(
            (embedding.sum() * 0.0 for embedding in self._cached_embeddings),
            torch.zeros((), device=self.device),
        )
        constraint = zero
        if self.config.enable_constraint:
            constraint = paper_constraint_regularizer(
                self._cached_embeddings,
                clip_C=self.config.clip_C,
                coefficient=self.config.lambda_constraint,
            )
        gate = zero
        if self.config.enable_gates:
            gate = sum(
                (paper_gate_regularizer(mu, self.config.gate_tau, self.config.lambda_gate) for mu in self.gate_means),
                gate,
            )
        total = constraint + gate
        self.last_regularization_stats = {
            "constraint": float(constraint.detach().item()),
            "gate": float(gate.detach().item()),
            "total": float(total.detach().item()),
        }
        return total

    @staticmethod
    def _passthrough_stats(tensor: torch.Tensor) -> dict:
        flat = tensor.detach().reshape(tensor.shape[0], -1)
        norm_mean = float(torch.linalg.vector_norm(flat, ord=2, dim=1).mean().item())
        return {
            "dp_applied": False,
            "raw_norm_mean": norm_mean,
            "clipped_norm_mean": norm_mean,
            "clip_hit_rate": 0.0,
            "epsilon_min": 0.0,
            "epsilon_max": 0.0,
            "sigma": 0.0,
            "sigma_min": 0.0,
            "sigma_mean": 0.0,
            "sigma_max": 0.0,
            "noise_std": 0.0,
        }

    def privatize_forward_embeddings(self, embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
        if not self.config.enable_forward_noise:
            self.last_forward_stats = [self._passthrough_stats(embedding) for embedding in embeddings]
            return list(embeddings)

        released = []
        stats_list = []
        theorem_mode = self.config.privacy_calibration == "theorem_one_joint"
        empirical_rho_mode = self.config.privacy_calibration == "paper_algorithm2_rho"
        rdp_subsampled_mode = self.config.privacy_calibration == "rdp_subsampled_algorithm2"
        noise_delta = self._noise_delta()
        if theorem_mode:
            forward_multiplier, _, _ = self._theorem_one_noise_multipliers()
        for embedding in embeddings:
            if theorem_mode:
                noisy, stats = add_theorem_one_dynamic_noise(
                    embedding,
                    clip_C=self.config.clip_C,
                    equivalent_multiplier=forward_multiplier,
                    temperature=self.config.temperature,
                    knn_k=self.config.knn_k,
                )
            elif self.config.enable_dynamic_forward_noise:
                noisy, stats = add_paper_dynamic_noise(
                    embedding,
                    clip_C=self.config.clip_C,
                    epsilon=(
                        self.config.forward_step_epsilon
                        if rdp_subsampled_mode
                        else self.config.total_epsilon
                    ),
                    delta=noise_delta,
                    temperature=self.config.temperature,
                    knn_k=self.config.knn_k,
                    noise_injection_ratio=(
                        self.config.noise_injection_ratio if empirical_rho_mode else 1.0
                    ),
                )
            else:
                noisy, stats = add_gaussian_noise(
                    embedding,
                    clip_C=self.config.clip_C,
                    epsilon=(
                        self.config.forward_step_epsilon
                        if rdp_subsampled_mode
                        else self.config.total_epsilon
                    ),
                    delta=noise_delta,
                )
                stats["epsilon_min"] = float(
                    self.config.forward_step_epsilon
                    if rdp_subsampled_mode
                    else self.config.total_epsilon
                )
                stats["epsilon_max"] = stats["epsilon_min"]
            stats["dp_applied"] = True
            released.append(noisy)
            stats_list.append(stats)
        self.last_forward_stats = stats_list
        return released

    def privatize_backward_grads(self, gradients: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        if not self.config.enable_backward_noise:
            self.last_backward_stats = [self._passthrough_stats(gradient) for gradient in gradients]
            return list(gradients)

        released = []
        stats_list = []
        theorem_mode = self.config.privacy_calibration == "theorem_one_joint"
        empirical_rho_mode = self.config.privacy_calibration == "paper_algorithm2_rho"
        rdp_subsampled_mode = self.config.privacy_calibration == "rdp_subsampled_algorithm2"
        noise_delta = self._noise_delta()
        if theorem_mode:
            _, backward_multiplier, _ = self._theorem_one_noise_multipliers()
        for gradient in gradients:
            if theorem_mode:
                noisy, stats = add_theorem_one_gaussian_noise(
                    gradient,
                    clip_C=self.config.backward_clip_C,
                    multiplier=backward_multiplier,
                )
            else:
                noisy, stats = add_gaussian_noise(
                    gradient,
                    clip_C=self.config.backward_clip_C,
                    epsilon=(
                        self.config.backward_step_epsilon
                        if rdp_subsampled_mode
                        else self.config.total_epsilon
                    ),
                    delta=noise_delta,
                    noise_injection_ratio=(
                        self.config.noise_injection_ratio if empirical_rho_mode else 1.0
                    ),
                )
            stats["dp_applied"] = True
            released.append(noisy)
            stats_list.append(stats)
        self.last_backward_stats = stats_list
        return released

    def get_debug_snapshot(self) -> dict:
        return {
            "gate": list(self.last_gate_stats),
            "forward": list(self.last_forward_stats),
            "backward": list(self.last_backward_stats),
            "regularization": dict(self.last_regularization_stats),
        }

    def reset_batch_state(self) -> None:
        self._cached_embeddings = []
