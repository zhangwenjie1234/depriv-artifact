from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaVFedConfig:
    """Paper-literal Ada-VFed defaults from Section VII-A.2."""

    enabled: bool = False
    # Training control aligned with the existing defense_all graph: passive
    # models receive the retained CE path and the explicit noisy return path.
    training_mode: str = "defense_all_aligned"
    # Keep aligned graph semantics but avoid defense_all's duplicate model step.
    use_entire_optimizer_update: bool = False
    clip_C: float = 2.0
    # Diagnostic comparison setting: preserve the paper forward threshold while
    # matching defense_all's gradient-scale threshold on the backward channel.
    backward_clip_C: float = 0.0005
    delta: float = 1e-4
    total_epsilon: float = 1.0
    # Manual ablation controls. Use 1e-4 in this implementation because both
    # Eq. (6) terms are summed over batches/features; paper's nominal 0.1 is
    # not scale-compatible with the current MNIST model.
    lambda_constraint: float = 1e-4
    lambda_gate: float = 1e-4
    enable_constraint: bool = True
    enable_gates: bool = False
    # Diagnostic-only channel switches. Keep both enabled for a DP result.
    enable_forward_noise: bool = True
    enable_backward_noise: bool = True
    # Algorithm 2's Laplacian-score allocation for the forward DP release.
    enable_dynamic_forward_noise: bool = True
    gate_tau: float = 1.0
    gate_initial_mu: float = 1.0
    temperature: float = 1.0
    knn_k: int = 5
    debug_interval: int = 200
    # This mode reuses the project's subsampled-RDP step-budget planner.
    # It is an accounting-aligned AdaVFed ablation, not a paper-default claim.
    privacy_calibration: str = "rdp_subsampled_algorithm2"
    forward_privacy_fraction: float = 0.5
    total_training_steps: int = 1
    forward_step_epsilon: float = 0.0
    backward_step_epsilon: float = 0.0
    rdp_accounting_delta: float = 1e-5
    noise_injection_ratio: float = 1.0

    @classmethod
    def from_args(cls, args):
        return cls(
            enabled=bool(getattr(args, "adavfed", False)),
            total_epsilon=float(getattr(args, "epsilon", 1.0)),
        )
