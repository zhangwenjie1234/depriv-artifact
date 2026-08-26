import torch


class MaliciousSGD(torch.optim.Optimizer):
    """
    Active model completion optimizer adapted from the malicious local
    optimizer in "Label Inference Attacks Against Vertical Federated
    Learning" (USENIX Security 2022).
    """

    def __init__(self, params, lr, momentum=0.9, gamma=1.0, rmax=5.0, rmin=1.0, weight_decay=0.0):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if rmin <= 0 or rmax < rmin:
            raise ValueError("Require 0 < rmin <= rmax")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            gamma=gamma,
            rmax=rmax,
            rmin=rmin,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        scale_values = []
        grad_norm_values = []
        velocity_norm_values = []
        nonfinite_grad_tensors = 0
        reset_velocity_tensors = 0
        fallback_step_tensors = 0

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            gamma = group["gamma"]
            rmax = group["rmax"]
            rmin = group["rmin"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                if not torch.isfinite(grad).all():
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                    nonfinite_grad_tensors += 1
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                grad_norm = float(grad.norm().item())
                grad_norm_values.append(grad_norm)

                state = self.state[p]
                if len(state) == 0:
                    state["velocity"] = torch.zeros_like(p)
                    state["initialized"] = False

                velocity = state["velocity"]
                if not torch.isfinite(velocity).all():
                    velocity.zero_()
                    state["initialized"] = False
                    reset_velocity_tensors += 1

                current_velocity = velocity.mul(momentum).add(grad, alpha=1.0 - momentum)
                if not torch.isfinite(current_velocity).all():
                    current_velocity = torch.nan_to_num(current_velocity, nan=0.0, posinf=0.0, neginf=0.0)
                    fallback_step_tensors += 1

                if state["initialized"]:
                    eps = 1e-12
                    prev_norm = float(velocity.norm().item())
                    curr_norm = float(current_velocity.norm().item())
                    if prev_norm > eps and torch.isfinite(torch.tensor(prev_norm)) and torch.isfinite(torch.tensor(curr_norm)):
                        # Use a tensor-level norm ratio instead of element-wise ratios.
                        # This is much more stable for deep passive backbones while still
                        # preserving AMC's "accelerate informative updates" behavior.
                        raw_scale = 1.0 + gamma * ((curr_norm / (prev_norm + eps)) - 1.0)
                        if not torch.isfinite(torch.tensor(raw_scale)):
                            raw_scale = 1.0
                            fallback_step_tensors += 1
                        scale = max(rmin, min(rmax, float(raw_scale)))
                        new_velocity = current_velocity.mul(scale)
                        scale_values.append(scale)
                    else:
                        new_velocity = current_velocity
                        scale_values.append(1.0)
                else:
                    new_velocity = current_velocity
                    scale_values.append(1.0)

                if not torch.isfinite(new_velocity).all():
                    new_velocity = torch.nan_to_num(current_velocity, nan=0.0, posinf=0.0, neginf=0.0)
                    fallback_step_tensors += 1

                velocity.copy_(new_velocity)
                state["initialized"] = True
                velocity_norm_values.append(float(new_velocity.norm().item()))
                p.add_(new_velocity, alpha=-lr)

        if len(scale_values) > 0:
            self.last_step_stats = {
                "scale_mean": float(sum(scale_values) / len(scale_values)),
                "scale_min": float(min(scale_values)),
                "scale_max": float(max(scale_values)),
                "grad_norm_mean": float(sum(grad_norm_values) / max(1, len(grad_norm_values))),
                "velocity_norm_mean": float(sum(velocity_norm_values) / max(1, len(velocity_norm_values))),
                "num_tensors": int(len(scale_values)),
                "nonfinite_grad_tensors": int(nonfinite_grad_tensors),
                "reset_velocity_tensors": int(reset_velocity_tensors),
                "fallback_step_tensors": int(fallback_step_tensors),
            }
        else:
            self.last_step_stats = None

        return loss
