import numpy as np
import torch
from opacus.accountants.analysis import rdp as rdp_analysis

import utils.dp as dp


class PrivacyMonitor:
    def __init__(
        self,
        num_parties,
        sample_rate,
        delta=1e-5,
        mechanism='gaussian',
        use_dim_amplification=False,
        verbose=True,
        progress_interval=100,
    ):
        self.enabled = True
        self.delta = delta
        self.sample_rate = sample_rate
        self.mechanism = mechanism
        self.num_parties = num_parties
        self.use_dim_amplification = use_dim_amplification
        self.verbose = verbose
        self.progress_interval = progress_interval

        self.alphas = np.array([
            1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0,
            8.0, 10.0, 13.0, 16.0, 20.0, 25.0, 32.0, 45.0, 64.0, 90.0,
            128.0, 256.0, 512.0, 1024.0,
        ])

        self.rdp_history = np.zeros((num_parties, len(self.alphas)))
        self.noise_factor = np.sqrt(2 * np.log(1.25 / delta))

        self.steps_since_last_check = 0
        self.last_avg_eps = 0.0
        self.laplace_history = [[] for _ in range(num_parties)]

    def _accumulate_release(self, eps_list, q_sparse_list=None, steps=1):
        if not self.enabled:
            return None, None

        q_data = self.sample_rate
        last_accounted_sample_rate = None
        last_safe_nm = None

        for i, eps in enumerate(eps_list):
            if eps < 1e-9:
                continue

            if self.mechanism == 'gaussian':
                raw_nm = self.noise_factor / eps
                safe_nm = max(min(raw_nm, 500.0), 0.01)
                # Coordinate sparsification after noising is treated as post-processing.
                # In the current code path, only data-level minibatch sampling
                # participates in privacy accounting. q_sparse_list is ignored unless
                # use_dim_amplification is explicitly enabled for experimental use.
                q_dim = q_sparse_list[i] if q_sparse_list is not None else 1.0
                accounted_sample_rate = q_data * q_dim if self.use_dim_amplification else q_data
                accounted_sample_rate = max(accounted_sample_rate, 1e-7)

                single_step_rdp = rdp_analysis.compute_rdp(
                    q=accounted_sample_rate,
                    noise_multiplier=safe_nm,
                    steps=steps,
                    orders=self.alphas,
                )
                self.rdp_history[i] += single_step_rdp
                last_accounted_sample_rate = accounted_sample_rate
                last_safe_nm = safe_nm

            elif self.mechanism == 'laplace':
                pass

        return last_accounted_sample_rate, last_safe_nm

    def _maybe_print_progress(self, accounted_sample_rate, safe_nm):
        if not self.verbose or self.progress_interval <= 0:
            return
        if self.steps_since_last_check <= 0:
            return
        if self.steps_since_last_check % self.progress_interval != 0:
            return

        shown_rate = 0.0 if accounted_sample_rate is None else accounted_sample_rate
        shown_nm = 0.0 if safe_nm is None else safe_nm
        print(
            f">>> [RDP-核算进度] 已运行: {self.steps_since_last_check} 步 | "
            f"会计采样率: {shown_rate:.5f} | nm: {shown_nm:.2f}"
        )

    def step(self, eps_list, q_sparse_list=None, steps=1):
        accounted_sample_rate, safe_nm = self._accumulate_release(
            eps_list,
            q_sparse_list=q_sparse_list,
            steps=steps,
        )
        self.steps_since_last_check += steps
        self._maybe_print_progress(accounted_sample_rate, safe_nm)

    def compose(self, release_specs, steps=1):
        if not self.enabled:
            return

        last_accounted_sample_rate = None
        last_safe_nm = None
        for eps_list, q_sparse_list in release_specs:
            accounted_sample_rate, safe_nm = self._accumulate_release(
                eps_list,
                q_sparse_list=q_sparse_list,
                steps=steps,
            )
            if accounted_sample_rate is not None:
                last_accounted_sample_rate = accounted_sample_rate
            if safe_nm is not None:
                last_safe_nm = safe_nm

        self.steps_since_last_check += steps
        self._maybe_print_progress(last_accounted_sample_rate, last_safe_nm)

    def get_avg_spent(self, force=False):
        if not self.enabled:
            return 0.0

        if not force and self.steps_since_last_check < 20:
            return self.last_avg_eps

        total_eps = 0.0
        for i in range(self.num_parties):
            eps_at_alphas = self.rdp_history[i] + (
                np.log(1.0 / self.delta) / (self.alphas - 1)
            )
            total_eps += np.min(eps_at_alphas)

        self.last_avg_eps = total_eps / self.num_parties
        return self.last_avg_eps

    def report(self):
        print(f"\n{'=' * 20} RDP Report {'=' * 20}")
        avg_eps = self.get_avg_spent(force=True)
        for i in range(self.num_parties):
            eps_at_alphas = self.rdp_history[i] + (
                np.log(1.0 / self.delta) / (self.alphas - 1)
            )
            print(f"  Party {i}: Cumulative eps = {np.min(eps_at_alphas):.4f} (at delta={self.delta})")
        print(f"  [System Average]: Avg eps = {avg_eps:.4f}")
        print("=" * 68 + "\n")


def initialize_bidirectional_privacy_plan(vfl):
    total_target_eps = float(vfl.args.epsilon)
    forward_ratio = float(np.clip(getattr(vfl.args, 'forward_budget_ratio', 0.3), 1e-6, 1.0 - 1e-6))

    manual_forward_step_eps = float(getattr(vfl.args, 'forward_fixed_eps', -1.0))
    if manual_forward_step_eps > 0.0:
        vfl.fixed_forward_eps = manual_forward_step_eps
        vfl.forward_total_eps_target = dp.compute_epsilon_from_constant_step(
            vfl.fixed_forward_eps,
            total_steps=vfl.total_training_steps,
            q=vfl.sample_rate,
            delta=vfl.privacy_monitor.delta,
        )
    else:
        vfl.forward_total_eps_target = total_target_eps * forward_ratio
        vfl.fixed_forward_eps = dp.compute_step_epsilon_for_target(
            vfl.forward_total_eps_target,
            total_steps=vfl.total_training_steps,
            q=vfl.sample_rate,
            delta=vfl.privacy_monitor.delta,
        )

    empty_rdp = np.zeros_like(vfl.privacy_monitor.rdp_history)
    vfl.base_step_eps = dp.solve_backward_step_epsilon_for_total(
        current_rdp_history=empty_rdp,
        remaining_steps=vfl.total_training_steps,
        q=vfl.sample_rate,
        total_target_eps=total_target_eps,
        forward_step_epsilon=vfl.fixed_forward_eps,
        delta=vfl.privacy_monitor.delta,
        initial_high=max(vfl.fixed_forward_eps, 1.0),
    )
    vfl.backward_total_eps_target = dp.compute_epsilon_from_constant_step(
        vfl.base_step_eps,
        total_steps=vfl.total_training_steps,
        q=vfl.sample_rate,
        delta=vfl.privacy_monitor.delta,
    )
    vfl.projected_composed_eps = dp.project_final_average_epsilon(
        current_rdp_history=empty_rdp,
        remaining_steps=vfl.total_training_steps,
        q=vfl.sample_rate,
        future_step_eps_list=[vfl.fixed_forward_eps, vfl.base_step_eps],
        delta=vfl.privacy_monitor.delta,
    )

    if vfl.args.rl:
        return
    prefix = "BiDir Baseline"
    print(
        f"[{prefix}] Forward budget target(view) = {vfl.forward_total_eps_target:.4f} "
        f"(ratio={forward_ratio:.2f})"
    )
    print(f"[{prefix}] Fixed forward per-step eps = {vfl.fixed_forward_eps:.6f}")
    print(f"[{prefix}] Planned backward budget(view) = {vfl.backward_total_eps_target:.4f}")
    print(f"[{prefix}] Initial backward per-step eps cap = {vfl.base_step_eps:.6f}")


def get_current_backward_cap_eps(vfl, epoch, batch_idx):
    if not getattr(vfl.args, 'rl', False):
        return 0.0
    return float(max(vfl.base_step_eps, 1e-8))


def get_fixed_forward_eps_list(vfl):
    if not getattr(vfl.args, 'rl', False):
        return [0.0] * vfl.args.num_passive
    fixed_forward_eps = float(max(getattr(vfl, 'fixed_forward_eps', 0.0), 0.0))
    return [fixed_forward_eps] * vfl.args.num_passive


def step_rl_privacy_accounting(vfl, forward_eps_list, backward_eps_list, backward_q_list):
    if not getattr(vfl.args, 'rl', False):
        return
    release_specs = []
    if forward_eps_list is not None:
        forward_q_list = [1.0] * vfl.args.num_passive
        vfl.forward_privacy_monitor.step(forward_eps_list, q_sparse_list=forward_q_list)
        release_specs.append((forward_eps_list, forward_q_list))
    if backward_eps_list is not None:
        vfl.backward_privacy_monitor.step(backward_eps_list, q_sparse_list=backward_q_list)
        release_specs.append((backward_eps_list, backward_q_list))
    if release_specs:
        vfl.privacy_monitor.compose(release_specs)
