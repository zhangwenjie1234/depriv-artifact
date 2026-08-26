import numpy as np


def get_control_party_risks(vfl):
    tracker = getattr(vfl, "released_gradient_risk_tracker", None)
    risks = tracker.get_scores() if tracker is not None else []

    risks = [float(np.clip(value, 0.0, 100.0)) for value in risks[:vfl.args.num_passive]]
    if len(risks) < vfl.args.num_passive:
        risks.extend([0.0] * (vfl.args.num_passive - len(risks)))
    return risks


def construct_rl_state(vfl, smooth_party_risks, current_utility):
    total_steps = vfl.args.epochs * vfl.iteration
    current_step = vfl.epoch * vfl.iteration + vfl.batch_idx
    progress = current_step / total_steps

    cached_avg_spent = getattr(vfl, "_cached_avg_privacy_spent", None)
    if cached_avg_spent is None:
        current_avg_spent = vfl.privacy_monitor.get_avg_spent(force=True)
    else:
        current_avg_spent = float(cached_avg_spent)
    target_eps = vfl.args.epsilon
    budget_usage = current_avg_spent / target_eps

    party_risks = list(smooth_party_risks)
    if len(party_risks) < vfl.args.num_passive:
        party_risks.extend([0.0] * (vfl.args.num_passive - len(party_risks)))

    state = np.array(
        [risk / 100.0 for risk in party_risks[:vfl.args.num_passive]]
        + [current_utility / 100.0, progress, budget_usage],
        dtype=np.float32,
    )
    return state


def get_train_utility_value(vfl):
    released_utility = getattr(vfl, "last_dp_utility", None)
    if released_utility is not None:
        return float(released_utility)
    if hasattr(vfl, "utility_tracker") and len(vfl.utility_tracker) > 0:
        return float(vfl.utility_tracker[-1])
    return 0.0


def released_gradient_party_qualities(released_gradients, noise_stds):
    """Return one DP-release quality score per available party release.

    Each score is a bounded signal-to-noise proxy. It uses only the released
    tensor, its public Gaussian noise scale, and tensor dimensionality.
    """
    qualities = []
    for released, noise_std in zip(released_gradients or [], noise_stds or []):
        if released is None:
            qualities.append(0.0)
            continue
        released_norm = float(released.detach().norm().item())
        noise_floor = float(max(float(noise_std), 0.0)) * float(released.numel()) ** 0.5
        qualities.append(100.0 * released_norm / max(released_norm + noise_floor, 1e-12))
    return qualities


def released_gradient_utility(released_gradients, noise_stds):
    """Average party quality for the existing SAC utility state."""
    qualities = released_gradient_party_qualities(released_gradients, noise_stds)
    return float(np.mean(qualities)) if qualities else 0.0


def evaluate_rl_performance(vfl):
    if not vfl.args.rl or vfl.rl_agent is None:
        return None

    smooth_party_risks = get_control_party_risks(vfl)
    current_utility = get_train_utility_value(vfl)

    state_t = construct_rl_state(
        vfl,
        smooth_party_risks,
        current_utility,
    )

    action_t_tanh, _, _ = vfl.rl_agent.choose_action(state_t, evaluate=True)
    executed_eval_action = action_t_tanh.cpu().numpy().flatten()
    eps_tuple = vfl.rl_agent.scale_action(
        executed_eval_action,
        base_step_eps=vfl.base_step_eps,
    )
    if hasattr(vfl, '_apply_backward_risk_gate'):
        eps_tuple = vfl._apply_backward_risk_gate(
            eps_tuple,
            smooth_party_risks,
            vfl.base_step_eps,
        )
    total_reward = vfl.rl_agent.calculate_reward(
        state_t,
        state_t,
        eps_tuple,
    )
    steps = 1

    avg_reward = total_reward / steps if steps > 0 else 0.0
    return avg_reward
