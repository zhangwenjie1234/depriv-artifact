"""Temporary, optional RL runtime diagnostics.

Delete this file to disable the diagnostics; ``attackers/vflbase.py`` treats
the import as optional.  This module reads only public configuration and state
derived from historical DP releases.
"""


def print_config(vfl):
    args = vfl.args
    parties = int(args.num_passive)
    risk_t = [float(vfl.proxy_risk_threshold_pct)] * parties
    base_fwd_c = float(getattr(args, "clip_threshold_forward", 1.0))
    fwd_c = [
        vfl.task_prior.get_forward_clip_value(pid, base_fwd_c)
        for pid in range(parties)
    ]
    mode = {0: "dense", 1: "public_topk", 2: "random"}[
        int(getattr(args, "backward_topk_mask", 1))
    ]
    risk_mode = str(getattr(vfl, "risk_mode", getattr(args, "rl_risk_mode", "geometry")))
    tracker = getattr(vfl, "released_gradient_risk_tracker", None)
    public_priors = (
        tracker.get_public_priors()
        if tracker is not None and hasattr(tracker, "get_public_priors")
        else None
    )
    print(
        "[RLDBG Config] risk_mode={} risk_T={} risk_window={}/{} update={} ema={:.2f}"
        .format(
            risk_mode,
            [round(value, 2) for value in risk_t],
            int(getattr(args, "rl_risk_min_samples", 64)),
            int(getattr(args, "rl_risk_window_samples", 256)),
            int(getattr(args, "rl_risk_update_interval", 10)),
            float(getattr(args, "rl_risk_ema_beta", 0.9)),
        )
    )
    if public_priors is not None:
        print(
            "[RLDBG Config] public_risk_priors={}".format(
                [round(float(value), 2) for value in public_priors]
            )
        )
    print(
        "[RLDBG Config] eps_total={:.4f} eps_fwd_step={:.8f} "
        "eps_bwd_cap={:.8f} C_fwd={} C_bwd0={:.8f}"
        .format(
            float(args.epsilon),
            float(vfl.fixed_forward_eps),
            float(vfl.base_step_eps),
            [round(value, 6) for value in fwd_c],
            float(getattr(args, "clip_threshold", 1.0)),
        )
    )
    print(
        "[RLDBG Config] backward={} q_normal=[{:.2f},{:.2f}] "
        "q_risk=[{:.2f},{:.2f}] warmup_q={:.2f} SAC_window={} update_every={}"
        .format(
            mode,
            float(getattr(args, "rl_q_min", 0.05)),
            float(getattr(args, "rl_q_max", 0.15)),
            float(getattr(args, "rl_risk_q_min", 0.15)),
            float(getattr(args, "rl_risk_q_max", 0.50)),
            float(vfl.q_weak_fixed),
            int(vfl.state_window_size),
            int(vfl.sac_update_interval),
        )
    )


def start_epoch(vfl):
    parties = int(vfl.args.num_passive)
    vfl._rldbg_steps = 0
    vfl._rldbg_risk_sum = [0.0] * parties
    vfl._rldbg_quality_sum = [0.0] * parties
    vfl._rldbg_bwd_eps_sum = [0.0] * parties
    vfl._rldbg_q_sum = [0.0] * parties
    vfl._rldbg_trigger_start = list(vfl.risk_gate_trigger_counts)
    vfl._rldbg_decision_start = list(vfl.risk_gate_decision_counts)


def record_step(vfl, backward_eps, backward_q):
    vfl._rldbg_steps += 1
    risks = list(getattr(
        vfl, "last_control_party_risks", [0.0] * vfl.args.num_passive
    ))
    qualities = list(getattr(
        vfl, "last_dp_party_qualities", [100.0] * vfl.args.num_passive
    ))
    for pid in range(vfl.args.num_passive):
        vfl._rldbg_risk_sum[pid] += float(risks[pid])
        vfl._rldbg_quality_sum[pid] += float(qualities[pid])
        vfl._rldbg_bwd_eps_sum[pid] += float(backward_eps[pid])
        vfl._rldbg_q_sum[pid] += float(backward_q[pid])


def finish_epoch(vfl, epoch):
    steps = max(int(vfl._rldbg_steps), 1)
    base_fwd_c = float(getattr(vfl.args, "clip_threshold_forward", 1.0))
    rows = []
    tracker = getattr(vfl, "released_gradient_risk_tracker", None)
    public_priors = (
        tracker.get_public_priors()
        if tracker is not None and hasattr(tracker, "get_public_priors")
        else [None] * vfl.args.num_passive
    )
    for pid in range(vfl.args.num_passive):
        triggers = vfl.risk_gate_trigger_counts[pid] - vfl._rldbg_trigger_start[pid]
        decisions = vfl.risk_gate_decision_counts[pid] - vfl._rldbg_decision_start[pid]
        trigger_rate = 100.0 * triggers / decisions if decisions else 0.0
        initial_fwd_c = vfl.task_prior.get_forward_clip_value(pid, base_fwd_c)
        current_fwd_c = float(vfl.forward_norm_tracker.get(pid, initial_fwd_c))
        current_bwd_c = float(vfl.norm_tracker.get(
            pid, float(getattr(vfl.args, "clip_threshold", 1.0))
        ))
        prior_text = (
            " prior={:.1f}%".format(float(public_priors[pid]))
            if public_priors[pid] is not None else ""
        )
        rows.append(
            "P{} risk={:.1f}% T={:.1f}%{} quality={:.1f}% trig={}/{}({:.1f}%) "
            "Cf={:.4f} Cb={:.6f} epsb={:.6f} q={:.3f}".format(
                pid,
                vfl._rldbg_risk_sum[pid] / steps,
                float(vfl.proxy_risk_threshold_pct),
                prior_text,
                vfl._rldbg_quality_sum[pid] / steps,
                triggers,
                decisions,
                trigger_rate,
                current_fwd_c,
                current_bwd_c,
                vfl._rldbg_bwd_eps_sum[pid] / steps,
                vfl._rldbg_q_sum[pid] / steps,
            )
        )
    print("[RLDBG E{}/{}] {}".format(epoch + 1, vfl.args.epochs, " | ".join(rows)))
