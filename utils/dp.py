import torch
import numpy as np
import math
from opacus.accountants.analysis import rdp as rdp_analysis

RDP_ORDERS = np.array([
    1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0,
    8.0, 10.0, 13.0, 16.0, 20.0, 25.0, 32.0, 45.0, 64.0, 90.0,
    128.0, 256.0, 512.0, 1024.0,
])


def _gaussian_noise_multiplier_from_step_epsilon(step_epsilon, delta=1e-5):
    step_epsilon = float(max(step_epsilon, 1e-12))
    noise_factor = np.sqrt(2 * np.log(1.25 / delta))
    raw_nm = noise_factor / step_epsilon
    return max(min(raw_nm, 500.0), 0.01)


def _step_rdp_from_epsilon(step_epsilon, q, steps, delta=1e-5, orders=None):
    if orders is None:
        orders = RDP_ORDERS
    if step_epsilon <= 1e-12 or steps <= 0:
        return np.zeros(len(orders), dtype=float)
    safe_q = max(float(q), 1e-7)
    noise_multiplier = _gaussian_noise_multiplier_from_step_epsilon(step_epsilon, delta=delta)
    return rdp_analysis.compute_rdp(
        q=safe_q,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=orders,
    )


def _epsilon_from_rdp(rdp_vector, delta=1e-5, orders=None):
    if orders is None:
        orders = RDP_ORDERS
    eps_at_alphas = rdp_vector + (np.log(1.0 / delta) / (orders - 1))
    return float(np.min(eps_at_alphas))


def compute_step_epsilon_for_target(target_eps, total_steps, q, delta=1e-5, orders=None):
    if orders is None:
        orders = RDP_ORDERS

    target_eps = float(max(target_eps, 1e-8))
    total_steps = int(max(total_steps, 1))
    safe_q = max(float(q), 1e-7)

    low, high = 0.01, 100.0
    for _ in range(30):
        mid = (low + high) / 2
        rdp = rdp_analysis.compute_rdp(q=safe_q, noise_multiplier=mid, steps=total_steps, orders=orders)
        eps_at_alphas = rdp + (np.log(1.0 / delta) / (orders - 1))

        if np.min(eps_at_alphas) < target_eps:
            high = mid
        else:
            low = mid

    noise_factor = np.sqrt(2 * np.log(1.25 / delta))
    return float(noise_factor / high)


def compute_epsilon_from_constant_step(step_epsilon, total_steps, q, delta=1e-5, orders=None):
    if orders is None:
        orders = RDP_ORDERS
    rdp = _step_rdp_from_epsilon(step_epsilon, q=q, steps=total_steps, delta=delta, orders=orders)
    return _epsilon_from_rdp(rdp, delta=delta, orders=orders)


def project_final_average_epsilon(
    current_rdp_history,
    remaining_steps,
    q,
    future_step_eps_list,
    delta=1e-5,
    orders=None,
):
    if orders is None:
        orders = RDP_ORDERS

    current_rdp_history = np.asarray(current_rdp_history, dtype=float)
    if current_rdp_history.ndim == 1:
        current_rdp_history = current_rdp_history[None, :]

    future_rdp = np.zeros(len(orders), dtype=float)
    for step_eps in future_step_eps_list:
        future_rdp += _step_rdp_from_epsilon(
            step_eps,
            q=q,
            steps=remaining_steps,
            delta=delta,
            orders=orders,
        )

    total_eps = 0.0
    for i in range(current_rdp_history.shape[0]):
        total_eps += _epsilon_from_rdp(current_rdp_history[i] + future_rdp, delta=delta, orders=orders)
    return total_eps / max(current_rdp_history.shape[0], 1)


def solve_backward_step_epsilon_for_total(
    current_rdp_history,
    remaining_steps,
    q,
    total_target_eps,
    forward_step_epsilon,
    delta=1e-5,
    orders=None,
    initial_high=10.0,
):
    if orders is None:
        orders = RDP_ORDERS

    remaining_steps = int(max(remaining_steps, 0))
    if remaining_steps <= 0:
        return 0.0

    low, high = 1e-8, max(float(initial_high), 1e-6)
    projected_high = project_final_average_epsilon(
        current_rdp_history=current_rdp_history,
        remaining_steps=remaining_steps,
        q=q,
        future_step_eps_list=[forward_step_epsilon, high],
        delta=delta,
        orders=orders,
    )
    while projected_high < total_target_eps and high < 1e4:
        high *= 2.0
        projected_high = project_final_average_epsilon(
            current_rdp_history=current_rdp_history,
            remaining_steps=remaining_steps,
            q=q,
            future_step_eps_list=[forward_step_epsilon, high],
            delta=delta,
            orders=orders,
        )

    for _ in range(30):
        mid = (low + high) / 2.0
        projected_mid = project_final_average_epsilon(
            current_rdp_history=current_rdp_history,
            remaining_steps=remaining_steps,
            q=q,
            future_step_eps_list=[forward_step_epsilon, mid],
            delta=delta,
            orders=orders,
        )
        if projected_mid < total_target_eps:
            low = mid
        else:
            high = mid
    return float(max(low, 1e-8))


def compute_step_epsilon(args, steps_per_epoch, q, target_eps=None):

    target_eps = args.epsilon if target_eps is None else target_eps
    epochs = args.epochs
    total_steps = epochs * steps_per_epoch
    return compute_step_epsilon_for_target(target_eps, total_steps, q, delta=1e-5, orders=RDP_ORDERS)
def compute_epsilon(args, actual_total_steps):
    """
    借鉴 dp_noise 的 RDP 逻辑，反推单步隐私消耗
    """
    from opacus.accountants.analysis import rdp as rdp_analysis
    
    target_eps = args.epsilon
    epochs = args.epochs
    batch_size = args.batch_size
    delta = 1e-5

    total_steps = actual_total_steps
    # 计算总迭代步数 T 和采样率 q
    q = args.epochs / actual_total_steps

    # 预设 alpha 集合
    alphas = np.array([1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 
                       8.0, 10.0, 13.0, 16.0, 20.0, 25.0, 32.0, 45.0, 64.0, 90.0, 128.0])
    
    # 二分搜索噪声倍数 nm
    low, high = 0.01, 100.0
    for _ in range(20):
        mid = (low + high) / 2
        rdp = rdp_analysis.compute_rdp(q=q, noise_multiplier=mid, steps=total_steps, orders=alphas)
        eps_at_alphas = rdp + (np.log(1.0 / delta) / (alphas - 1))
        if np.min(eps_at_alphas) < target_eps:
            high = mid
        else:
            low = mid
    
    # 将最终搜索到的 noise_multiplier 反推为单步 epsilon 用于账户核算
    noise_factor = np.sqrt(2 * np.log(1.25 / delta))
    step_epsilon = noise_factor / high 
    return step_epsilon
def dp_defense(grad, args, device='cpu'):
    """
    Apply DP defense to the specific attack party's gradient.
    Supports both Gaussian (L2) and Laplace (L1) mechanisms.
    """
    # 获取攻击者的梯度 (args.attack_id)
    flatten_grad = grad[args.attack_id].flatten()
    grad_norm = abs(flatten_grad.norm(dim=0, p=2))
    
    # Clip gradient
    # 注意：这里假设 args 里面有 epsilon，如果没有传参则可能报错，所以确保 args 完整
    epsilon = args.epsilon
    
    clip_grad = flatten_grad.clip(-grad_norm, grad_norm).reshape(grad[args.attack_id].shape)
    
    # Get mechanism type
    mechanism = getattr(args, 'mechanism', 'gaussian')
    
    if mechanism == 'laplace':
        # Laplace Noise: Scale = Sensitivity / Epsilon
        scale = grad_norm / epsilon
        noise = torch.distributions.laplace.Laplace(0, scale).sample(clip_grad.shape).to(device)
    else:
        # Gaussian Noise
        delta = 1e-5
        gaussian_factor = np.sqrt(2 * np.log(1.25 / delta))
        sigma = (grad_norm * gaussian_factor) / epsilon
        noise = torch.randn_like(clip_grad, device=device) * sigma

    noise_grad = clip_grad + noise
    
    # Reconstruct the gradient tuple
    noise_grad_list = []
    for i in range(args.num_passive):
        if i == args.attack_id:
            noise_grad_list.append(noise_grad)
        else:
            noise_grad_list.append(grad[i])
    
    return tuple(noise_grad_list)


def ensure_dp_noise_plan(args, steps_per_epoch, q):
    state = getattr(args, '_dp_noise_plan', None)
    if state is not None:
        return state

    total_steps = int(max(args.epochs * steps_per_epoch, 1))
    sample_rate = max(float(q), 1e-7)
    total_target_eps = float(args.epsilon)
    mode = getattr(args, 'budget_sweep_mode', 'joint')
    forward_ratio = float(np.clip(getattr(args, 'forward_budget_ratio', 0.3), 1e-6, 1.0 - 1e-6))
    manual_forward_step_eps = float(getattr(args, 'forward_fixed_eps', -1.0))

    if mode == 'forward_only':
        forward_ratio = 1.0
        forward_target_eps = total_target_eps
        forward_step_eps = compute_step_epsilon_for_target(
            forward_target_eps,
            total_steps,
            sample_rate,
            delta=1e-5,
            orders=RDP_ORDERS,
        )
        backward_step_eps = 0.0
        backward_target_eps = 0.0
    elif mode == 'backward_only':
        forward_ratio = 0.0
        forward_target_eps = 0.0
        forward_step_eps = 0.0
        backward_step_eps = compute_step_epsilon_for_target(
            total_target_eps,
            total_steps,
            sample_rate,
            delta=1e-5,
            orders=RDP_ORDERS,
        )
        backward_target_eps = total_target_eps
    else:
        if manual_forward_step_eps > 0.0:
            forward_step_eps = manual_forward_step_eps
            forward_target_eps = compute_epsilon_from_constant_step(
                forward_step_eps,
                total_steps,
                sample_rate,
                delta=1e-5,
                orders=RDP_ORDERS,
            )
        else:
            forward_target_eps = total_target_eps * forward_ratio
            forward_step_eps = compute_step_epsilon_for_target(
                forward_target_eps,
                total_steps,
                sample_rate,
                delta=1e-5,
                orders=RDP_ORDERS,
            )

        backward_step_eps = solve_backward_step_epsilon_for_total(
            current_rdp_history=np.zeros((args.num_passive, len(RDP_ORDERS)), dtype=float),
            remaining_steps=total_steps,
            q=sample_rate,
            total_target_eps=total_target_eps,
            forward_step_epsilon=forward_step_eps,
            delta=1e-5,
            orders=RDP_ORDERS,
            initial_high=max(forward_step_eps, 1.0),
        )
        backward_target_eps = compute_epsilon_from_constant_step(
            backward_step_eps,
            total_steps,
            sample_rate,
            delta=1e-5,
            orders=RDP_ORDERS,
        )

    state = {
        'mode': mode,
        'forward_enabled': bool(mode != 'backward_only'),
        'backward_enabled': bool(mode != 'forward_only'),
        'forward_step_eps': float(forward_step_eps),
        'backward_step_eps': float(backward_step_eps),
        'forward_target_eps': float(forward_target_eps),
        'backward_target_eps': float(backward_target_eps),
        'forward_ratio': float(forward_ratio),
    }

    noise_factor = np.sqrt(2 * np.log(1.25 / 1e-5))
    sensitivity_scale = 2.0
    forward_clip = float(getattr(args, 'clip_threshold_forward', 1.0))
    backward_clip = float(getattr(args, 'clip_threshold', 1.0))
    state['forward_sigma'] = (
        0.0 if forward_step_eps <= 0.0 else
        float((noise_factor / max(forward_step_eps, 1e-12)) * forward_clip * sensitivity_scale)
    )
    state['backward_sigma'] = (
        0.0 if backward_step_eps <= 0.0 else
        float((noise_factor / max(backward_step_eps, 1e-12)) * backward_clip * sensitivity_scale)
    )
    state['forward_clip'] = forward_clip
    state['backward_clip'] = backward_clip
    state['sensitivity_scale'] = sensitivity_scale

    setattr(args, '_dp_noise_plan', state)

    if not getattr(args, '_dp_noise_logged', False):
        if mode == 'joint':
            print(
                f"[dp] mode = joint | forward budget ratio = {forward_ratio * 100:.1f}% "
                f"(view eps {forward_target_eps:.4f}/{total_target_eps:.4f})"
            )
        elif mode == 'forward_only':
            print(
                f"[dp] mode = forward_only | all budget assigned to forward path "
                f"(view eps {forward_target_eps:.4f}/{total_target_eps:.4f})"
            )
        else:
            print(
                f"[dp] mode = backward_only | all budget assigned to backward path "
                f"(view eps {backward_target_eps:.4f}/{total_target_eps:.4f})"
            )
        print(
            f"[dp] forward: step_eps={forward_step_eps:.6f}, "
            f"clip={forward_clip:.6f}, sigma={state['forward_sigma']:.6f}"
        )
        print(
            f"[dp] backward: step_eps={backward_step_eps:.6f}, "
            f"clip={backward_clip:.6f}, sigma={state['backward_sigma']:.6f}, "
            f"view eps={backward_target_eps:.4f}"
        )
        setattr(args, '_dp_noise_logged', True)

    return state


def _dp_clip_and_noise(tensor, clip_threshold, step_epsilon, device, return_stats=False):
    clip_threshold = float(max(clip_threshold, 1e-8))
    batch_size = tensor.shape[0]
    flat_tensor = tensor.view(batch_size, -1)
    sample_norm = torch.norm(flat_tensor, p=2, dim=1, keepdim=True)
    clip_coef = torch.clamp(clip_threshold / (sample_norm + 1e-9), max=1.0)
    clipped = tensor * clip_coef.view([batch_size] + [1] * (tensor.dim() - 1))

    noise_factor = np.sqrt(2 * np.log(1.25 / 1e-5))
    noise_multiplier = noise_factor / max(float(step_epsilon), 1e-12)
    sensitivity_scale = 2.0
    sigma = noise_multiplier * clip_threshold * sensitivity_scale
    noise = torch.randn_like(clipped, device=device) * sigma
    noised = clipped + noise

    if not return_stats:
        return noised

    sample_norm_flat = sample_norm.view(-1)
    quantiles = torch.tensor([0.5, 0.9, 0.95], device=sample_norm_flat.device, dtype=sample_norm_flat.dtype)
    norm_q = torch.quantile(sample_norm_flat.detach(), quantiles)
    stats = {
        "clip_threshold": clip_threshold,
        "step_epsilon": float(step_epsilon),
        "noise_multiplier": float(noise_multiplier),
        "sensitivity_scale": float(sensitivity_scale),
        "sigma": float(sigma),
        "mean_norm": float(sample_norm_flat.mean().item()),
        "norm_p50": float(norm_q[0].item()),
        "norm_p90": float(norm_q[1].item()),
        "norm_p95": float(norm_q[2].item()),
        "norm_max": float(sample_norm_flat.max().item()),
        "clip_hit_rate": float((sample_norm_flat > clip_threshold).float().mean().item()),
        "mean_clip_coef": float(clip_coef.mean().item()),
    }
    return noised, stats


def dp_defense_adaptive(
    grad,
    passive_id,
    epsilon,
    device,
    adaptive_state,
    target_quantile=0.5,
    lr_c=0.2,
    initial_c=0.1,
    min_c=1e-8,
    max_c=None,
):

    # 1. 获取并初始化裁剪阈值 C [cite: 158]
    if passive_id not in adaptive_state:
        adaptive_state[passive_id] = float(max(initial_c, min_c))
    
    current_C = float(max(adaptive_state[passive_id], min_c))
    if max_c is not None:
        current_C = float(min(current_C, max_c))
    adaptive_state[passive_id] = current_C
    
    # 2. 计算全局 L2 范数 
    grad_norm = torch.norm(grad, p=2)
    
    # 3. 计算本轮是否被裁剪的指示器 b [cite: 130]
    # b = 1 if norm <= C else 0
    b = 1.0 if grad_norm <= current_C else 0.0
    
    # 4. 对指示器 b 加噪以保护分位数隐私 (论文 Algorithm 1) [cite: 132]
    # 在 VFL 这种单参与方决策场景下，我们直接使用当前步的 epsilon 进行扰动
    delta = 1e-5
    std_b = 1.0 / epsilon # 简化处理，确保分位数估计也满足 DP
    b_noisy = b + np.random.normal(0, std_b)
    
    # 5. 使用几何更新规则更新下一轮的 C [cite: 122, 134]
    # C_{t+1} = C_t * exp(-lr_c * (b_noisy - gamma))
    new_C = current_C * math.exp(-lr_c * (b_noisy - target_quantile))
    new_C = float(max(new_C, min_c))
    if max_c is not None:
        new_C = float(min(new_C, max_c))
    adaptive_state[passive_id] = new_C
    
    # 6. 执行全局范数裁剪 [cite: 30, 134]
    clip_coef = min(1.0, current_C / (grad_norm + 1e-9))
    grad_clipped = grad * clip_coef
    
    # 7. 注入高斯噪声 [cite: 30, 134]
    gaussian_factor = np.sqrt(2 * np.log(1.25 / delta))
    sigma = (current_C * gaussian_factor) / epsilon
    noise = torch.randn_like(grad, device=device) * sigma
    
    # 诊断打印
    if np.random.rand() < 0.02: # 约每 50 步打印一次
        # 计算当前梯度的范数用于对比
        grad_norm = torch.norm(grad, p=2).item()
        # 计算 SNR
        delta = 1e-5
        gaussian_factor = np.sqrt(2 * np.log(1.25 / delta))
        sigma = (current_C * gaussian_factor) / epsilon
        snr = grad_norm / (sigma + 1e-9)

        print(f"[ACVFL-Check] Party:{passive_id} | C:{current_C:.6f} | GradNorm:{grad_norm:.6f} | SNR:{snr:.4f} | "
              f"Match:{'Below-C' if grad_norm <= current_C else 'Above-C'}")
    return grad_clipped + noise


def dp_noise(grad, args, device='cpu', steps_per_epoch=None, q=None, direction='backward', return_stats=False):
    if steps_per_epoch is None or q is None:
        raise ValueError("dp_noise requires steps_per_epoch and q.")

    plan = ensure_dp_noise_plan(args, steps_per_epoch=steps_per_epoch, q=q)
    if direction == 'forward':
        clip_threshold = getattr(args, 'clip_threshold_forward', 1.0)
        step_epsilon = plan['forward_step_eps']
    else:
        clip_threshold = getattr(args, 'clip_threshold', 1.0)
        step_epsilon = plan['backward_step_eps']

    is_tuple = isinstance(grad, tuple)
    is_list = isinstance(grad, list)
    tensors = list(grad) if (is_tuple or is_list) else [grad]
    if step_epsilon <= 0.0:
        if return_stats:
            zero_stats = []
            for tensor in tensors:
                flat_tensor = tensor.view(tensor.shape[0], -1)
                sample_norm = torch.norm(flat_tensor, p=2, dim=1)
                stats = {
                    "clip_threshold": float(clip_threshold),
                    "step_epsilon": 0.0,
                    "noise_multiplier": 0.0,
                    "sensitivity_scale": 2.0,
                    "sigma": 0.0,
                    "mean_norm": float(sample_norm.mean().item()),
                    "norm_p50": float(torch.quantile(sample_norm.detach(), 0.5).item()),
                    "norm_p90": float(torch.quantile(sample_norm.detach(), 0.9).item()),
                    "norm_p95": float(torch.quantile(sample_norm.detach(), 0.95).item()),
                    "norm_max": float(sample_norm.max().item()),
                    "clip_hit_rate": 0.0,
                    "mean_clip_coef": 1.0,
                }
                zero_stats.append(stats)
            if is_tuple:
                return tuple(tensors), 0.0, zero_stats
            if is_list:
                return tensors, 0.0, zero_stats
            return tensors[0], 0.0, zero_stats[0]
        if is_tuple:
            return tuple(tensors), 0.0
        if is_list:
            return tensors, 0.0
        return tensors[0], 0.0
    results = [
        _dp_clip_and_noise(tensor, clip_threshold, step_epsilon, device, return_stats=return_stats)
        for tensor in tensors
    ]
    if return_stats:
        noisy_tensors = [item[0] for item in results]
        stats_list = [item[1] for item in results]
    else:
        noisy_tensors = results
        stats_list = None

    if is_tuple:
        if return_stats:
            return tuple(noisy_tensors), step_epsilon, stats_list
        return tuple(noisy_tensors), step_epsilon
    if is_list:
        if return_stats:
            return noisy_tensors, step_epsilon, stats_list
        return noisy_tensors, step_epsilon
    if return_stats:
        return noisy_tensors[0], step_epsilon, stats_list[0]
    return noisy_tensors[0], step_epsilon


def apply_random_sparsification(tensors, keep_rate, return_stats=False):
    keep_rate = float(keep_rate)
    if not (0.0 < keep_rate <= 1.0):
        raise ValueError("keep_rate should be in the interval (0, 1].")

    is_tuple = isinstance(tensors, tuple)
    is_list = isinstance(tensors, list)
    tensor_list = list(tensors) if (is_tuple or is_list) else [tensors]

    sparse_tensors = []
    stats_list = []
    for tensor in tensor_list:
        if keep_rate >= 1.0:
            sparse_tensor = tensor
            actual_keep_rate = 1.0
            zero_ratio = 0.0
        else:
            mask = (torch.rand_like(tensor, dtype=torch.float32) < keep_rate).to(tensor.dtype)
            sparse_tensor = tensor * mask
            actual_keep_rate = float(mask.float().mean().item())
            zero_ratio = 1.0 - actual_keep_rate
        sparse_tensors.append(sparse_tensor)
        stats_list.append(
            {
                "target_keep_rate": keep_rate,
                "actual_keep_rate": actual_keep_rate,
                "zero_ratio": zero_ratio,
            }
        )

    if is_tuple:
        packed = tuple(sparse_tensors)
    elif is_list:
        packed = sparse_tensors
    else:
        packed = sparse_tensors[0]

    if return_stats:
        if is_tuple or is_list:
            return packed, stats_list
        return packed, stats_list[0]
    return packed


def apply_importance_sparsification(tensor, keep_rate, importance_prior=None, random_ratio=0.0, return_stats=False):
    keep_rate = float(keep_rate)
    random_ratio = float(random_ratio)
    if not (0.0 < keep_rate <= 1.0):
        raise ValueError("keep_rate should be in the interval (0, 1].")
    if not (0.0 <= random_ratio <= 1.0):
        raise ValueError("random_ratio should be in the interval [0, 1].")
    if tensor.dim() < 2:
        raise ValueError(f"Expected batched tensor, got shape {tuple(tensor.shape)}")

    flat = tensor.reshape(tensor.shape[0], -1)
    feature_dim = flat.shape[1]
    keep_count = min(feature_dim, max(1, int(math.ceil(feature_dim * keep_rate))))

    if keep_count >= feature_dim:
        mask = torch.ones(feature_dim, device=flat.device, dtype=flat.dtype)
    else:
        random_keep = min(keep_count, int(round(keep_count * random_ratio)))
        guided_keep = keep_count - random_keep
        mask = torch.zeros(feature_dim, device=flat.device, dtype=flat.dtype)

        candidate = None
        if importance_prior is not None:
            prior = importance_prior.to(device=flat.device, dtype=torch.float32).view(-1)
            if prior.numel() == feature_dim:
                candidate = prior

        selected = torch.empty(0, device=flat.device, dtype=torch.long)
        if guided_keep > 0 and candidate is not None:
            selected = torch.topk(candidate, k=guided_keep, largest=True, sorted=False).indices
            mask[selected] = 1.0

        remaining = keep_count - int(mask.sum().item())
        if remaining > 0:
            available = torch.nonzero(mask < 0.5, as_tuple=False).view(-1)
            if available.numel() > 0:
                perm = torch.randperm(available.numel(), device=flat.device)[:remaining]
                mask[available[perm]] = 1.0

    sparse_flat = flat * mask.unsqueeze(0)
    sparse = sparse_flat.view_as(tensor)
    if not return_stats:
        return sparse

    actual_keep_rate = float(mask.float().mean().item())
    pre_norm = torch.norm(flat.detach(), p=2, dim=1).mean().item()
    post_norm = torch.norm(sparse_flat.detach(), p=2, dim=1).mean().item()
    stats = {
        "target_keep_rate": keep_rate,
        "actual_keep_rate": actual_keep_rate,
        "zero_ratio": 1.0 - actual_keep_rate,
        "guided_keep_rate": float(mask.sum().item()) / float(feature_dim),
        "random_ratio": random_ratio,
        "pre_norm": float(pre_norm),
        "post_norm": float(post_norm),
        "norm_ratio": float(post_norm / max(pre_norm, 1e-12)),
    }
    return sparse, stats


def dp_perturb_adaptive(grad, passive_id, epsilon, device, history_grads=None, mechanism='gaussian', current_step=-1):
    # 保持原有初始化逻辑
    clip_factor = 0.5
    use_history = False
    beta = 0.5
    
    if (history_grads is not None and passive_id in history_grads and 
        history_grads[passive_id] is not None and 
        history_grads[passive_id].shape == grad.shape):
        use_history = True

    if not use_history:
        threshold = torch.abs(grad).detach()
    else:
        threshold = clip_factor * torch.abs(history_grads[passive_id])

    min_physical_threshold = 0.0005 
    threshold = torch.clamp(threshold, min=min_physical_threshold)
    grad_clipped = torch.min(torch.max(grad, -threshold), threshold)

    if epsilon > 1e-6:
        if mechanism == 'laplace':
            scale_tensor = threshold / epsilon
            m = torch.distributions.laplace.Laplace(0, scale_tensor)
            noise = m.sample().to(device)
        else: 
            delta = 1e-5
            gaussian_factor = np.sqrt(2 * np.log(1.25 / delta))
            sigma_tensor = (threshold * gaussian_factor ) / epsilon
            noise = torch.randn_like(grad, device=device) * sigma_tensor
        noise_grad = grad_clipped + noise
    else:
        noise_grad = grad_clipped

    current_grad_detached = grad.detach().clone()
    if history_grads is not None:
        if use_history:
            history_grads[passive_id] = beta * history_grads[passive_id] + \
                                        (1 - beta) * current_grad_detached
        else:
            history_grads[passive_id] = current_grad_detached

    return noise_grad


def dp_perturb_sampling_adaptive(
    grad,
    passive_id,
    epsilon,
    sampling_rate,
    device,
    fixed_mask=None,
    history_grads=None,
    norm_tracker=None,
    beta=0.9,
    initial_clip=1.0,
    sensitivity_scale=2.0,
    clip_growth_limit=2.0,
    return_stats=True,
):

    """
    Delayed-EMA + strict clipping + independent Gaussian noise + sparse release.

    The current round uses only the previously released safe state `C_{t-1}` to clip the
    private gradient. The next clip bound is updated from the noisy released gradient norm,
    so the adaptive state remains post-processing of DP outputs.
    """
    q = max(min(sampling_rate, 1.0), 0.05)
    prev_clip = max(float(initial_clip), 1e-8)
    if norm_tracker is not None and passive_id in norm_tracker:
        prev_clip = max(float(norm_tracker[passive_id]), 1e-8)

    current_norm = torch.norm(grad, p=2)
    clip_coef = min(1.0, prev_clip / (current_norm.item() + 1e-9))
    grad_clipped = grad * clip_coef
    clipped_norm = torch.norm(grad_clipped, p=2).item() if return_stats else None

    current_sigma = 0.0
    noise_std = 0.0
    if epsilon > 1e-6:
        delta = 1e-5
        gaussian_factor = np.sqrt(2 * np.log(1.25 / delta))
        current_sigma = gaussian_factor / epsilon
        sensitivity = max(float(sensitivity_scale) * prev_clip, 1e-8)
        noise_std = sensitivity * current_sigma
        noise = torch.randn_like(grad_clipped, device=device) * noise_std
        grad_released = grad_clipped + noise
    else:
        grad_released = grad_clipped

    grad_released = torch.nan_to_num(grad_released, nan=0.0, posinf=0.0, neginf=0.0)
    if fixed_mask is not None:
        if fixed_mask.dim() == 1:
            per_sample_shape = list(grad_released.shape[1:])
            expected_numel = int(grad_released[0].numel())
            if fixed_mask.numel() != expected_numel:
                raise ValueError(
                    "fixed_mask length {} does not match per-sample gradient size {} for shape {}".format(
                        fixed_mask.numel(),
                        expected_numel,
                        tuple(grad_released.shape),
                    )
                )
            view_shape = [1] + per_sample_shape
            mask = fixed_mask.view(view_shape).to(device=device, dtype=grad_released.dtype)
            mask = mask.expand_as(grad_released)
        else:
            mask = fixed_mask.to(device=device, dtype=grad_released.dtype)
            if mask.shape != grad_released.shape:
                mask = mask.expand_as(grad_released)
        if return_stats:
            effective_keep_rate = float(mask.reshape(-1).mean().item())
            mask_mode = "public_topk"
    else:
        mask = (torch.rand_like(grad_released) < q).float().to(device)
        if return_stats:
            effective_keep_rate = float(q)
            mask_mode = "random_keep"
    final_grad = grad_released * mask
    final_grad = torch.nan_to_num(final_grad, nan=0.0, posinf=0.0, neginf=0.0)
    released_norm = None
    debiased_norm = None
    safe_norm = None
    if norm_tracker is not None:
        released_norm = torch.norm(grad_released, p=2).item()
        noise_floor = math.sqrt(float(grad_released.numel())) * noise_std
        debiased_norm = max(released_norm - noise_floor, 0.0)
        min_clip = max(float(initial_clip), 1e-8)
        max_clip = max(prev_clip, min_clip) * max(float(clip_growth_limit), 1.0)
        safe_norm = min(max(debiased_norm, min_clip), max_clip)
        norm_tracker[passive_id] = beta * prev_clip + (1 - beta) * safe_norm

    if not return_stats:
        return final_grad, None

    final_norm = torch.norm(final_grad, p=2).item()
    structure_score = final_norm
    if final_grad.dim() >= 2 and final_grad.shape[0] > 1:
        flat_final = final_grad.view(final_grad.size(0), -1)
        row_norms = torch.norm(flat_final, p=2, dim=1)
        if row_norms.numel() > 1:
            structure_score = row_norms.std().item() + row_norms.mean().item()

    if released_norm is None:
        released_norm = torch.norm(grad_released, p=2).item()
        debiased_norm = released_norm
        safe_norm = released_norm
    non_zero = torch.count_nonzero(final_grad).item()
    total = final_grad.numel()
    real_sparsity = 1.0 - (non_zero / total) if total > 0 else 1.0
    stats = {
        "sparsity": real_sparsity,
        "sigma": current_sigma,
        "noise_std": noise_std,
        "clip": prev_clip,
        "norm": current_norm.item(),
        "clip_coef": clip_coef,
        "clipped_norm": clipped_norm,
        "released_norm": released_norm,
        "debiased_norm": debiased_norm,
        "next_clip_target": safe_norm,
        "final_norm": final_norm,
        "structure_score": structure_score,
        "expected_noise_l2": math.sqrt(float(grad_released.numel())) * noise_std,
        "snr": clipped_norm / (math.sqrt(float(grad_released.numel())) * noise_std + 1e-12),
        "released_grad": grad_released.detach().clone(),
        "keep_rate": effective_keep_rate,
        "mask_mode": mask_mode,
    }
    return final_grad, stats


def build_importance_weights(
    importance_prior,
    feature_dim,
    device,
    dtype,
    importance_strength=2.0,
    min_eps_ratio=0.5,
    max_eps_ratio=2.0,
    band_start=1.0,
    band_end=1.0,
    band_damping=0.0,
    band_power=1.0,
    tail_start=1.0,
    tail_damping=0.0,
    tail_power=2.0,
):
    weights = torch.ones(feature_dim, device=device, dtype=dtype)
    if importance_prior is None:
        return weights

    prior = importance_prior.to(device=device, dtype=dtype).view(-1)
    if prior.numel() != feature_dim:
        return weights

    prior = torch.nan_to_num(prior, nan=0.5, posinf=0.5, neginf=0.5)
    p_min, p_max = prior.min(), prior.max()
    if float((p_max - p_min).item()) > 1e-8:
        prior = (prior - p_min) / (p_max - p_min)
    else:
        prior = torch.ones_like(prior) * 0.5

    safe_band_start = float(np.clip(band_start, 0.0, 0.999))
    safe_band_end = float(np.clip(band_end, safe_band_start + 1e-6, 1.0))
    safe_band_damping = float(max(band_damping, 0.0))
    safe_band_power = float(max(band_power, 1.0))
    band_width = max(safe_band_end - safe_band_start, 1e-6)
    if safe_band_damping > 1e-8 and safe_band_start < 0.999:
        band_unit = torch.clamp((prior - safe_band_start) / band_width, min=0.0, max=1.0)
        band_shift = safe_band_damping * band_width * band_unit.pow(safe_band_power)
        band_shift = torch.where(prior > safe_band_end, torch.full_like(prior, safe_band_damping * band_width), band_shift)
        prior = torch.clamp(prior - band_shift, 0.0, 1.0)

    safe_tail_start = float(np.clip(tail_start, 0.0, 0.999))
    safe_tail_damping = float(max(tail_damping, 0.0))
    safe_tail_power = float(max(tail_power, 1.0))
    if safe_tail_damping > 1e-8 and safe_tail_start < 0.999:
        tail = torch.clamp((prior - safe_tail_start) / max(1.0 - safe_tail_start, 1e-8), min=0.0)
        prior = prior - safe_tail_damping * tail.pow(safe_tail_power)
        prior = torch.clamp(prior, 0.0, 1.0)

    # Keep the map smooth and conservative: uncertain middle-ranked dimensions
    # stay near 1.0, while only clearly important / unimportant coordinates
    # receive stronger reweighting.
    centered = prior - 0.5
    deadzone = 0.10
    abs_centered = torch.abs(centered)
    scaled = torch.zeros_like(centered)
    active = abs_centered > deadzone
    scaled[active] = torch.sign(centered[active]) * (
        (abs_centered[active] - deadzone) / max(0.5 - deadzone, 1e-8)
    )
    scaled = torch.tanh(float(importance_strength) * scaled)

    safe_min = max(float(min_eps_ratio), 1e-4)
    safe_max = max(float(max_eps_ratio), safe_min)
    log_span = min(abs(np.log(safe_max + 1e-8)), abs(np.log(safe_min + 1e-8)))
    log_span = max(log_span, 1e-6)
    weights = torch.exp(log_span * scaled)
    weights = weights / (weights.mean() + 1e-8)
    weights = torch.clamp(weights, min=safe_min, max=safe_max)

    weights = weights / (weights.mean() + 1e-8)
    return torch.clamp(weights, min=1e-4)

def dp_forward_perturb_adaptive(
    emb,
    passive_id,
    epsilon,
    device,
    norm_tracker=None,
    beta=0.95,
    initial_clip=1.0,
    sensitivity_scale=1.0,
    clip_growth_limit=1.2,
    clip_target=0.35,
    clip_tolerance=0.05,
    clip_lr=0.05,
    clip_min_ratio=1.0,
    allow_clip_decay=False,
    importance_prior=None,
    importance_strength=2.0,
    min_eps_ratio=0.5,
    max_eps_ratio=2.0,
    band_start=1.0,
    band_end=1.0,
    band_damping=0.0,
    band_power=1.0,
    tail_start=1.0,
    tail_damping=0.0,
    tail_power=2.0,
    importance_weights=None,
    return_stats=True,
    return_public_stats=False,
):
    """
    RL-only forward-path adaptive perturbation with strict weighted clipping.

    We first transform the embedding into a whitened space using a public
    task-aware diagonal metric W, clip in that weighted space, add isotropic
    Gaussian noise there, and finally map the result back to the original
    coordinates. This keeps the accountant aligned with a standard Gaussian
    mechanism in the weighted space while allowing heterogeneous noise in the
    original feature dimensions.
    """
    if emb.dim() < 2:
        raise ValueError(f"Expected batched embedding tensor, got shape {tuple(emb.shape)}")

    batch_size = emb.shape[0]
    flat_emb = emb.view(batch_size, -1)
    feature_dim = flat_emb.shape[1]
    raw_sample_norms = torch.norm(flat_emb, p=2, dim=1) if return_stats else None

    if importance_weights is None:
        weights = build_importance_weights(
            importance_prior=importance_prior,
            feature_dim=feature_dim,
            device=flat_emb.device,
            dtype=flat_emb.dtype,
            importance_strength=importance_strength,
            min_eps_ratio=min_eps_ratio,
            max_eps_ratio=max_eps_ratio,
            band_start=band_start,
            band_end=band_end,
            band_damping=band_damping,
            band_power=band_power,
            tail_start=tail_start,
            tail_damping=tail_damping,
            tail_power=tail_power,
        )
    else:
        weights = importance_weights.to(device=flat_emb.device, dtype=flat_emb.dtype)

    sqrt_weights = torch.sqrt(weights).unsqueeze(0)
    flat_weighted = flat_emb * sqrt_weights
    weighted_sample_norms = torch.norm(flat_weighted, p=2, dim=1)

    if return_stats:
        quantiles = torch.tensor([0.5, 0.9, 0.95], device=flat_emb.device, dtype=flat_emb.dtype)
        raw_norm_quantiles = torch.quantile(raw_sample_norms.detach(), quantiles)
        weighted_norm_quantiles = torch.quantile(weighted_sample_norms.detach(), quantiles)

    clip_reference = max(float(initial_clip), 1e-8)
    clip_floor = max(float(clip_min_ratio) * clip_reference, 1e-8)
    prev_clip = clip_reference
    if norm_tracker is not None and passive_id in norm_tracker:
        prev_clip = max(float(norm_tracker[passive_id]), 1e-8)

    sample_norms = weighted_sample_norms.unsqueeze(1)
    clip_coef = torch.clamp(prev_clip / (sample_norms + 1e-9), max=1.0)
    flat_weighted_clipped = flat_weighted * clip_coef
    flat_clipped = flat_weighted_clipped / sqrt_weights
    emb_clipped = flat_clipped.view_as(emb)
    raw_clipped_sample_norms = torch.norm(flat_clipped, p=2, dim=1) if return_stats else None

    current_sigma = 0.0
    weighted_noise_std = 0.0
    orig_noise_std_mean = 0.0
    orig_noise_std_min = 0.0
    orig_noise_std_max = 0.0
    if epsilon > 1e-6:
        delta = 1e-5
        gaussian_factor = np.sqrt(2 * np.log(1.25 / delta))
        current_sigma = gaussian_factor / epsilon
        sensitivity = max(float(sensitivity_scale) * prev_clip, 1e-8)
        weighted_noise_std = sensitivity * current_sigma
        weighted_noise = torch.randn_like(flat_weighted_clipped, device=device) * weighted_noise_std
        flat_weighted_released = flat_weighted_clipped + weighted_noise
        flat_released = flat_weighted_released / sqrt_weights
        emb_released = flat_released.view_as(emb)

        if return_stats:
            orig_noise_std = (weighted_noise_std / sqrt_weights.squeeze(0)).to(flat_emb.dtype)
            orig_noise_std_mean = float(orig_noise_std.mean().item())
            orig_noise_std_min = float(orig_noise_std.min().item())
            orig_noise_std_max = float(orig_noise_std.max().item())
    else:
        flat_weighted_released = flat_weighted_clipped
        flat_released = flat_clipped
        emb_released = emb_clipped

    emb_released = torch.nan_to_num(emb_released, nan=0.0, posinf=0.0, neginf=0.0)
    flat_released = emb_released.view(batch_size, -1)
    flat_weighted_released = flat_released * sqrt_weights

    released_sample_norms = torch.norm(flat_weighted_released, p=2, dim=1)
    released_hit = float((released_sample_norms > prev_clip).float().mean().item())
    released_sq_norms = torch.sum(flat_weighted_released.pow(2), dim=1)
    debiased_sample_norms = released_sample_norms
    if epsilon > 1e-6:
        noise_energy = float(feature_dim) * (weighted_noise_std ** 2)
        debiased_sq_norms = torch.clamp(released_sq_norms - noise_energy, min=0.0)
        debiased_sample_norms = torch.sqrt(debiased_sq_norms)
    debiased_hit = float((debiased_sample_norms > prev_clip).float().mean().item())

    # Match the original SAC-era update exactly: C_{t+1} is driven by the
    # clipping hit rate observed on the released DP representation.  The
    # current release still uses C_t, so this remains delayed DP post-processing.
    control_hit = released_hit
    safe_norm = prev_clip

    if norm_tracker is not None:
        min_clip = max(clip_floor, 1e-8)
        max_clip = max(float(clip_growth_limit), clip_min_ratio) * clip_reference
        target = float(np.clip(clip_target, 0.0, 1.0))
        gain = max(float(clip_lr), 0.0)

        proposed_clip = prev_clip
        if control_hit > target + clip_tolerance:
            proposed_clip = prev_clip * (1.0 + gain * (control_hit - target))
        elif allow_clip_decay and control_hit < target - clip_tolerance:
            proposed_clip = prev_clip * (1.0 - gain * (target - control_hit))

        proposed_clip = min(max(proposed_clip, min_clip), max_clip)
        safe_norm = beta * prev_clip + (1.0 - beta) * proposed_clip
        safe_norm = min(max(safe_norm, min_clip), max_clip)
        norm_tracker[passive_id] = safe_norm

    if not return_stats:
        if not return_public_stats:
            return emb_released, None
        # These diagnostics are all derived from the released DP embedding,
        # the public clipping configuration, or known noise parameters.
        public_stats = {
            "sigma": current_sigma,
            "weighted_noise_std": weighted_noise_std,
            "clip": prev_clip,
            "clip_reference": clip_reference,
            "released_hit": released_hit,
            "debiased_hit": debiased_hit,
            "control_hit": control_hit,
            "next_clip_target": safe_norm,
            "clip_target": float(clip_target),
        }
        return emb_released, public_stats

    clipped_sample_norms = torch.norm(flat_weighted_clipped, p=2, dim=1)
    raw_released_sample_norms = torch.norm(flat_released, p=2, dim=1)
    pre_noise_hit = float((weighted_sample_norms > prev_clip).float().mean().item())
    clipped_norm_quantiles = torch.quantile(clipped_sample_norms.detach(), quantiles)
    released_mean_norm = released_sample_norms.mean().item()
    debiased_norm = float(debiased_sample_norms.mean().item())

    stats = {
        "sigma": current_sigma,
        "noise_std": orig_noise_std_mean,
        "noise_std_min": orig_noise_std_min,
        "noise_std_max": orig_noise_std_max,
        "weighted_noise_std": weighted_noise_std,
        "weight_min": float(weights.min().item()),
        "weight_max": float(weights.max().item()),
        "band_start": float(band_start),
        "band_end": float(band_end),
        "band_damping": float(band_damping),
        "tail_start": float(tail_start),
        "tail_damping": float(tail_damping),
        "clip": prev_clip,
        "clip_reference": clip_reference,
        "pre_noise_hit": pre_noise_hit,
        "released_hit": released_hit,
        "debiased_hit": debiased_hit,
        "control_hit": control_hit,
        "clip_hit": control_hit,
        "norm": raw_sample_norms.mean().item(),
        "norm_p50": float(raw_norm_quantiles[0].item()),
        "norm_p90": float(raw_norm_quantiles[1].item()),
        "norm_p95": float(raw_norm_quantiles[2].item()),
        "norm_max": float(raw_sample_norms.max().item()),
        "weighted_norm": weighted_sample_norms.mean().item(),
        "weighted_norm_p50": float(weighted_norm_quantiles[0].item()),
        "weighted_norm_p90": float(weighted_norm_quantiles[1].item()),
        "weighted_norm_p95": float(weighted_norm_quantiles[2].item()),
        "weighted_norm_max": float(weighted_sample_norms.max().item()),
        "clip_coef": clip_coef.mean().item(),
        "clipped_norm": clipped_sample_norms.mean().item(),
        "weighted_clipped_norm_p50": float(clipped_norm_quantiles[0].item()),
        "weighted_clipped_norm_p90": float(clipped_norm_quantiles[1].item()),
        "weighted_clipped_norm_p95": float(clipped_norm_quantiles[2].item()),
        "weighted_clipped_norm_max": float(clipped_sample_norms.max().item()),
        "released_norm": released_mean_norm,
        "debiased_norm": debiased_norm,
        "next_clip_target": safe_norm,
        "final_norm": released_mean_norm,
        "raw_clipped_norm": raw_clipped_sample_norms.mean().item(),
        "raw_released_norm": raw_released_sample_norms.mean().item(),
        "expected_noise_l2": math.sqrt(float(feature_dim)) * weighted_noise_std,
        "snr": clipped_sample_norms.mean().item() / (math.sqrt(float(feature_dim)) * weighted_noise_std + 1e-12),
        "clip_target": float(clip_target),
        "released_emb": emb_released.detach().clone(),
    }
    return emb_released, stats

def dp_forward_adaptive(emb, passive_id, epsilon, lambda_f, alpha_f, adaptive_clipping_state, history_grads):
    """
    前向自适应加噪 (Ada-VFed 风格的软稀疏门)
    结合了 RL 输出的 alpha_f (范数裁剪) 和 lambda_f (重要性维度噪声分配)
    """
    device = emb.device
    b_sz, dim = emb.shape

    # ==========================================
    # 1. 自适应阈值截断 (Adaptive L2 Clipping) -> 受 RL 动作 alpha_f 控制
    # ==========================================
    current_batch_norm = torch.mean(torch.norm(emb, p=2, dim=1)).item()
    
    beta = 0.9
    if passive_id not in adaptive_clipping_state:
        adaptive_clipping_state[passive_id] = current_batch_norm
    else:
        adaptive_clipping_state[passive_id] = beta * adaptive_clipping_state[passive_id] + (1 - beta) * current_batch_norm
        
    ema_norm = adaptive_clipping_state[passive_id]
    
    # 物理裁剪阈值 C
    C = max(ema_norm * alpha_f, 1e-4) 
    
    # 执行样本级 L2 裁剪
    sample_norms = torch.norm(emb, p=2, dim=1, keepdim=True)
    clip_coef = torch.clamp(C / (sample_norms + 1e-9), max=1.0)
    emb_clipped = emb * clip_coef

    # ==========================================
    # 2. 特征级异构加噪 (Soft Sparse Gate) -> 受 RL 动作 lambda_f 控制
    # ==========================================
    if epsilon > 1e-6:
        # 计算特征重要性 (依赖全局反向传播的真实历史梯度)
        if passive_id in history_grads and history_grads[passive_id] is not None:
            # 对一个 batch 的梯度求平均绝对值，作为该维度的重要性
            hist_grad = history_grads[passive_id]
            importance = torch.mean(torch.abs(hist_grad), dim=0)
        else:
            # 初始阶段(第0步)没有梯度时，默认所有维度同等重要
            importance = torch.ones(dim, device=device)
        
        # 使用 softmax 结合 lambda_f (温度/聚焦因子) 计算预算分配权重
        weights = torch.softmax(importance * lambda_f, dim=0)
        
        # 分配单维度的 epsilon (重要维度分得多，非重要维度分得少)
        eps_d = weights * dim * epsilon
        
        # 计算各维度独立的噪声标准差
        delta = 1e-5
        gaussian_factor = np.sqrt(2 * np.log(1.25 / delta))
        sigma_d = (C * gaussian_factor) / (eps_d + 1e-9)
        
        # 注入异构噪声 (广播机制为每个维度加不同方差的噪声)
        noise = torch.randn_like(emb_clipped, device=device) * sigma_d.unsqueeze(0)
        emb_noisy = emb_clipped + noise
    else:
        emb_noisy = emb_clipped
        
    return emb_noisy
    



class DopplerFilter:
    def __init__(self, shape, device, a_coeffs=None, b_coeffs=None):
        """
        Legacy low-pass filter helper retained for backward compatibility.
        """
        self.device = device
        
        # 解析系数
        self.a_coeffs = [float(x) for x in (a_coeffs if a_coeffs is not None else [-0.81818181])]
        self.b_coeffs = [float(x) for x in (b_coeffs if b_coeffs is not None else [0.09090909, 0.09090909])]

        self.n_a = len(self.a_coeffs)
        self.n_b = len(self.b_coeffs)

        # Buffer for m (output history) and g (input history)
        self.m_buffer = [torch.zeros(shape).to(device) for _ in range(self.n_a)]
        self.g_buffer = [torch.zeros(shape).to(device) for _ in range(self.n_b)]

        # Buffer for Bias Correction factors
        self.c_a_buffer = [0.0] * self.n_a
        self.c_b_buffer = [0.0] * self.n_b
        self.c_a_curr = 0.0

    def step(self, current_grad):
        """
        Apply filter to the current noisy gradient.
        Args:
            current_grad: The privatized gradient at step t (g_t)
        Returns:
            Filtered gradient (m_hat_t)
        """
        # === 修复 Shape Mismatch 的关键逻辑 ===
        # 检查 Buffer 中的历史梯度形状是否与当前梯度一致
        # 如果不一致（例如遇到最后一个 Batch），则重置滤波器状态
        if len(self.g_buffer) > 0 and self.g_buffer[0].shape != current_grad.shape:
            # 1. 重置历史梯度和输出 Buffer 为新形状的零张量
            self.m_buffer = [torch.zeros_like(current_grad) for _ in range(self.n_a)]
            self.g_buffer = [torch.zeros_like(current_grad) for _ in range(self.n_b)]
            
            # 2. 重置偏差修正系数 (相当于重启滤波器)
            self.c_a_buffer = [0.0] * self.n_a
            self.c_b_buffer = [0.0] * self.n_b
            self.c_a_curr = 0.0
            
            # 可选：打印调试信息，但为了不刷屏通常省略
            # print(f"DopplerFilter: Shape mismatch detected (New: {current_grad.shape}). Resetting filter.")

        # === 以下逻辑保持不变 ===

        # 1. Update g_buffer (Insert g_t at front)
        self.g_buffer.insert(0, current_grad)
        if len(self.g_buffer) > self.n_b:
            self.g_buffer.pop()

        # 2. Compute m_t (Paper Alg 2 Line 6)
        m_t = torch.zeros_like(current_grad)
        
        # Input part (b terms)
        for tau, b_val in enumerate(self.b_coeffs):
            if tau < len(self.g_buffer):
                m_t += b_val * self.g_buffer[tau]
        
        # Recursive part (a terms)
        for tau, a_val in enumerate(self.a_coeffs):
            if tau < len(self.m_buffer):
                m_t -= a_val * self.m_buffer[tau]

        # Update m_buffer
        self.m_buffer.insert(0, m_t)
        if len(self.m_buffer) > self.n_a:
            self.m_buffer.pop()

        # 3. Compute Bias Correction (Paper Alg 2 Line 7)
        self.c_b_buffer.insert(0, 1.0)
        if len(self.c_b_buffer) > self.n_b:
            self.c_b_buffer.pop()
        
        c_a_t = 0.0
        for tau, b_val in enumerate(self.b_coeffs):
             if tau < len(self.c_b_buffer):
                 c_a_t += b_val * self.c_b_buffer[tau]
        for tau, a_val in enumerate(self.a_coeffs):
            if tau < len(self.c_a_buffer):
                c_a_t -= a_val * self.c_a_buffer[tau]
        
        self.c_a_curr = c_a_t
        
        self.c_a_buffer.insert(0, c_a_t)
        if len(self.c_a_buffer) > self.n_a:
            self.c_a_buffer.pop()

        # 4. Normalize (Paper Alg 2 Line 8)
        if abs(self.c_a_curr) > 1e-9:
            m_hat = m_t / self.c_a_curr
        else:
            m_hat = m_t 
            
        return m_hat

    
