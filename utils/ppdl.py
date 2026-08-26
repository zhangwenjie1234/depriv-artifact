import torch
import numpy as np

def ppdl_defense(grad, args, step_eps=None, device='cpu'):
    """
    PPDL Defense Mechanism - ACVFL Aligned
    """
    # 1. 强制使用传入的 step_eps (对齐 ACVFL)
    if step_eps is None:
        raise ValueError("PPDL must receive step_eps from the dynamic allocator!")

    # 2. 核心处理逻辑
    if isinstance(grad, (tuple, list)):
        sanitized_list = []
        for i, g in enumerate(grad):
            if g is not None:
                sanitized_list.append(_apply_ppdl_core(g, args, step_eps, device, party_id=i))
            else:
                sanitized_list.append(None)
        return tuple(sanitized_list), step_eps
    else:
        return (_apply_ppdl_core(grad, args, step_eps, device),), step_eps

def _apply_ppdl_core(grad_tensor, args, step_eps, device, party_id=None):
    grad_tensor = grad_tensor.to(device)
    original_shape = grad_tensor.shape
    flat_grad = grad_tensor.flatten()
    total_num = flat_grad.numel()
    
    orig_tau = getattr(args, 'ppdl_tau', 0.001)
    theta_u = getattr(args, 'ppdl_theta_u', 1.0)
    orig_noise_std = getattr(args, 'ppdl_noise_std', 0.01)
    
    clip_C = getattr(args, 'clip_threshold', 0.03)
    delta = getattr(args, 'delta_c', 1e-5)

    # 3. L2 范数裁剪
    grad_norm = torch.norm(flat_grad, p=2)
    clip_coef = min(1.0, clip_C / (grad_norm.item() + 1e-9))
    flat_grad_clipped = flat_grad * clip_coef

    # 4. 动态计算出真正合规的 DP 噪声 (使用对齐 ACVFL 的 step_eps)
    gaussian_factor = np.sqrt(2 * np.log(1.25 / delta))
    sensitivity_scale = 2.0
    dynamic_noise_std = (clip_C * gaussian_factor * sensitivity_scale) / step_eps

    # 5. 动态自适应修正 tau
    dynamic_tau = orig_tau * (dynamic_noise_std / orig_noise_std)

    # 6. 生成并注入高斯噪声
    noise = torch.randn_like(flat_grad_clipped) * dynamic_noise_std
    noisy_vals = flat_grad_clipped + noise

    # 7. 稀疏化筛选
    mask = noisy_vals.abs() > dynamic_tau
    valid_count = mask.sum().item()

    desired_count = int(np.ceil(theta_u * total_num))
    new_flat_grad = torch.zeros_like(flat_grad_clipped)

    if valid_count > desired_count:
        valid_indices = torch.nonzero(mask, as_tuple=False).view(-1)
        perm = torch.randperm(valid_indices.numel(), device=device)
        selected_indices = valid_indices[perm[:desired_count]]
        new_flat_grad[selected_indices] = noisy_vals[selected_indices]
        action_msg = f"Downsampled from {valid_count} to {desired_count}"
    else:
        new_flat_grad[mask] = noisy_vals[mask]
        action_msg = f"Kept all {valid_count} gradients (Under limit)"

    # 调试打印
    if np.random.rand() < 0.005: 
        print(f"\n[PPDL Debug] Party {party_id if party_id is not None else '?'}")
        print(f"  > Target Eps: {args.epsilon}, step_eps: {step_eps:.6f}")
        print(f"  > Adapted Params: dynamic_tau={dynamic_tau:.6f}, dynamic_noise={dynamic_noise_std:.6f}")
        print(f"  > Action: {action_msg}")

    return new_flat_grad.view(original_shape)
