import torch

from utils.malicious_optimizer import MaliciousSGD


def resolve_weight_decay(args):
    default_wd = 5e-4 if getattr(args, "dataset", None) == "cifar10" else 0.0
    wd_active = float(getattr(args, "weight_decay_active", -1.0))
    wd_passive = float(getattr(args, "weight_decay_passive", -1.0))
    if wd_active < 0.0:
        wd_active = default_wd
    if wd_passive < 0.0:
        wd_passive = default_wd
    return wd_active, wd_passive


def resolve_sgd_momentum(args):
    default_momentum = 0.9 if getattr(args, "dataset", None) in ["mnist", "fashionmnist", "cifar10", "criteo"] else 0.0
    configured_momentum = float(getattr(args, "sgd_momentum", -1.0))
    return default_momentum if configured_momentum < 0.0 else configured_momentum


def build_main_task_optimizers(args, model):
    wd_active, wd_passive = resolve_weight_decay(args)
    momentum = resolve_sgd_momentum(args)

    optimizer_entire = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_active,
        weight_decay=wd_active,
        momentum=momentum,
    )
    optimizer_active = torch.optim.SGD(
        model.active.parameters(),
        lr=args.lr_active,
        weight_decay=wd_active,
        momentum=momentum,
    )

    optimizer_passive = []
    for i in range(args.num_passive):
        lr = args.lr_attack if i == args.attack_id else args.lr_passive
        if (
            args.attack == "amc"
            and i == args.attack_id
            and not bool(getattr(args, "amc_disable_malicious_optimizer", False))
        ):
            optimizer_passive.append(
                MaliciousSGD(
                    model.passive[i].parameters(),
                    lr=lr,
                    momentum=float(getattr(args, "amc_momentum", 0.9)),
                    gamma=float(getattr(args, "amc_gamma", 1.0)),
                    rmax=float(getattr(args, "amc_rmax", 5.0)),
                    rmin=float(getattr(args, "amc_rmin", 1.0)),
                    weight_decay=wd_passive,
                )
            )
        else:
            optimizer_passive.append(
                torch.optim.SGD(
                    model.passive[i].parameters(),
                    lr=lr,
                    weight_decay=wd_passive,
                    momentum=momentum,
                )
            )
    return optimizer_entire, optimizer_active, optimizer_passive


def resolve_main_lr_scheduler_name(args):
    requested = str(getattr(args, "main_lr_scheduler", "auto")).lower()
    if requested != "auto":
        return requested
    if getattr(args, "dataset", None) in ["cifar10", "cifar100"]:
        return "cosine"
    return "none"


def build_main_task_schedulers(args, optimizer_entire, optimizer_active, optimizer_passive):
    scheduler_name = resolve_main_lr_scheduler_name(args)
    if scheduler_name == "none":
        return []

    all_optimizers = [optimizer_entire, optimizer_active] + list(optimizer_passive)
    total_epochs = max(1, int(getattr(args, "epochs", 1)))

    if scheduler_name == "cosine":
        min_factor = float(getattr(args, "main_lr_min_factor", 0.05))
        schedulers = []
        for optimizer in all_optimizers:
            base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
            eta_min = min(base_lrs) * min_factor if base_lrs else 0.0
            schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_epochs,
                    eta_min=eta_min,
                )
            )
        return schedulers

    if scheduler_name == "multistep":
        milestones = sorted({int(m) for m in getattr(args, "main_lr_milestones", [6, 8]) if int(m) > 0})
        gamma = float(getattr(args, "main_lr_decay_gamma", 0.1))
        return [
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=gamma,
            )
            for optimizer in all_optimizers
        ]

    raise ValueError("Unsupported main LR scheduler: {}".format(scheduler_name))


def step_main_task_schedulers(schedulers):
    for scheduler in schedulers:
        scheduler.step()


def summarize_main_task_lrs(args, optimizer_active, optimizer_passive):
    active_lr = float(optimizer_active.param_groups[0]["lr"])
    passive_lrs = [float(opt.param_groups[0]["lr"]) for opt in optimizer_passive]
    attack_id = int(getattr(args, "attack_id", 0))
    attack_lr = passive_lrs[attack_id] if 0 <= attack_id < len(passive_lrs) else 0.0
    passive_mean_lr = sum(passive_lrs) / max(1, len(passive_lrs))
    return "active={:.6f}, passive_mean={:.6f}, passive_attack={:.6f}".format(
        active_lr,
        passive_mean_lr,
        attack_lr,
    )
