import inspect

import numpy as np
import torch

import utils.dp as dp


class TaskPrior:
    """
    Auxiliary-data-calibrated task-aware importance prior.

    The prior is estimated once on the auxiliary held-out validation split
    using a first-order Taylor criterion |z * dL/dz| and then fixed during
    private training.
    """

    def __init__(self, args, vflbase_instance):
        self.args = args
        self.vflbase = vflbase_instance
        self.global_min_eps_ratio = 0.50
        self.global_max_eps_ratio = 2.00

        self.importance_accumulators = {}
        self.importance_sq_accumulators = {}
        self.warmup_counts = {}
        self.static_importance = {}
        self.party_importance_scores = {}
        self.forward_clip_recommendations = {}
        self.forward_clip_stats = {}
        self.party_weight_stats = {}
        self.party_weight_configs = {}
        self.logged_role_weight_configs = set()
        self.last_calibration_summary = {}
        self._forward_weight_cache = {}
        self._backward_mask_cache = {}

    @staticmethod
    def _normalize_vector(vec):
        v_min, v_max = vec.min(), vec.max()
        if float((v_max - v_min).item()) < 1e-8:
            return torch.ones_like(vec) * 0.5
        return (vec - v_min) / (v_max - v_min)

    @staticmethod
    def _rank_normalize(vec):
        if vec.numel() <= 1:
            return torch.ones_like(vec) * 0.5
        order = torch.argsort(vec)
        ranks = torch.empty_like(vec)
        ranks[order] = torch.arange(vec.numel(), device=vec.device, dtype=vec.dtype)
        return ranks / float(max(vec.numel() - 1, 1))

    def _accumulate_importance(self, party_id, output, grad):
        flat_output = output.detach().reshape(output.shape[0], -1)
        flat_grad = grad.detach().reshape(grad.shape[0], -1)
        device = flat_output.device
        feature_dim = flat_output.shape[1]

        if party_id not in self.importance_accumulators:
            self.importance_accumulators[party_id] = torch.zeros(feature_dim, device=device)
            self.importance_sq_accumulators[party_id] = torch.zeros(feature_dim, device=device)
            self.warmup_counts[party_id] = 0

        step_importance = torch.mean(torch.abs(flat_output * flat_grad), dim=0)
        self.importance_accumulators[party_id] += step_importance
        self.importance_sq_accumulators[party_id] += step_importance.pow(2)
        self.warmup_counts[party_id] += 1

    def _resolve_aux_supervision_labels(self, pred, labels):
        if bool(getattr(self.args, "aux_public_label_compatible", False)):
            return labels
        # For cross-dataset auxiliary data (e.g. EMNIST Letters / CIFAR-100),
        # we use the target model's current pseudo labels so the public prior
        # remains task-shaped without assuming aligned class semantics.
        return pred.detach().argmax(dim=1)

    def _finalize_static_importance(self, party_id):
        if party_id in self.static_importance:
            return
        if party_id not in self.importance_accumulators:
            return

        count = max(1, self.warmup_counts[party_id])
        avg_importance = self.importance_accumulators[party_id] / count
        avg_sq_importance = self.importance_sq_accumulators[party_id] / count
        var_importance = torch.clamp(avg_sq_importance - avg_importance.pow(2), min=0.0)
        std_importance = torch.sqrt(var_importance + 1e-12)

        # Public-data consistency acts as a confidence score so a few spiky
        # coordinates do not dominate the weighted perturbation map.
        stability = avg_importance / (std_importance + avg_importance.mean() * 0.1 + 1e-8)
        mean_rank = self._rank_normalize(avg_importance)
        stability_rank = self._rank_normalize(stability)
        mean_norm = self._normalize_vector(avg_importance)
        stability_norm = self._normalize_vector(stability)

        combined_prior = (
            0.45 * mean_norm
            + 0.20 * stability_norm
            + 0.20 * mean_rank
            + 0.15 * stability_rank
        )
        norm_importance = self._normalize_vector(combined_prior)
        self.party_importance_scores[party_id] = float(norm_importance.mean().item())

        self.static_importance[party_id] = norm_importance

    def _collect_public_weight_stats(self, aux_dataset, model, device, total_batches):
        stats = {
            party_id: {
                "raw_norms": [],
                "weighted_norms": [],
            }
            for party_id in range(self.vflbase.args.num_passive)
        }

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(aux_dataset):
                if batch_idx >= total_batches:
                    break

                data, _ = batch_data
                data = [d.to(device) for d in data]
                emb, _, _ = model(data)

                for party_id, output in enumerate(emb):
                    flat_output = output.detach().reshape(output.shape[0], -1)
                    raw_norms = torch.norm(flat_output, p=2, dim=1)
                    importance_prior = self.static_importance.get(party_id)
                    weights = dp.build_importance_weights(
                        importance_prior=(
                            importance_prior.to(flat_output.device)
                            if importance_prior is not None else None
                        ),
                        feature_dim=flat_output.shape[1],
                        device=flat_output.device,
                        dtype=flat_output.dtype,
                        min_eps_ratio=self.global_min_eps_ratio,
                        max_eps_ratio=self.global_max_eps_ratio,
                    )
                    weighted_norms = torch.norm(
                        flat_output * torch.sqrt(weights).unsqueeze(0),
                        p=2,
                        dim=1,
                    )
                    stats[party_id]["raw_norms"].append(raw_norms.cpu())
                    stats[party_id]["weighted_norms"].append(weighted_norms.cpu())

        return stats

    def _calibrate_weight_profiles(self, aux_dataset, model, device, total_batches):
        public_stats = self._collect_public_weight_stats(aux_dataset, model, device, total_batches)

        for party_id in range(self.vflbase.args.num_passive):
            party_stats = public_stats[party_id]
            if len(party_stats["raw_norms"]) == 0 or len(party_stats["weighted_norms"]) == 0:
                self.party_weight_stats[party_id] = {
                    "amplification": 1.0,
                    "raw_p90": 0.0,
                    "weighted_p90": 0.0,
                    "high_conf_mass": 0.0,
                }
                continue

            raw_norms = torch.cat(party_stats["raw_norms"], dim=0).to(dtype=torch.float32)
            weighted_norms = torch.cat(party_stats["weighted_norms"], dim=0).to(dtype=torch.float32)
            q90 = torch.tensor(0.90, dtype=raw_norms.dtype)
            raw_p90 = float(torch.quantile(raw_norms, q90).item())
            weighted_p90 = float(torch.quantile(weighted_norms, q90).item())
            amplification = weighted_p90 / max(raw_p90, 1e-8)
            importance_prior = self.static_importance.get(party_id)
            high_conf_mass = 0.0
            if importance_prior is not None:
                high_conf_mass = float((importance_prior >= 0.85).float().mean().item())

            self.party_weight_stats[party_id] = {
                "amplification": float(amplification),
                "raw_p90": float(raw_p90),
                "weighted_p90": float(weighted_p90),
                "high_conf_mass": float(high_conf_mass),
            }

    def _build_party_weight_config(self, party_id):
        weight_stats = self.party_weight_stats.get(party_id, {})
        amplification = float(weight_stats.get("amplification", 1.0))
        high_conf_mass = float(weight_stats.get("high_conf_mass", 0.0))

        config = {
            "importance_strength": 1.9,
            "min_eps_ratio": self.global_min_eps_ratio,
            "max_eps_ratio": self.global_max_eps_ratio,
            "band_start": 1.0,
            "band_end": 1.0,
            "band_damping": 0.0,
            "band_power": 1.0,
            "tail_start": 1.0,
            "tail_damping": 0.0,
            "tail_power": 2.0,
            "role": "party",
            "amplification": amplification,
            "high_conf_mass": high_conf_mass,
        }

        if party_id not in self.logged_role_weight_configs:
            if not getattr(self.args, "rl", False):
                print(
                    "    [Task Prior] P{} NeutralWeight Amp={:.3f} HighConfMass={:.1f}% WeightRange=[{:.2f},{:.2f}]".format(
                        party_id,
                        amplification,
                        100.0 * high_conf_mass,
                        self.global_min_eps_ratio,
                        self.global_max_eps_ratio,
                    )
                )
            self.logged_role_weight_configs.add(party_id)

        return config

    def _calibrate_forward_clips(self, aux_dataset, model, device, total_batches):
        base_clip = float(getattr(self.args, "clip_threshold_forward", 1.0))
        target_hit = float(np.clip(getattr(self.args, "forward_public_clip_target_hit", 0.40), 1e-4, 1.0 - 1e-4))
        target_quantile = 1.0 - target_hit
        scale_min = float(max(getattr(self.args, "forward_public_clip_scale_min", 0.90), 1e-4))
        scale_max = float(max(getattr(self.args, "forward_public_clip_scale_max", 1.25), scale_min))
        oversize_hit_floor = 0.02
        oversize_quantile_gap = 0.75
        min_public_clip = 1e-4

        raw_norm_buffers = {party_id: [] for party_id in range(self.vflbase.args.num_passive)}
        weighted_norm_buffers = {party_id: [] for party_id in range(self.vflbase.args.num_passive)}

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(aux_dataset):
                if batch_idx >= total_batches:
                    break

                data, _ = batch_data
                data = [d.to(device) for d in data]
                emb, _, _ = model(data)

                for party_id, output in enumerate(emb):
                    flat_output = output.detach().reshape(output.shape[0], -1)
                    raw_norms = torch.norm(flat_output, p=2, dim=1)
                    importance_prior = self.static_importance.get(party_id)
                    weights = dp.build_importance_weights(
                        importance_prior=(
                            importance_prior.to(flat_output.device)
                            if importance_prior is not None else None
                        ),
                        feature_dim=flat_output.shape[1],
                        device=flat_output.device,
                        dtype=flat_output.dtype,
                        min_eps_ratio=self.global_min_eps_ratio,
                        max_eps_ratio=self.global_max_eps_ratio,
                    )
                    weighted_norms = torch.norm(flat_output * torch.sqrt(weights).unsqueeze(0), p=2, dim=1)
                    raw_norm_buffers[party_id].append(raw_norms.cpu())
                    weighted_norm_buffers[party_id].append(weighted_norms.cpu())

        for party_id in range(self.vflbase.args.num_passive):
            if len(raw_norm_buffers[party_id]) == 0 or len(weighted_norm_buffers[party_id]) == 0:
                self.forward_clip_recommendations[party_id] = base_clip
                continue

            raw_norms = torch.cat(raw_norm_buffers[party_id], dim=0).to(dtype=torch.float32)
            weighted_norms = torch.cat(weighted_norm_buffers[party_id], dim=0).to(dtype=torch.float32)
            q_tensor = torch.tensor(target_quantile, dtype=raw_norms.dtype)
            weighted_clip = float(torch.quantile(weighted_norms, q_tensor).item())
            raw_hit = float((raw_norms > base_clip).float().mean().item())
            weighted_hit_at_base = float((weighted_norms > base_clip).float().mean().item())
            raw_p50 = float(torch.quantile(raw_norms, torch.tensor(0.50, dtype=raw_norms.dtype)).item())
            raw_p90 = float(torch.quantile(raw_norms, torch.tensor(0.90, dtype=raw_norms.dtype)).item())
            raw_p95 = float(torch.quantile(raw_norms, torch.tensor(0.95, dtype=raw_norms.dtype)).item())
            raw_max = float(raw_norms.max().item())
            weighted_p50 = float(torch.quantile(weighted_norms, torch.tensor(0.50, dtype=weighted_norms.dtype)).item())
            weighted_p90 = float(torch.quantile(weighted_norms, torch.tensor(0.90, dtype=weighted_norms.dtype)).item())
            weighted_p95 = float(torch.quantile(weighted_norms, torch.tensor(0.95, dtype=weighted_norms.dtype)).item())
            weighted_max = float(weighted_norms.max().item())

            default_clip = min(max(weighted_clip, base_clip * scale_min), base_clip * scale_max)
            oversize_detected = (
                raw_hit < oversize_hit_floor
                and weighted_hit_at_base < oversize_hit_floor
                and weighted_p95 < base_clip * oversize_quantile_gap
            )
            if oversize_detected:
                # If the nominal base clip is far above the public embedding norms,
                # keeping it unchanged disables clipping almost entirely. In that case
                # we still drop the oversized base constant, but we warm-start from
                # a more conservative upper-tail estimate instead of the exact target
                # quantile. This reduces early over-clipping bias while leaving the
                # later online tracker and RL policy unchanged.
                calibrated_clip = max(weighted_p95, min_public_clip)
            else:
                calibrated_clip = default_clip
            self.forward_clip_recommendations[party_id] = calibrated_clip
            self.forward_clip_stats[party_id] = {
                "raw_hit_at_base": raw_hit,
                "weighted_hit_at_base": weighted_hit_at_base,
                "recommended_clip": calibrated_clip,
                "weighted_quantile_clip": weighted_clip,
                "raw_p50": raw_p50,
                "raw_p90": raw_p90,
                "raw_p95": raw_p95,
                "raw_max": raw_max,
                "weighted_p50": weighted_p50,
                "weighted_p90": weighted_p90,
                "weighted_p95": weighted_p95,
                "weighted_max": weighted_max,
                "oversize_detected": float(oversize_detected),
            }

    def calibrate_from_aux_data(self, aux_dataset, model, device, max_batches=None):
        if not getattr(self.args, "rl", False):
            return
        if aux_dataset is None or len(aux_dataset) == 0:
            return

        self.importance_accumulators = {}
        self.importance_sq_accumulators = {}
        self.warmup_counts = {}
        self.static_importance = {}
        self.party_importance_scores = {}
        self.forward_clip_recommendations = {}
        self.forward_clip_stats = {}
        self.party_weight_stats = {}
        self.party_weight_configs = {}
        self.logged_role_weight_configs = set()
        self.last_calibration_summary = {}
        self._forward_weight_cache = {}
        self._backward_mask_cache = {}
        self.vflbase.forward_norm_tracker = {}

        was_training_model = model.training
        was_training_active = model.active.training
        passive_training_flags = [module.training for module in model.passive]

        model.eval()
        model.active.eval()
        for module in model.passive:
            module.eval()

        total_batches = len(aux_dataset) if max_batches is None else min(len(aux_dataset), max_batches)
        for batch_idx, batch_data in enumerate(aux_dataset):
            if batch_idx >= total_batches:
                break

            data, labels = batch_data
            data = [d.to(device) for d in data]
            labels = labels.to(device)

            model.zero_grad(set_to_none=True)
            emb, _, pred = model(data)
            supervision_labels = self._resolve_aux_supervision_labels(pred, labels)
            loss = self.vflbase.loss(pred, supervision_labels)
            emb_grads = torch.autograd.grad(
                loss,
                emb,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )

            for party_id, (output, grad) in enumerate(zip(emb, emb_grads)):
                self._accumulate_importance(party_id, output, grad)

        for party_id in range(self.vflbase.args.num_passive):
            self._finalize_static_importance(party_id)
            self.party_weight_configs[party_id] = self._build_party_weight_config(party_id)

        if bool(getattr(self.args, "forward_public_clip_calibration", 1)):
            self._calibrate_forward_clips(aux_dataset, model, device, total_batches)

        importance_means = []
        for party_id in range(self.vflbase.args.num_passive):
            importance_prior = self.static_importance.get(party_id)
            if importance_prior is not None:
                importance_means.append(float(importance_prior.mean().item()))
        clip_values = [
            float(self.forward_clip_recommendations[party_id])
            for party_id in range(self.vflbase.args.num_passive)
            if party_id in self.forward_clip_recommendations
        ]
        self.last_calibration_summary = {
            "parties": int(self.vflbase.args.num_passive),
            "importance_mean": float(np.mean(importance_means)) if len(importance_means) > 0 else 0.0,
            "clip_min": float(np.min(clip_values)) if len(clip_values) > 0 else 0.0,
            "clip_max": float(np.max(clip_values)) if len(clip_values) > 0 else 0.0,
        }
        if bool(getattr(self.args, "forward_weighted_clipping", 1)) and not getattr(self.args, "rl", False):
            print(
                "  [Task Prior] Ready: parties={} weight_range=[{:.2f}, {:.2f}] clip_range=[{:.2f}, {:.2f}] mean_importance={:.4f}".format(
                    self.last_calibration_summary["parties"],
                    self.global_min_eps_ratio,
                    self.global_max_eps_ratio,
                    self.last_calibration_summary["clip_min"],
                    self.last_calibration_summary["clip_max"],
                    self.last_calibration_summary["importance_mean"],
                )
            )
            for party_id in range(self.vflbase.args.num_passive):
                clip_stats = self.forward_clip_stats.get(party_id, {})
                if len(clip_stats) == 0:
                    continue
                print(
                    "    [Task Prior] P{} PublicNorm Raw[p50={:.2f},p90={:.2f},p95={:.2f},max={:.2f}] "
                    "Weighted[p50={:.2f},p90={:.2f},p95={:.2f},max={:.2f}] "
                    "BaseHit={:.1f}%/WBaseHit={:.1f}%/RecClip={:.2f}".format(
                        party_id,
                        clip_stats.get("raw_p50", 0.0),
                        clip_stats.get("raw_p90", 0.0),
                        clip_stats.get("raw_p95", 0.0),
                        clip_stats.get("raw_max", 0.0),
                        clip_stats.get("weighted_p50", 0.0),
                        clip_stats.get("weighted_p90", 0.0),
                        clip_stats.get("weighted_p95", 0.0),
                        clip_stats.get("weighted_max", 0.0),
                        100.0 * clip_stats.get("raw_hit_at_base", 0.0),
                        100.0 * clip_stats.get("weighted_hit_at_base", 0.0),
                        clip_stats.get("recommended_clip", 0.0),
                    )
                )
        elif not getattr(self.args, "rl", False):
            print(
                "  [Task Prior] Ready: parties={} mean_importance={:.4f}".format(
                    self.last_calibration_summary["parties"],
                    self.last_calibration_summary["importance_mean"],
                )
            )

        model.zero_grad(set_to_none=True)
        model.train(was_training_model)
        model.active.train(was_training_active)
        for module, was_training in zip(model.passive, passive_training_flags):
            module.train(was_training)

    def calibrate_from_public_data(self, public_dataset, model, device, max_batches=None):
        self.calibrate_from_aux_data(public_dataset, model, device, max_batches=max_batches)

    def get_party_importance_scores(self, num_parties):
        scores = []
        for party_id in range(num_parties):
            scores.append(float(self.party_importance_scores.get(party_id, 0.0)))
        return scores

    def get_backward_topk_mask(self, party_id, feature_dim, keep_rate, device):
        if feature_dim <= 0:
            return None

        safe_keep_rate = float(np.clip(keep_rate, 1e-6, 1.0))
        keep_count = min(feature_dim, max(1, int(np.ceil(feature_dim * safe_keep_rate))))

        importance_prior = self.static_importance.get(party_id)
        if importance_prior is None:
            return None

        cache_key = (int(party_id), int(feature_dim), int(keep_count), str(torch.device(device)))
        cached_mask = self._backward_mask_cache.get(cache_key)
        if cached_mask is not None:
            return cached_mask

        prior = importance_prior.to(device=device, dtype=torch.float32).view(-1)
        if prior.numel() != feature_dim:
            return None

        topk_idx = torch.topk(prior, k=keep_count, largest=True, sorted=False).indices
        mask = torch.zeros(feature_dim, device=device, dtype=torch.float32)
        mask[topk_idx] = 1.0
        self._backward_mask_cache[cache_key] = mask
        return mask

    def get_forward_clip_value(self, party_id, default_clip):
        return float(self.forward_clip_recommendations.get(party_id, default_clip))

    def get_weight_config(self, party_id):
        return self.party_weight_configs.get(party_id, {})

    def get_forward_importance_weights(self, party_id, feature_dim, device, dtype):
        cache_key = (int(party_id), int(feature_dim), str(torch.device(device)), dtype)
        cached_weights = self._forward_weight_cache.get(cache_key)
        if cached_weights is not None:
            return cached_weights

        use_weighted_clipping = bool(getattr(self.args, "forward_weighted_clipping", 1))
        importance_prior = self.static_importance.get(party_id) if use_weighted_clipping else None
        weight_config = self.get_weight_config(party_id) if use_weighted_clipping else {}
        weights = dp.build_importance_weights(
            importance_prior=importance_prior,
            feature_dim=feature_dim,
            device=device,
            dtype=dtype,
            importance_strength=weight_config.get("importance_strength", 2.0),
            min_eps_ratio=weight_config.get("min_eps_ratio", self.global_min_eps_ratio),
            max_eps_ratio=weight_config.get("max_eps_ratio", self.global_max_eps_ratio),
            band_start=weight_config.get("band_start", 1.0),
            band_end=weight_config.get("band_end", 1.0),
            band_damping=weight_config.get("band_damping", 0.0),
            band_power=weight_config.get("band_power", 1.0),
            tail_start=weight_config.get("tail_start", 1.0),
            tail_damping=weight_config.get("tail_damping", 0.0),
            tail_power=weight_config.get("tail_power", 2.0),
        ).detach()
        self._forward_weight_cache[cache_key] = weights
        return weights

    def get_hook(self, party_id):
        def hook(module, input, output):
            if getattr(self.args, "attack", "") == "amc":
                raw_embeddings = getattr(self.vflbase, "_last_raw_forward_embeddings", None)
                if raw_embeddings is not None and party_id < len(raw_embeddings):
                    raw_embeddings[party_id] = output.detach()

            if not getattr(self.args, "rl", False):
                return output

            forward_plan = getattr(self.vflbase, "current_forward_defense_plan", None)
            if forward_plan is None or party_id not in forward_plan:
                return output

            plan = forward_plan[party_id]
            if plan.get("enabled", False):
                use_weighted_clipping = bool(getattr(self.args, "forward_weighted_clipping", 1))
                importance_prior = self.static_importance.get(party_id) if use_weighted_clipping else None
                weight_config = self.get_weight_config(party_id) if use_weighted_clipping else {}
                initial_clip = float(plan.get("initial_clip", 1.0))
                base_clip = float(getattr(self.args, "clip_threshold_forward", 1.0))
                forward_eps = float(max(plan.get("epsilon", 0.0), 1e-8))
                gaussian_factor = float(np.sqrt(2.0 * np.log(1.25 / 1e-5)))
                sigma_proxy = gaussian_factor / forward_eps
                relax_ratio = float(1.0 / (1.0 + (sigma_proxy / 0.8) ** 2))
                clip_cap_scale = float(max(plan.get("clip_cap_scale", 1.0), 1.0))
                scaled_base_clip = max(base_clip * clip_cap_scale, initial_clip)
                max_clip = initial_clip + relax_ratio * max(scaled_base_clip - initial_clip, 0.0)
                growth_limit = max(max_clip / max(initial_clip, 1e-8), 1.0)
                base_target = float(
                    np.clip(
                        plan.get(
                            "clip_target",
                            getattr(self.args, "forward_public_clip_target_hit", 0.25),
                        ),
                        1e-4,
                        1.0 - 1e-4,
                    )
                )
                adaptive_target = float(base_target)
                freeze_public_clip = bool(getattr(self.args, "forward_freeze_public_clip", 0))
                feature_dim = int(output.reshape(output.shape[0], -1).shape[1])
                importance_weights = self.get_forward_importance_weights(
                    party_id=party_id,
                    feature_dim=feature_dim,
                    device=output.device,
                    dtype=output.dtype,
                )
                forward_dp_kwargs = dict(
                    passive_id=party_id,
                    epsilon=forward_eps,
                    device=output.device,
                    norm_tracker=None if freeze_public_clip else self.vflbase.forward_norm_tracker,
                    beta=0.995,
                    initial_clip=initial_clip,
                    clip_growth_limit=growth_limit,
                    clip_target=adaptive_target,
                    clip_tolerance=0.02,
                    clip_lr=0.50,
                    clip_min_ratio=1.0,
                    allow_clip_decay=False,
                    importance_strength=weight_config.get("importance_strength", 2.0),
                    min_eps_ratio=weight_config.get("min_eps_ratio", self.global_min_eps_ratio),
                    max_eps_ratio=weight_config.get("max_eps_ratio", self.global_max_eps_ratio),
                    band_start=weight_config.get("band_start", 1.0),
                    band_end=weight_config.get("band_end", 1.0),
                    band_damping=weight_config.get("band_damping", 0.0),
                    band_power=weight_config.get("band_power", 1.0),
                    tail_start=weight_config.get("tail_start", 1.0),
                    tail_damping=weight_config.get("tail_damping", 0.0),
                    tail_power=weight_config.get("tail_power", 2.0),
                    importance_weights=importance_weights,
                )
                supports_public_stats = "return_public_stats" in inspect.signature(
                    dp.dp_forward_perturb_adaptive
                ).parameters
                if supports_public_stats:
                    forward_dp_kwargs.update(
                        return_stats=False,
                        return_public_stats=True,
                    )
                else:
                    # Compatibility with backups that predate the dedicated
                    # DP-safe stats return.  Only released/public fields below
                    # are consumed; raw diagnostic fields never enter SAC.
                    forward_dp_kwargs["return_stats"] = True
                output, forward_stats = dp.dp_forward_perturb_adaptive(
                    output,
                    **forward_dp_kwargs,
                )
                if forward_stats is not None:
                    forward_stats.pop("released_emb", None)
                log_interval = int(getattr(self.args, "forward_clip_log_interval", 100))
                batch_idx = int(getattr(self.vflbase, "batch_idx", 0))
                should_log = log_interval > 0 and (
                    batch_idx == 0 or (batch_idx + 1) % log_interval == 0
                )
                if should_log:
                    clip_used = float(forward_stats["clip"])
                    clip_next = float(forward_stats["next_clip_target"])
                    clip_max = max_clip
                    bound_tol = max(1e-8, 1e-6 * initial_clip)
                    if clip_next > clip_used + bound_tol:
                        direction = "up"
                    elif clip_next < clip_used - bound_tol:
                        direction = "down"
                    else:
                        direction = "hold"

                    epoch_idx = int(getattr(self.vflbase, "epoch", 0)) + 1
                    total_epochs = int(getattr(self.args, "epochs", 0))
                    log_key = (epoch_idx, batch_idx)
                    if getattr(self.vflbase, "_forward_clip_log_key", None) != log_key:
                        self.vflbase._forward_clip_log_key = log_key
                        self.vflbase._forward_clip_log_rows = []
                    self.vflbase._forward_clip_log_rows.append(
                        f"P{party_id}:{clip_used:.4f}->{clip_next:.4f}"
                        f"({direction},cap={clip_max:.4f})"
                    )

                    if len(self.vflbase._forward_clip_log_rows) == self.args.num_passive:
                        risk_threshold = float(getattr(self.vflbase, "proxy_risk_threshold_pct", 0.0))
                        risk_values = list(getattr(
                            self.vflbase,
                            "last_risk_gate_values",
                            getattr(self.vflbase, "last_control_party_risks", [0.0] * self.args.num_passive),
                        ))
                        gate_flags = list(getattr(
                            self.vflbase,
                            "last_risk_gate_flags",
                            [False] * self.args.num_passive,
                        ))
                        trigger_counts = list(getattr(
                            self.vflbase,
                            "risk_gate_trigger_counts",
                            [0] * self.args.num_passive,
                        ))
                        decision_counts = list(getattr(
                            self.vflbase,
                            "risk_gate_decision_counts",
                            [0] * self.args.num_passive,
                        ))
                        party_rates = [
                            100.0 * triggered / decided if decided > 0 else 0.0
                            for triggered, decided in zip(trigger_counts, decision_counts)
                        ]
                        total_decisions = sum(decision_counts)
                        if total_decisions == 0:
                            risk_values = list(getattr(
                                self.vflbase,
                                "last_control_party_risks",
                                risk_values,
                            ))
                        overall_rate = (
                            100.0 * sum(trigger_counts) / total_decisions
                            if total_decisions > 0 else 0.0
                        )
                        risk_text = ",".join(f"{value:.1f}" for value in risk_values)
                        gate_text = "".join("1" if flag else "0" for flag in gate_flags)
                        party_rate_text = ",".join(f"{rate:.1f}" for rate in party_rates)
                        print(
                            f"[FwdClip] E{epoch_idx}/{total_epochs} B{batch_idx + 1} | "
                            f"{' '.join(self.vflbase._forward_clip_log_rows)} | "
                            f"Risk T={risk_threshold:.1f}% values=[{risk_text}] gate={gate_text} | "
                            f"trigger={overall_rate:.1f}% parties=[{party_rate_text}] "
                            f"steps={total_decisions // max(self.args.num_passive, 1)}"
                        )
                sparse_keep_rate = float(np.clip(getattr(self.args, "forward_sparse_keep_rate", 1.0), 1e-6, 1.0))
                sparse_random_ratio = float(np.clip(getattr(self.args, "forward_sparse_random_ratio", 0.0), 0.0, 1.0))
                if sparse_keep_rate < 1.0:
                    output = dp.apply_importance_sparsification(
                        output,
                        keep_rate=sparse_keep_rate,
                        importance_prior=importance_prior,
                        random_ratio=sparse_random_ratio,
                        return_stats=False,
                    )
            return output

        return hook
