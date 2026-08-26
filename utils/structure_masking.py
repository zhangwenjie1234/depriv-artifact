import torch

import utils.dp as dp


class StructureMasking:
    """
    Public-data-calibrated structure-aware importance prior.

    The prior is built from clean passive embeddings only, without labels or
    gradients. Each feature dimension receives a structural score based on:
    1. activation energy,
    2. activation spread, and
    3. participation in the dominant covariance eigenspace.

    The resulting static per-dimension prior is then consumed by the existing
    weighted forward clipping/noise routine.
    """

    def __init__(self, args, vflbase_instance):
        self.args = args
        self.vflbase = vflbase_instance

        self.embedding_buffers = {}
        self.sample_counts = {}
        self.static_importance = {}
        self.party_importance_scores = {}

    def _accumulate_embeddings(self, party_id, output):
        flat_output = output.detach().reshape(output.shape[0], -1).to(dtype=torch.float32)
        if party_id not in self.embedding_buffers:
            self.embedding_buffers[party_id] = []
            self.sample_counts[party_id] = 0

        self.embedding_buffers[party_id].append(flat_output.cpu())
        self.sample_counts[party_id] += int(flat_output.shape[0])

    def _compute_structural_importance(self, flat_embeddings):
        if flat_embeddings.numel() == 0:
            return torch.ones(1, dtype=torch.float32) * 0.5

        flat_embeddings = flat_embeddings.to(dtype=torch.float32)
        num_samples, feature_dim = flat_embeddings.shape

        mean = flat_embeddings.mean(dim=0)
        centered = flat_embeddings - mean.unsqueeze(0)
        rms = torch.sqrt(torch.mean(flat_embeddings.pow(2), dim=0) + 1e-8)
        std = torch.sqrt(torch.mean(centered.pow(2), dim=0) + 1e-8)

        if num_samples >= 2 and feature_dim >= 2:
            cov = centered.t().mm(centered) / max(num_samples - 1, 1)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            topk = min(8, feature_dim)
            top_eigvals = torch.clamp(eigvals[-topk:], min=0.0)
            top_eigvecs = eigvecs[:, -topk:]
            eig_mass = float(top_eigvals.sum().item())
            if eig_mass > 1e-12:
                explained = top_eigvals / top_eigvals.sum()
                pca_participation = torch.sum(torch.abs(top_eigvecs) * explained.unsqueeze(0), dim=1)
            else:
                pca_participation = torch.ones(feature_dim, dtype=torch.float32) / float(feature_dim)
        else:
            pca_participation = torch.ones(feature_dim, dtype=torch.float32)

        def normalize(vec):
            v_min, v_max = vec.min(), vec.max()
            if float((v_max - v_min).item()) < 1e-8:
                return torch.ones_like(vec) * 0.5
            return (vec - v_min) / (v_max - v_min)

        rms_norm = normalize(rms)
        std_norm = normalize(std)
        pca_norm = normalize(pca_participation)

        structure_score = 0.4 * rms_norm + 0.3 * std_norm + 0.3 * pca_norm
        return normalize(structure_score)

    def _finalize_static_importance(self, party_id):
        if party_id in self.static_importance:
            return
        if party_id not in self.embedding_buffers:
            return

        flat_embeddings = torch.cat(self.embedding_buffers[party_id], dim=0)
        norm_importance = self._compute_structural_importance(flat_embeddings)
        self.static_importance[party_id] = norm_importance
        self.party_importance_scores[party_id] = float(norm_importance.mean().item())

        topk = min(5, norm_importance.numel())
        top_vals, top_idx = torch.topk(norm_importance, k=topk)
        top_desc = ", ".join(
            "d{}:{:.3f}".format(int(idx), float(val))
            for idx, val in zip(top_idx.tolist(), top_vals.tolist())
        )
        concentration = float(top_vals.mean().item())
        print(f"  [Structure Mask] Party {party_id} structural prior locked!")
        print(
            "    [Structure Mask] Mean={:.4f} Std={:.4f} Samples={} Top{}=[{}]".format(
                float(norm_importance.mean().item()),
                float(norm_importance.std().item()),
                int(self.sample_counts.get(party_id, 0)),
                topk,
                top_desc,
            )
        )
        print(
            "    [Structure Mask] Range=[{:.4f}, {:.4f}] Concentration={:.4f}".format(
                float(norm_importance.min().item()),
                float(norm_importance.max().item()),
                concentration,
            )
        )

    def calibrate_from_aux_data(self, aux_dataset, model, device, max_batches=None):
        if not getattr(self.args, "rl", False):
            return
        if aux_dataset is None or len(aux_dataset) == 0:
            return

        self.embedding_buffers = {}
        self.sample_counts = {}
        self.static_importance = {}
        self.party_importance_scores = {}

        was_training_model = model.training
        was_training_active = model.active.training
        passive_training_flags = [module.training for module in model.passive]

        model.eval()
        model.active.eval()
        for module in model.passive:
            module.eval()

        total_batches = len(aux_dataset) if max_batches is None else min(len(aux_dataset), max_batches)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(aux_dataset):
                if batch_idx >= total_batches:
                    break

                data, _ = batch_data
                data = [d.to(device) for d in data]
                emb, _, _ = model(data)

                for party_id, output in enumerate(emb):
                    self._accumulate_embeddings(party_id, output)

        for party_id in range(self.vflbase.args.num_passive):
            self._finalize_static_importance(party_id)

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

    def get_hook(self, party_id):
        def hook(module, input, output):
            if not getattr(self.args, "rl", False):
                return output

            forward_plan = getattr(self.vflbase, "current_forward_defense_plan", None)
            if forward_plan is None or party_id not in forward_plan:
                return output

            plan = forward_plan[party_id]
            if plan.get("enabled", False):
                importance_prior = self.static_importance.get(party_id)
                output, _ = dp.dp_forward_perturb_adaptive(
                    output,
                    passive_id=party_id,
                    epsilon=plan.get("epsilon", 0.0),
                    device=output.device,
                    norm_tracker=None,
                    initial_clip=plan.get("initial_clip", 1.0),
                    importance_prior=(
                        importance_prior.to(output.device)
                        if importance_prior is not None else None
                    ),
                    return_stats=False,
                )
            return output

        return hook
