import math
import os
import random

import torch

from .vflbase import BaseVFL
from .pmc import Completion, unpack_cached_completion_batch
import utils.datasets as datasets


def stratified_auxiliary_indices(labels, num_aux_samples, sampling_seed):
    """Select a balanced labeled subset without depending on global RNG state."""
    labels_cpu = labels.detach().reshape(-1).cpu()
    total_samples = int(labels_cpu.numel())
    target = min(total_samples, max(0, int(num_aux_samples)))
    if target == 0:
        return [], {}, list(range(total_samples))

    rng = random.Random(int(sampling_seed))
    class_to_indices = {}
    for index, label in enumerate(labels_cpu.tolist()):
        class_to_indices.setdefault(int(label), []).append(index)

    classes = sorted(class_to_indices)
    rng.shuffle(classes)
    for class_indices in class_to_indices.values():
        rng.shuffle(class_indices)

    selected = []
    offsets = {class_id: 0 for class_id in classes}
    while len(selected) < target:
        added = False
        for class_id in classes:
            offset = offsets[class_id]
            class_indices = class_to_indices[class_id]
            if offset < len(class_indices):
                selected.append(class_indices[offset])
                offsets[class_id] = offset + 1
                added = True
                if len(selected) == target:
                    break
        if not added:
            break

    rng.shuffle(selected)
    selected_set = set(selected)
    remaining = [index for index in range(total_samples) if index not in selected_set]
    histogram = {
        class_id: offsets[class_id]
        for class_id in sorted(offsets)
        if offsets[class_id] > 0
    }
    return selected, histogram, remaining


def log_amc_optimizer_stats(vfl, batch_idx):
    if vfl.args.attack != 'amc':
        return
    if (batch_idx + 1) % 100 != 0 and (batch_idx + 1) != vfl.iteration:
        return

    logs = []
    for i, opt in enumerate(vfl.optimizer_passive):
        stats = getattr(opt, "last_step_stats", None)
        if stats is None:
            logs.append("P{}:StdSGD".format(i))
        else:
            logs.append(
                "P{}:S={:.3f}/{:.3f}/{:.3f} G={:.3f} V={:.3f} T={} NF={} RV={} FB={}".format(
                    i,
                    stats["scale_mean"],
                    stats["scale_min"],
                    stats["scale_max"],
                    stats["grad_norm_mean"],
                    stats["velocity_norm_mean"],
                    stats["num_tensors"],
                    stats.get("nonfinite_grad_tensors", 0),
                    stats.get("reset_velocity_tensors", 0),
                    stats.get("fallback_step_tensors", 0),
                )
            )
    if logs:
        print("  [AMC Opt  ] {}".format(" ".join(logs)))


class Attacker(BaseVFL):
    """
    Standard AMC attacker under the passive-party threat model.

    Training stage:
    - only the attacking passive party uses the malicious local optimizer

    Attack stage:
    - completion from the final cached communication embeddings
    - a small labeled auxiliary set
    - optional semi-supervised refinement using unlabeled cached embeddings
    """

    def __init__(self, args, model, train_dataset, test_dataset):
        super().__init__(args, model, train_dataset, test_dataset)
        self.args = args
        print("Attacker: {} (Semi-Supervised Completion Mode)".format(args.attack))

    def train(self):
        return super().train()
    def test(self):
        return super().test()

    def _collect_cached_embeddings(self):
        attack_id = int(self.args.attack_id)
        batch_files = sorted(
            file_name
            for file_name in os.listdir(self.data_dir)
            if file_name.endswith(".pt")
        )
        embeddings = []
        labels = []
        for batch_file in batch_files:
            batch_data = torch.load(
                os.path.join(self.data_dir, batch_file),
                map_location="cpu",
            )
            batch_embeddings, batch_labels = unpack_cached_completion_batch(batch_data)
            embeddings.append(batch_embeddings[attack_id].detach().cpu())
            labels.append(batch_labels.detach().cpu())
        if not embeddings:
            return None, None
        return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)

    def _split_aux_and_unlabeled(self, labels, sampling_seed):
        total_samples = int(labels.size(0))
        num_aux_samples = min(total_samples, self.args.completion_aux_num_labels)
        sampling_desc = "{} samples".format(num_aux_samples)
        if bool(getattr(self.args, "amc_stratified_aux", 0)):
            aux_idx, histogram, unlabeled_idx = stratified_auxiliary_indices(
                labels,
                num_aux_samples,
                sampling_seed,
            )
            return aux_idx, unlabeled_idx, sampling_desc, histogram

        rng = random.Random(int(sampling_seed))
        all_indices = list(range(total_samples))
        rng.shuffle(all_indices)
        aux_idx = all_indices[:num_aux_samples]
        aux_set = set(aux_idx)
        unlabeled_idx = [index for index in range(total_samples) if index not in aux_set]
        histogram = {}
        labels_cpu = labels.detach().reshape(-1).cpu()
        for index in aux_idx:
            label = int(labels_cpu[index].item())
            histogram[label] = histogram.get(label, 0) + 1
        return aux_idx, unlabeled_idx, sampling_desc, histogram

    def _train_semisupervised_completion(
        self,
        model,
        x_labeled,
        y_labeled,
        x_unlabeled,
        attack_seed,
    ):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4,
        )
        criterion = torch.nn.CrossEntropyLoss()
        pseudo_threshold = float(getattr(self.args, "amc_pseudo_threshold", 0.95))
        ssl_lambda = float(getattr(self.args, "amc_ssl_lambda", 0.5))
        warmup_epochs = int(getattr(self.args, "amc_pseudo_warmup_epochs", 0))
        permutation_generator = torch.Generator(device="cpu")
        permutation_generator.manual_seed(int(attack_seed))

        labeled_count = x_labeled.size(0)
        unlabeled_count = x_unlabeled.size(0)
        labeled_batch_size = min(max(8, labeled_count), 64)
        unlabeled_batch_size = min(max(64, labeled_batch_size * 8), 512)
        if unlabeled_count == 0:
            steps_per_epoch = max(1, math.ceil(labeled_count / max(1, labeled_batch_size)))
        else:
            steps_per_epoch = max(1, math.ceil(unlabeled_count / max(1, unlabeled_batch_size)))

        model.train()
        pseudo_accepted = 0
        pseudo_candidates = 0
        for epoch_idx in range(self.args.attack_model_epochs):
            labeled_perm = torch.randperm(
                labeled_count,
                generator=permutation_generator,
            ).to(self.device)
            unlabeled_perm = (
                torch.randperm(
                    unlabeled_count,
                    generator=permutation_generator,
                ).to(self.device)
                if unlabeled_count > 0 else None
            )

            for step_idx in range(steps_per_epoch):
                l_start = (step_idx * labeled_batch_size) % labeled_count
                l_end = min(l_start + labeled_batch_size, labeled_count)
                labeled_idx = labeled_perm[l_start:l_end]
                if labeled_idx.numel() == 0:
                    labeled_idx = labeled_perm[:min(labeled_batch_size, labeled_count)]

                batch_x_l = x_labeled.index_select(0, labeled_idx)
                batch_y_l = y_labeled.index_select(0, labeled_idx)

                pred_l = model(batch_x_l)
                loss = criterion(pred_l, batch_y_l)

                if (
                    epoch_idx >= warmup_epochs
                    and unlabeled_count > 0
                    and ssl_lambda > 0.0
                ):
                    u_start = step_idx * unlabeled_batch_size
                    u_end = min(u_start + unlabeled_batch_size, unlabeled_count)
                    if u_start >= unlabeled_count:
                        u_start = 0
                        u_end = min(unlabeled_batch_size, unlabeled_count)
                    unlabeled_idx = unlabeled_perm[u_start:u_end]
                    if unlabeled_idx.numel() == 0:
                        unlabeled_idx = unlabeled_perm[:min(unlabeled_batch_size, unlabeled_count)]

                    batch_x_u = x_unlabeled.index_select(0, unlabeled_idx)
                    pred_u = model(batch_x_u)
                    probs_u = torch.softmax(pred_u.detach(), dim=1)
                    confidence, pseudo_u = probs_u.max(dim=1)
                    mask = confidence >= pseudo_threshold
                    pseudo_candidates += int(mask.numel())
                    pseudo_accepted += int(mask.sum().item())
                    if torch.any(mask):
                        loss = loss + ssl_lambda * criterion(pred_u[mask], pseudo_u[mask])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        return pseudo_accepted, pseudo_candidates

    def _run_completion_restart(
        self,
        local_embeddings,
        all_labels,
        input_size,
        num_classes,
        restart_id,
    ):
        sampling_seed = int(self.args.completion_aux_seed) + restart_id
        attack_seed = int(getattr(self.args, "amc_attack_seed", 0)) + restart_id
        aux_idx, unlabeled_idx, sampling_desc, histogram = self._split_aux_and_unlabeled(
            all_labels,
            sampling_seed,
        )
        sampling_mode = (
            "class-stratified"
            if bool(getattr(self.args, "amc_stratified_aux", 0))
            else "random"
        )
        print(
            "AMC restart {}/{} auxiliary sampling: {} {}, class_counts={}.".format(
                restart_id + 1,
                int(getattr(self.args, "amc_num_restarts", 1)),
                sampling_mode,
                sampling_desc,
                histogram,
            )
        )
        if not aux_idx:
            return None

        aux_idx_tensor = torch.tensor(aux_idx, dtype=torch.long, device=self.device)
        unlabeled_idx_tensor = torch.tensor(
            unlabeled_idx,
            dtype=torch.long,
            device=self.device,
        )
        labels_device = all_labels.to(self.device)
        emb_device = local_embeddings.to(self.device)
        x_labeled = emb_device.index_select(0, aux_idx_tensor)
        y_labeled = labels_device.index_select(0, aux_idx_tensor)
        x_unlabeled = emb_device.index_select(0, unlabeled_idx_tensor)

        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(attack_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(attack_seed)
            model = Completion(input_size, num_classes, self.args.dataset).to(self.device)
            pseudo_accepted, pseudo_candidates = self._train_semisupervised_completion(
                model,
                x_labeled,
                y_labeled,
                x_unlabeled,
                attack_seed,
            )

            model.eval()
            with torch.no_grad():
                train_correct = 0
                train_total = 0
                eval_batch_size = 1024
                for start in range(0, x_unlabeled.size(0), eval_batch_size):
                    end = min(start + eval_batch_size, x_unlabeled.size(0))
                    preds = model(x_unlabeled[start:end])
                    labels = labels_device.index_select(0, unlabeled_idx_tensor[start:end])
                    train_correct += preds.argmax(dim=1).eq(labels).sum().item()
                    train_total += labels.size(0)

        train_acc = 100.0 * train_correct / max(1, train_total)
        acceptance_rate = 100.0 * pseudo_accepted / max(1, pseudo_candidates)
        print(
            "AMC restart {}/{}: held-out accuracy={:.2f}%, pseudo-label acceptance={:.2f}%.".format(
                restart_id + 1,
                int(getattr(self.args, "amc_num_restarts", 1)),
                train_acc,
                acceptance_rate,
            )
        )
        return train_acc

    def attack(self, init=False):
        _ = init
        attack_id = int(self.args.attack_id)
        local_embeddings, all_labels = self._collect_cached_embeddings()
        if local_embeddings is None:
            print("AMC completion skipped: no cached communication embeddings were found.")
            return

        num_classes = datasets.datasets_classes[self.args.dataset]
        input_size = local_embeddings.reshape(local_embeddings.shape[0], -1).shape[1]
        restart_accs = []
        for restart_id in range(int(getattr(self.args, "amc_num_restarts", 1))):
            restart_acc = self._run_completion_restart(
                local_embeddings,
                all_labels,
                input_size,
                num_classes,
                restart_id,
            )
            if restart_acc is not None:
                restart_accs.append(restart_acc)
        if not restart_accs:
            print("AMC completion skipped: no auxiliary labels were provided.")
            return

        restart_tensor = torch.tensor(restart_accs, dtype=torch.float64)
        train_acc = float(restart_tensor.mean().item())
        restart_std = (
            float(restart_tensor.std(unbiased=True).item())
            if len(restart_accs) > 1
            else 0.0
        )
        print(
            "AMC Attack Accuracy of Passive {}: {:.2f}% +/- {:.2f}% over {} restarts".format(
                attack_id,
                train_acc,
                restart_std,
                len(restart_accs),
            )
        )
        self.metrics.attack_acc.append(train_acc)
        self.metrics.write()
