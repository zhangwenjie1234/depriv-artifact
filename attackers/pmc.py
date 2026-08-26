import os
import random
import time
import torch
from torch import nn
from .vflbase import BaseVFL
import utils.datasets as datasets


class Flatten(nn.Module):
    """Flatten the input."""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Completion(nn.Module):
    """Completion model with regularization to prevent overfitting."""

    def __init__(self, input_size, num_classes, dataset):
        super(Completion, self).__init__()
        if dataset == "cifar100":
            self.completion = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, num_classes),
            )
        else:
            self.completion = nn.Sequential(
                Flatten(),
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(64, num_classes),
            )

    def forward(self, x):
        pred = self.completion(x)
        return pred


def unpack_cached_completion_batch(batch_data):
    if len(batch_data) == 4:
        _, emb, _, labels = batch_data
        return emb, labels
    if len(batch_data) == 3:
        emb, _, labels = batch_data
        return emb, labels
    raise ValueError(
        "Unexpected cached completion batch format: expected 3 or 4 items, "
        f"got {len(batch_data)}."
    )


class Attacker(BaseVFL):
    """
    Passive model completion (PMC) with stricter held-out evaluation.
    """

    def __init__(self, args, model, train_dataset, test_dataset):
        super(Attacker, self).__init__(args, model, train_dataset, test_dataset)
        self.args = args
        print("Attacker: {} (Strict Evaluation Mode)".format(args.attack))

    def train(self):
        return super().train()
    def test(self):
        return super().test()
    def attack(self, init=False):
        """
        Train on a small randomly sampled auxiliary set.
        Evaluate on the remaining held-out samples.
        """
        self.completion_models = []
        self.completion_optimizers = []
        self.completion_loss = torch.nn.CrossEntropyLoss()

        all_batches = sorted(
            [batch_file for batch_file in os.listdir(self.data_dir) if batch_file.endswith(".pt")]
        )
        if len(all_batches) == 0:
            return

        data = torch.load(os.path.join(self.data_dir, all_batches[0]), map_location="cpu")
        tmp_emb, tmp_labels = unpack_cached_completion_batch(data)
        input_size_list = [
            tmp_emb[passive_id].reshape(tmp_emb[passive_id].shape[0], -1).shape[1]
            for passive_id in range(self.args.num_passive)
        ]
        num_classes = datasets.datasets_classes[self.args.dataset]

        for passive_id in range(self.args.num_passive):
            attack_model = Completion(input_size_list[passive_id], num_classes, self.args.dataset)
            self.completion_models.append(attack_model.to(self.device))
            self.completion_optimizers.append(
                torch.optim.Adam(
                    attack_model.parameters(),
                    lr=0.001,
                    weight_decay=1e-4,
                )
            )

        batch_sizes = {}
        sample_refs = []
        for batch_file in all_batches:
            batch_data = torch.load(os.path.join(self.data_dir, batch_file), map_location="cpu")
            _, labels = unpack_cached_completion_batch(batch_data)
            batch_sizes[batch_file] = labels.size(0)
            sample_refs.extend((batch_file, idx) for idx in range(labels.size(0)))

        total_samples = len(sample_refs)
        if total_samples == 0:
            return

        num_train_samples = min(total_samples, self.args.completion_aux_num_labels)
        sampling_desc = "{} samples".format(num_train_samples)
        if num_train_samples == 0:
            print("PMC completion skipped: no auxiliary labels were provided.")
            return

        rng = random.Random(self.args.completion_aux_seed)
        selected_refs = rng.sample(sample_refs, num_train_samples)
        aux_by_batch = {}
        for batch_file, idx in selected_refs:
            aux_by_batch.setdefault(batch_file, []).append(idx)

        print("Completion auxiliary sampling: random {}.".format(sampling_desc))

        for passive_id in range(self.args.num_passive):
            self.completion_models[passive_id].train()

            for epoch_idx in range(self.args.attack_model_epochs):
                for batch_file, sample_indices in aux_by_batch.items():
                    batch_data = torch.load(os.path.join(self.data_dir, batch_file), map_location="cpu")
                    emb, labels = unpack_cached_completion_batch(batch_data)

                    idx_tensor = torch.tensor(sample_indices, dtype=torch.long)
                    emb_gpu = emb[passive_id].index_select(0, idx_tensor).to(self.device)
                    labels_gpu = labels.index_select(0, idx_tensor).to(self.device)

                    pred = self.completion_models[passive_id](emb_gpu)
                    loss = self.completion_loss(pred, labels_gpu)
                    self.completion_optimizers[passive_id].zero_grad()
                    loss.backward()
                    self.completion_optimizers[passive_id].step()

        tot_acc = []
        for passive_id in range(self.args.num_passive):
            self.completion_models[passive_id].eval()
            correct = 0
            test_samples_count = 0

            with torch.no_grad():
                for batch_file in all_batches:
                    batch_data = torch.load(os.path.join(self.data_dir, batch_file), map_location="cpu")
                    emb, labels = unpack_cached_completion_batch(batch_data)

                    holdout_mask = torch.ones(batch_sizes[batch_file], dtype=torch.bool)
                    for idx in aux_by_batch.get(batch_file, []):
                        holdout_mask[idx] = False
                    if not torch.any(holdout_mask):
                        continue

                    idx_tensor = holdout_mask.nonzero(as_tuple=False).squeeze(1)
                    emb_gpu = emb[passive_id].index_select(0, idx_tensor).to(self.device)
                    labels_gpu = labels.index_select(0, idx_tensor).to(self.device)

                    preds = self.completion_models[passive_id](emb_gpu)
                    correct += preds.argmax(dim=1).eq(labels_gpu).sum().item()
                    test_samples_count += labels_gpu.size(0)

            acc = 100.0 * correct / max(1, test_samples_count)
            tot_acc.append(acc)
            print(
                "Average Attack Accuracy of Passive {} (Held-out Test Set): {:.2f}%".format(
                    passive_id, acc
                )
            )

        self.metrics.attack_acc.append(tot_acc)
        self.metrics.write()

