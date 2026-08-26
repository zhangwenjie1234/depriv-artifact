import os
import torch
import numpy as np
from .vflbase import BaseVFL
import utils.datasets as datasets
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


class Attacker(BaseVFL):
    '''
    LIA using cluster approach.
    '''
    def __init__(self, args, model, train_dataset, test_dataset):
        super(Attacker, self).__init__(args, model, train_dataset, test_dataset)
        self.args = args
        self._cached_attack_steps = 0
        print('Attacker: {}'.format(args.attack))

    def train(self):
        return super().train()
    def test(self):
        return super().test()
    def attack(self, data, labels, batch_idx):
        '''
        Implement cluster attack.
        '''
        self._cache_attack_batch(data, labels, batch_idx)

        if batch_idx == self.iteration - 1:
            self._evaluate_cached_epoch()

    def _cache_attack_batch(self, data, labels, batch_idx):
        if len(data) != self.args.num_passive:
            raise ValueError("The number of attack inputs is not equal to the number of passive parties.")

        cached_party_data = []
        for passive_id in range(self.args.num_passive):
            cached_party_data.append(data[passive_id].detach().cpu().clone())

        cache_payload = [cached_party_data, labels.detach().cpu().clone()]
        torch.save(cache_payload, os.path.join(self.data_dir, "data_{}.pt".format(batch_idx)))
        self._cached_attack_steps += 1

    def _evaluate_cached_epoch(self):
        num_classes = datasets.datasets_classes[self.args.dataset]
        cached_batches = sorted(
            [batch_file for batch_file in os.listdir(self.data_dir) if batch_file.endswith(".pt")]
        )
        if len(cached_batches) == 0:
            return

        labels_list = []
        party_data_list = [[] for _ in range(self.args.num_passive)]

        for batch_file in cached_batches:
            batch_data = torch.load(os.path.join(self.data_dir, batch_file), map_location="cpu")
            if len(batch_data) != 2:
                continue
            cached_party_data, labels = batch_data
            if len(cached_party_data) != self.args.num_passive:
                continue
            labels_list.append(labels.reshape(-1).clone())
            for passive_id in range(self.args.num_passive):
                party_tensor = cached_party_data[passive_id]
                party_data_list[passive_id].append(party_tensor.reshape(party_tensor.shape[0], -1))

        if len(labels_list) == 0:
            return

        labels = torch.cat(labels_list, dim=0)
        if labels.shape[0] < num_classes:
            print(
                "[Cluster Warning] Cached samples ({}) are fewer than num_classes ({}); skip epoch evaluation.".format(
                    labels.shape[0], num_classes
                )
            )
            return

        merged_party_data = []
        for passive_id in range(self.args.num_passive):
            if len(party_data_list[passive_id]) == 0:
                merged_party_data.append(np.zeros((0, 1), dtype=np.float32))
                continue
            party_tensor = torch.cat(party_data_list[passive_id], dim=0)
            party_np = party_tensor.cpu().numpy()
            if self.args.tsne:
                party_np = self._tsne(party_np)
            merged_party_data.append(party_np)

        acc_list = self._kmeans(num_classes, merged_party_data, labels)

        for passive_id, acc in enumerate(acc_list):
            print(
                "Average Attack Accuracy of Passive {} (Cluster): {:.2f}%".format(
                    passive_id, acc
                )
            )

        self.metrics.attack_acc.append(acc_list)
        self.metrics.write()
        self._cached_attack_steps = 0


    def _kmeans(self, num_classes, data, labels):
        '''
        K-means clustering with Simulated Batch-Mapping Bias.
        Restored to older weak-attacker baseline.
        '''
        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        acc_list = [0.0] * self.args.num_passive

        samples_per_class_limit = max(1, self.args.batch_size // (num_classes * 2))

        for passive_id in range(self.args.num_passive):
            data_cpu = data[passive_id]
            if isinstance(data_cpu, torch.Tensor):
                data_cpu = data_cpu.cpu().numpy()
            data_cpu = np.asarray(data_cpu, dtype=np.float32)

            if data_cpu.shape[0] < num_classes:
                continue
            
            if not np.isfinite(data_cpu).all():
                data_cpu = np.nan_to_num(data_cpu, nan=0.0, posinf=0.0, neginf=0.0)

            kmeans = KMeans(n_clusters=num_classes, random_state=0, n_init=1)
            kmeans.fit(data_cpu)
            kmeans_labels = kmeans.labels_
            distances = kmeans.transform(data_cpu) 
            
            cluster_to_label = {}
            for i in range(num_classes):
                cluster_indices = np.where(kmeans_labels == i)[0]
                if len(cluster_indices) > 0:
                    limited_indices = cluster_indices[:samples_per_class_limit]
                    
                    sub_distances = distances[limited_indices, i]
                    relative_closest_idx = np.argmin(sub_distances)
                    absolute_closest_idx = limited_indices[relative_closest_idx]
                    
                    cluster_to_label[i] = labels_np[absolute_closest_idx]
                else:
                    cluster_to_label[i] = 0

            # 3. 搴旂敤鏄犲皠骞惰绠楀噯纭巼
            mapped_predictions = np.array([cluster_to_label[l] for l in kmeans_labels])
            correct = (mapped_predictions == labels_np).sum()
            
            attack_acc = 100.0 * correct / max(1, labels_np.shape[0])
            acc_list[passive_id] = attack_acc

        return acc_list

    def _map_cluster_labels(self, cluster_ids, labels_np, num_classes):
        contingency = np.zeros((num_classes, num_classes), dtype=np.int64)
        for cluster_id, label_id in zip(cluster_ids, labels_np):
            if 0 <= cluster_id < num_classes and 0 <= label_id < num_classes:
                contingency[cluster_id, label_id] += 1

        if linear_sum_assignment is not None:
            try:
                row_ind, col_ind = linear_sum_assignment(contingency, maximize=True)
            except TypeError:
                max_count = int(contingency.max()) if contingency.size > 0 else 0
                row_ind, col_ind = linear_sum_assignment(max_count - contingency)
            cluster_to_label = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
        else:
            cluster_to_label = {}
            used_labels = set()
            for cluster_id in range(num_classes):
                label_order = np.argsort(contingency[cluster_id])[::-1]
                chosen_label = 0
                for label_id in label_order:
                    label_id = int(label_id)
                    if label_id not in used_labels:
                        chosen_label = label_id
                        break
                cluster_to_label[cluster_id] = chosen_label
                used_labels.add(chosen_label)

        mapped_labels = np.zeros_like(cluster_ids, dtype=np.int64)
        for cluster_id, label_id in cluster_to_label.items():
            mapped_labels[cluster_ids == cluster_id] = label_id
        return mapped_labels

    
    def _tsne(self, data):
        tsne = TSNE(n_components=3, init='pca', random_state=0)
        tsne.fit_transform(data)

        return tsne.embedding_

