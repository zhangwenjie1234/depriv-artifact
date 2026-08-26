import torch
import numpy as np
from .vflbase import BaseVFL
import utils.datasets as datasets
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


class Attacker(BaseVFL):
    '''
    LIA using cluster approach.
    '''
    def __init__(self, args, model, train_dataset, test_dataset):
        super(Attacker, self).__init__(args, model, train_dataset, test_dataset)
        self.args = args
        self.total_acc = [0] * self.args.num_passive
        self.round = 0
        print('Attacker: {}'.format(args.attack))

    def train(self):
        return super().train()
    def test(self):
        return super().test()
    def attack(self, data, labels, batch_idx):
        '''
        Implement cluster attack.
        '''
        num_classes = datasets.datasets_classes[self.args.dataset]

        if labels.shape[0] != self.args.batch_size:
            pass
        else:
            self.round += 1

            if self.args.use_emb:
                passive_emb = []
                if self.args.tsne:
                    for passive_id in range(len(data)):
                        passive_emb.append(self._tsne(data[passive_id].detach().cpu().numpy()))
                else:
                    for passive_id in range(len(data)):
                        passive_emb.append(data[passive_id].reshape(data[passive_id].shape[0], -1).detach().cpu().numpy())  
                # Using K-means to cluster embeddings for different passive parties.
                acc_list = self._kmeans(num_classes, passive_emb, labels)
            else:
                if len(data) != self.args.num_passive:
                    raise ValueError("The number of gradients is not equal to the number of passive parties.")
                
                # process grad
                passive_grad = []
                if self.args.tsne:
                    for passive_id in range(len(data)):
                        passive_grad.append(self._tsne(data[passive_id].detach().cpu().numpy()))
                    passive_grad = torch.Tensor(passive_grad) 
                else:
                    for passive_id in range(len(data)):
                        passive_grad.append(data[passive_id].reshape(data[passive_id].shape[0], -1).detach().cpu().numpy())  # KMeans expected dim <= 2.
                # Using K-means to cluster gradients for different passive parties.
                acc_list = self._kmeans(num_classes, passive_grad, labels)

            # update total accuracy
            for passive_id in range(self.args.num_passive):
                self.total_acc[passive_id] += acc_list[passive_id]

        # calculate average accuracy and write metrics
        if batch_idx == self.iteration - 1:
            avg_acc = []
            for passive_id in range(self.args.num_passive):
                avg_acc.append(self.total_acc[passive_id] / self.round)
                print('Average Attack Accuracy of Passive {} (each epoch): {:.2f}%'.format(passive_id, avg_acc[passive_id]))
            self.metrics.attack_acc.append(avg_acc)
            self.metrics.write()

            self.total_acc = [0] * self.args.num_passive
            self.round = 0


    def _kmeans(self, num_classes, data, labels):
        '''
        K-means clustering.
        '''
        # initialize the attack predicted labels
        cluster_labels = torch.randint(low=0, high=num_classes, size=labels.shape, dtype=torch.long, device=labels.device)

        acc_list = [0] * self.args.num_passive
        for passive_id in range(self.args.num_passive):
        #     # algorithm{'lloyd', 'elkan', 'auto', 'full'}, default='lloyd'
        #     kmeans = KMeans(n_clusters=num_classes, random_state=0, n_init='auto')
        #     kmeans.fit(data[passive_id])
        #     kmeans_labels = kmeans.predict(data[passive_id])

        #     # calculate the closest point to the center
        #     dis = kmeans.transform(data[passive_id]).min(axis=1)  # n_samples * n_clusters
            X = data[passive_id]
            if hasattr(X, "detach"):
                X = X.detach().cpu().numpy()
            else:
                X = np.asarray(X)

            n_unique = np.unique(X, axis=0).shape[0]
            k_eff = min(num_classes, n_unique)
            if k_eff <= 1:
                acc_list[passive_id]=0.0
                continue
            
            X = X + (1e-6 * np.random.randn(*X.shape))
            kmeans = KMeans(n_clusters=k_eff, random_state=0, n_init=10)
            kmeans.fit(X)
            kmeans_labels = kmeans.labels_ # numpy
            dists = kmeans.transform(X)  # [n_samples, k_eff] (numpy)
            
            always_correct = 0
            for i in range(k_eff): # [BUG-FIX NOTE] 寰幆搴斿埌 k_eff, 浣嗕繚鎸佸師閫昏緫
                i_idx = np.where(kmeans_labels == i)[0] # numpy
                if i_idx.shape[0] == 0:
                    continue
                always_correct += 1
                
                closest_idx = i_idx[np.argmin(dists[i_idx, i])]

                # update labels according to the closest point

                cluster_labels[torch.from_numpy(i_idx)] = labels[closest_idx]

            # calculate the accuracy
            correct = cluster_labels.eq(labels).sum().item()
            attack_acc = 100. * (correct - always_correct) / (labels.shape[0] - always_correct)
            acc_list[passive_id] = attack_acc

        return acc_list

    
    def _tsne(self, data):
        tsne = TSNE(n_components=3, init='pca', random_state=0)
        tsne.fit_transform(data)

        return tsne.embedding_
