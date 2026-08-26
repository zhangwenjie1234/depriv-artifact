import torch
from torch import nn
from .vflbase import BaseVFL
import utils.datasets as datasets
import heapq
import os


class Flatten(nn.Module):
    '''Flatten the input'''
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Completion(nn.Module):
    '''Completion model.'''
    def __init__(self, input_size, num_classes, dataset):
        super(Completion, self).__init__()
        if dataset == 'cifar100':
            self.completion = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(512, 128),  # adjust dimensions as needed
                nn.Linear(128, num_classes),
            )
        else:
            self.completion = nn.Sequential(
                Flatten(),
                nn.Linear(input_size, 128),  # adjust dimensions as needed
                nn.Linear(128, num_classes),
            )
        print("Completion Model", self.completion)

    def forward(self, x):
        pred = self.completion(x)
        return pred


class Attacker(BaseVFL):
    '''
    LIA using model completion approach.
    '''
    def __init__(self, args, model, train_dataset, test_dataset):
        super(Attacker, self).__init__(args, model, train_dataset, test_dataset)
        self.args = args
        self.total_acc = 0
        self.round = 0
        print('Attacker: {}'.format(args.attack))

        # Initialize the inference head
        self.completion_models = []

        # Initialize the optimizer for the inference head
        self.completion_optimizers = []

    def train(self):
        return super().train()
    def test(self):
        return super().test()
    def attack(self, init=False):
        '''
        Forward pass through the original model
        Pass the output through the inference head
        '''
        if init:
            file_batch_0 = os.listdir(self.data_dir)[0]
            data = torch.load(os.path.join(self.data_dir, file_batch_0), map_location='cpu')
            tmp_emb, _, _ = data
            input_size_list = []
            for passive_id in range(self.args.num_passive):
                input_size_list.append(tmp_emb[passive_id].reshape(tmp_emb[passive_id].shape[0], -1).shape[1])
            num_classes = datasets.datasets_classes[self.args.dataset]
            for passive_id in range(self.args.num_passive):
                attack_model = Completion(input_size_list[passive_id], num_classes, self.args.dataset)
                self.completion_models.append(attack_model.to(self.device))
                self.completion_optimizers.append(torch.optim.Adam(attack_model.parameters(), lr=self.args.lr_attack_model))
            self.completion_loss = torch.nn.CrossEntropyLoss()

        # select batches with largest gradient norm
        heaps = []
        for passive_id in range(self.args.num_passive):
            heap = []
            for batch_file in os.listdir(self.data_dir):
                # +++ [淇敼] 鍔犺浇鏁版嵁鍒?CPU (榛樿) +++
                batch_data = torch.load(os.path.join(self.data_dir, batch_file), map_location='cpu')
                # +++++++++++++++++++++++++++++++++++++
                emb, grad, labels = batch_data         
                grad_norm = torch.norm(grad[passive_id]).item()
                if len(heap) < 1:  # the number of batches to select
                    heapq.heappush(heap, (grad_norm, batch_data))
                elif grad_norm > heap[0][0]:
                    heapq.heapreplace(heap, (grad_norm, batch_data))
            selected_batches = [item[1] for item in heap]
            heaps.append(selected_batches)

        # train the completion model
        for passive_id in range(self.args.num_passive):
            self.completion_models[passive_id].train()
            for _ in range(self.args.attack_model_epochs):
                for batch_data in heaps[passive_id]:
                    emb, grad, labels = batch_data
                    
                    emb_gpu = emb[passive_id].to(self.device)
                    labels_gpu = labels.to(self.device)

                    pred = self.completion_models[passive_id](emb_gpu)
                    loss = self.completion_loss(pred, labels_gpu)
                    self.completion_optimizers[passive_id].zero_grad()
                    loss.backward()
                    self.completion_optimizers[passive_id].step()

        # evaluate the completion model
        tot_acc = []
        for passive_id in range(self.args.num_passive):
            self.completion_models[passive_id].eval()
            correct = 0
            with torch.no_grad():
                for batch_file in os.listdir(self.data_dir):
                    batch_data = torch.load(os.path.join(self.data_dir, batch_file), map_location='cpu')
                    emb, _, labels = batch_data
                    emb_gpu = emb[passive_id].to(self.device)
                    labels_gpu = labels.to(self.device)

                    correct += self.completion_models[passive_id](emb_gpu).argmax(dim=1).eq(labels_gpu).sum().item()
            acc = 100. * correct / self.train_dataset_len
            tot_acc.append(acc)
            print('Average Attack Accuracy of Passive {} (each epoch): {:.2f}%'.format(passive_id, acc))
            
        self.metrics.attack_acc.append(tot_acc)
        self.metrics.write()
