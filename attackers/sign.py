import torch
from .vflbase import BaseVFL


def _find_classifier(active_model, num_classes):
    layers = [
        module for module in active_model.modules()
        if isinstance(module, torch.nn.Linear) and module.out_features == num_classes
    ]
    return layers[-1] if layers else None


def infer_labels_from_defended_gradients(gradients, passive_id, active_model, num_classes):
    """Map a party's embedding gradient back to classifier class scores."""
    classifier = _find_classifier(active_model, num_classes)
    if classifier is None:
        return None

    flat_gradients = [g.reshape(g.size(0), -1) if g is not None else None for g in gradients]
    widths = [g.size(1) if g is not None else 0 for g in flat_gradients]
    if sum(widths) != classifier.in_features or flat_gradients[passive_id] is None:
        return None

    start = sum(widths[:passive_id])
    party_weight = classifier.weight.detach()[:, start:start + widths[passive_id]]
    class_scores = flat_gradients[passive_id].matmul(party_weight.t())
    return class_scores.argmin(dim=1)


class Attacker(BaseVFL):
    '''
    LIA using gradient sign approach.
    '''
    def __init__(self, args, model, train_dataset, test_dataset):
        super(Attacker, self).__init__(args, model, train_dataset, test_dataset)
        self.args = args
        self.total_acc = 0
        self.round = 0
        print('Attacker: {}'.format(args.attack))

    def train(self):
        return super().train()
    def test(self):
        return super().test()
    def attack(self, grad, labels, batch_idx):
        '''
        Implement gradient sign attack.

        Due to the property of gradient sign, the attack of each passive parties is equal, because they can only use the last layer's gradient.
        '''

        self.round += 1  # inplement attack once

        if grad[0] is None:
            print('Attack Accuracy: Skipped (Gradient Buffered by RL)')
            if batch_idx == self.iteration - 1:
                divisor = self.round if self.round > 0 else 1
                avg_acc = self.total_acc / divisor
                print('Average Attack Accuracy (each epoch): {:.2f}%'.format(avg_acc))
                self.metrics.attack_acc.append(avg_acc)
                self.metrics.write()
                self.total_acc = 0
                self.round = 0
            return
        
        passive_id = int(getattr(self.args, 'attack_id', 0))
        passive_id = min(max(passive_id, 0), len(grad) - 1)
        batch_grad = grad[passive_id]
        projected_labels = None
        if batch_grad is not None and (batch_grad.dim() != 2 or batch_grad.size(1) != self.num_classes):
            projected_labels = infer_labels_from_defended_gradients(
                grad, passive_id, self.model.active, self.num_classes
            )

        if projected_labels is not None:
            attack_res = projected_labels
        elif batch_grad is not None and batch_grad.dim() > 2:
            if self.args.dataset == 'cifar10' and hasattr(self.model.active, 'linear'):
                # For CIFAR-10 HG models, defended gradients arrive as feature-map gradients
                # shaped like [B, C, H, W]. We pool them back to channel gradients and
                # score each class by alignment with the active classifier weights.
                pooled_grad = batch_grad.reshape(batch_grad.size(0), batch_grad.size(1), -1).mean(dim=2)
                linear_weight = self.model.active.linear.weight.detach()
                if linear_weight.size(1) % len(grad) == 0:
                    party_dim = linear_weight.size(1) // len(grad)
                    start = passive_id * party_dim
                    end = start + party_dim
                    party_weight = linear_weight[:, start:end]
                else:
                    party_weight = linear_weight
                if pooled_grad.size(1) == party_weight.size(1):
                    class_scores = torch.matmul(pooled_grad, party_weight.t())
                    attack_res = torch.argmin(class_scores, dim=1)
                else:
                    print('Attack Accuracy: Skipped (CIFAR-10 defended gradient channel size is incompatible with sign attack)')
                    if batch_idx == self.iteration - 1:
                        divisor = self.round if self.round > 0 else 1
                        avg_acc = self.total_acc / divisor
                        print('Average Attack Accuracy (each epoch): {:.2f}%'.format(avg_acc))
                        self.metrics.attack_acc.append(avg_acc)
                        self.metrics.write()
                        self.total_acc = 0
                        self.round = 0
                    return
            else:
                print('Attack Accuracy: Skipped (High-dimensional defended gradient is incompatible with sign attack)')
                if batch_idx == self.iteration - 1:
                    divisor = self.round if self.round > 0 else 1
                    avg_acc = self.total_acc / divisor
                    print('Average Attack Accuracy (each epoch): {:.2f}%'.format(avg_acc))
                    self.metrics.attack_acc.append(avg_acc)
                    self.metrics.write()
                    self.total_acc = 0
                    self.round = 0
                return
        elif (
            batch_grad is not None
            and batch_grad.dim() == 2
            and self.args.dataset == 'cifar10'
            and batch_grad.size(1) != self.num_classes
        ):
            if hasattr(self.model.active, 'linear'):
                linear_weight = self.model.active.linear.weight.detach()
                if linear_weight.size(1) % len(grad) == 0:
                    party_dim = linear_weight.size(1) // len(grad)
                    start = passive_id * party_dim
                    end = start + party_dim
                    party_weight = linear_weight[:, start:end]
                else:
                    party_weight = linear_weight
                if batch_grad.size(1) == party_weight.size(1):
                    class_scores = torch.matmul(batch_grad, party_weight.t())
                    attack_res = torch.argmin(class_scores, dim=1)
                else:
                    print('Attack Accuracy: Skipped (CIFAR-10 defended vector gradient is incompatible with sign attack)')
                    if batch_idx == self.iteration - 1:
                        divisor = self.round if self.round > 0 else 1
                        avg_acc = self.total_acc / divisor
                        print('Average Attack Accuracy (each epoch): {:.2f}%'.format(avg_acc))
                        self.metrics.attack_acc.append(avg_acc)
                        self.metrics.write()
                        self.total_acc = 0
                        self.round = 0
                    return
            else:
                print('Attack Accuracy: Skipped (CIFAR-10 defended vector gradient is incompatible with sign attack)')
                if batch_idx == self.iteration - 1:
                    divisor = self.round if self.round > 0 else 1
                    avg_acc = self.total_acc / divisor
                    print('Average Attack Accuracy (each epoch): {:.2f}%'.format(avg_acc))
                    self.metrics.attack_acc.append(avg_acc)
                    self.metrics.write()
                    self.total_acc = 0
                    self.round = 0
                return
        else:
            attack_res = []
            for g in batch_grad:
                if torch.nonzero(g<0).shape[0] > 1:
                    attack_res.append(torch.nonzero(g<0)[0].unsqueeze(0))
                elif torch.nonzero(g<0).shape[0] == 1:
                    attack_res.append(torch.nonzero(g<0))  # only the ground-truth is negative
                else:
                    attack_res.append(torch.randint(0, 10, (1, 1), device=g.device))
            attack_res = torch.cat(attack_res, dim=0).squeeze()
        
        # the last batch may be smaller than batch_size, but gradient is padded to batch_size, so the attack_res may be larger than labels, even larger than batch_size
        attack_res = attack_res[:labels.shape[0]].reshape(-1)
        correct = attack_res.eq(labels).sum().item()
        attack_acc = 100. * correct / labels.shape[0]
        self.total_acc += attack_acc
        if not getattr(self.args, 'adavfed', False):
            print('Attack Accuracy: {}/{} ({:.2f}%)'.format(correct, labels.shape[0], attack_acc))
        
        if batch_idx == self.iteration - 1:
            
            avg_acc = self.total_acc / self.round
            if not getattr(self.args, 'adavfed', False):
                print('Average Attack Accuracy (each epoch): {:.2f}%'.format(avg_acc))
            self.metrics.attack_acc.append(avg_acc)
            self.metrics.write()

            self.total_acc = 0
            self.round = 0

