import torch
import sys
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

#用于度量两个概率分布之间的差异
#KL散度越小，表示两个分布越相似
class KlDivLoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        logits: bool = True,
    ) -> None:
        super(KlDivLoss, self).__init__(size_average, reduce, reduction)
        self.logits = logits
        self.eps = 1e-12

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if self.logits:
            pred = torch.softmax(pred, dim=-1)

        loss = (-target * torch.log((pred / (target + self.eps)) + self.eps)).sum(
            dim=-1
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction is None:
            return loss
        else:
            sys.exit(f"Invalid reduction type: {self.reduction}")

#训练时希望预测分布Q对应的概率尽可能与目标分布P重合
#软交叉熵损失
class SoftCrossEntropyLoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        logits: bool = True,
    ) -> None:
        super(SoftCrossEntropyLoss, self).__init__(size_average, reduce, reduction)
        self.logits = logits
        self.eps = 1e-12

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if self.logits:
            loss = (-target * torch.log_softmax(pred, dim=-1)).sum(dim=-1)
        else:
            loss = (-target * torch.log(pred + self.eps)).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction is None:
            return loss
        else:
            sys.exit(f"Invalid reduction type: {self.reduction}")

#熵损失
#本质上是计算分布的信息熵
#惩罚预测分布的不确定性：如果模型输出过于均匀（熵大），损失值就大；如果模型输出更确定（熵小），损失就小
class EntropyLoss(_Loss):
    def __init__(self, logits=True, reduction="mean"):
        super(EntropyLoss, self).__init__()
        self.logits = logits
        self.eps = 1e-12
        self.reduction = reduction

    def forward(self, x):
        if self.logits:
            loss = -(F.softmax(x, dim=1) * F.log_softmax(x, dim=1)).sum(dim=-1)
        else:
            loss = -(x * torch.log(x + self.eps)).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction is None:
            return loss
        else:
            sys.exit(f"Invalid reduction type: {self.reduction}")