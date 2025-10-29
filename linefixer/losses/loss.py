import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, l1_weight=0.2, l2_weight=0.3):
        super().__init__()
        self.bce_weight = bce_weight
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        l1 = self.l1(pred, target)
        l2 = self.l2(pred, target)
        return l2
        return self.bce_weight * bce + self.l1_weight * l1 + self.l2_weight * l2
