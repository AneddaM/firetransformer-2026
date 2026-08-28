import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, label_smoothing=0.05):
        super().__init__(); self.gamma=gamma; self.alpha=alpha; self.label_smoothing=label_smoothing
    def forward(self, logits, targets):
        targets = targets.float()
        if self.label_smoothing:
            targets = targets * (1-self.label_smoothing) + 0.5*self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = targets*p + (1-targets)*(1-p)
        alpha_t = targets*self.alpha + (1-targets)*(1-self.alpha)
        return (alpha_t * (1-pt).pow(self.gamma) * bce).mean()
