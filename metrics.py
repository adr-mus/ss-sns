import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        torch.nn.Module.__init__(self)
        self.margin = margin

    def forward(self, output1, output2, label):
        dist = F.pairwise_distance(output1, output2)
        loss = (
            (1 - label) * dist.square() + label * F.relu(self.margin - dist).square()
        ).mean()

        return loss


class ModifiedCrossEntropy(nn.Module):
    def forward(self, outputs1, outputs2, labels):
        distances = F.pairwise_distance(outputs1, outputs2)
        predictions = 1 - F.tanh(distances)
        return F.binary_cross_entropy(predictions, labels)


class ReconstructionLoss(nn.Module):
    def forward(self, input, output):
        return torch.norm(output - input)  # / output.shape[0]
