import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        nn.Module.__init__(self)
        self.margin = margin

    def forward(self, output1, output2, label):
        dist = F.pairwise_distance(output1, output2)
        loss = label * dist.square() + (1 - label) * F.relu(self.margin - dist).square()

        return loss.mean()


class ModifiedCrossEntropy(nn.Module):
    def forward(self, outputs1, outputs2, labels):
        distances = F.pairwise_distance(outputs1, outputs2)
        predictions = 1 - distances.tanh()
        return F.binary_cross_entropy(predictions, labels.squeeze(), reduction="sum")


class ReconstructionLoss(nn.Module):
    def forward(self, input, output):
        input = input.flatten(start_dim=1)
        output = output.flatten(start_dim=1)
        distances = F.pairwise_distance(input, output)
        return distances.sum()
