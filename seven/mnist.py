import itertools as it

import torch
import torch.utils as utils
import torch.nn as nn
import torch.nn.functional as F

from criterions import ModifiedCrossEntropy, ReconstructionLoss


device = "cuda" if torch.cuda.is_available() else "cpu"


class DiscriminativeSNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # convolutional layers
        # input: 1 @ 28x28
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # 8 @ 28x28
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),  # 8 @ 14x14
            nn.Dropout2d(p=0.5),

            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2),  # 8 @ 14x14
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),  # 8 @ 7x7
            nn.Dropout2d(p=0.5),
        )

        self.flatten = nn.Flatten()

        # dense layers
        # input: 392
        self.dnn = nn.Sequential(
            nn.Linear(392, 128),
            nn.ReLU(),
        )

        # output: 128

    def _forward(self, x):
        # forward pass of one element of a pair
        return self.dnn(self.flatten(self.cnn(x)))

    def forward(self, input1, input2):
        output1 = self._forward(input1)
        output2 = self._forward(input2)
        return output1, output2


class GenerativeSNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # dense layers
        # input: 128
        self.dnn = nn.Sequential(
            nn.Linear(128, 392),
            nn.ReLU(),
        )

        # convolutional layers
        # input: 8 @ 7x7
        self.cnn = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 8 @ 14x14

            nn.ConvTranspose2d(8, 8, kernel_size=5, stride=1, padding=2),  # 8 @ 14x14
            nn.ReLU(),

            nn.Dropout2d(p=0.5),
            nn.Upsample(scale_factor=2),  # 8 @ 28x28

            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),  # 1 @ 28x28
            nn.Sigmoid(),
            
            nn.Dropout2d(p=0.5),
        )

        # output: 1 @ 28x28

    def _forward(self, x):
        # forward pass of one element of a pair
        output = self.dnn(x)
        output = output.view(output.shape[0], 8, 7, 7)
        output = self.cnn(output)
        
        return output

    def forward(self, input1, input2):
        output1 = self._forward(input1)
        output2 = self._forward(input2)

        return output1, output2
