import itertools as it

import torch
import torch.utils as utils
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm

from criterions import ModifiedCrossEntropy, ReconstructionLoss


device = "cuda" if torch.cuda.is_available() else "cpu"



class DiscriminativeCNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # self.gn = GaussianNoise(0.15)

        self.activation = nn.ReLU()

        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)

        self.mp1 = nn.MaxPool2d(2)
        self.drop1  = nn.Dropout(0.5)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)

        self.mp2 = nn.MaxPool2d(2)
        self.drop2  = nn.Dropout(0.5)

        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1))
        self.bn3c = nn.BatchNorm2d(128)

        self.ap3 = nn.AvgPool2d(6)

    def _forward(self, x):
        out = x
        ## layer 1-a###
        out = self.conv1a(out)
        out = self.bn1a(out)
        out = self.activation(out)

        ## layer 1-b###
        out = self.conv1b(out)
        out = self.bn1b(out)
        out = self.activation(out)

        ## layer 1-c###
        out = self.conv1c(out)
        out = self.bn1c(out)
        out = self.activation(out)

        out = self.mp1(out)
        out = self.drop1(out)

        ## layer 2-a###
        out = self.conv2a(out)
        out = self.bn2a(out)
        out = self.activation(out)

        ## layer 2-b###
        out = self.conv2b(out)
        out = self.bn2b(out)
        out = self.activation(out)

        ## layer 2-c###
        out = self.conv2c(out)
        out = self.bn2c(out)
        out = self.activation(out)

        out = self.mp2(out)
        out = self.drop2(out)

        ## layer 3-a###
        out = self.conv3a(out)
        out = self.bn3a(out)
        out = self.activation(out)

        ## layer 3-b###
        out = self.conv3b(out)
        out = self.bn3b(out)
        out = self.activation(out)

        ## layer 3-c###
        out = self.conv3c(out)
        out = self.bn3c(out)
        out = self.activation(out)

        out = self.ap3(out)

        out = out.view(-1, 128)

        return out

    def forward(self, input1, input2):
        output1 = self._forward(input1)
        output2 = self._forward(input2)
        return output1, output2


class GenerativeSNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.activation = nn.ReLU()

        self.up1 = nn.Upsample(scale_factor=6)

        self.tconv1a = weight_norm(nn.ConvTranspose2d(128, 256, 3))
        self.bn1a = nn.BatchNorm2d(256)
        self.tconv1b = weight_norm(nn.ConvTranspose2d(256, 512, 1))
        self.bn1b = nn.BatchNorm2d(512)
        self.tconv1c = weight_norm(nn.ConvTranspose2d(512, 256, 1))
        self.bn1c = nn.BatchNorm2d(256)

        self.up2 = nn.Upsample(scale_factor=2)
        self.drop1 = nn.Dropout(0.5)

        self.tconv2a = weight_norm(nn.ConvTranspose2d(256, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.tconv2b = weight_norm(nn.ConvTranspose2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.tconv2c = weight_norm(nn.ConvTranspose2d(256, 128, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(128)

        self.up3 = nn.Upsample(scale_factor=2)
        self.drop1 = nn.Dropout(0.5)

        self.tconv1a = weight_norm(nn.ConvTranspose2d(128, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.tconv1b = weight_norm(nn.ConvTranspose2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.tconv1c = weight_norm(nn.ConvTranspose2d(128, 3, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(3)

    def _forward(self, x):
        out = x.view(128, 1, 1 -1)

        out = self.up1(out)

        ## layer 1-a###
        out = self.tconv1a(out)
        out = self.bn1a(out)
        out = self.activation(out)

        ## layer 1-b###
        out = self.tconv1b(out)
        out = self.bn1b(out)
        out = self.activation(out)

        ## layer 1-c###
        out = self.tconv1c(out)
        out = self.bn1c(out)
        out = self.activation(out)


        out = self.up2(out)
        out = self.drop1(out)

        ## layer 2-a###
        out = self.tconv2a(out)
        out = self.bn2a(out)
        out = self.activation(out)

        ## layer 2-b###
        out = self.tconv2b(out)
        out = self.bn2b(out)
        out = self.activation(out)

        ## layer 2-c###
        out = self.tconv2c(out)
        out = self.bn2c(out)
        out = self.activation(out)


        out = self.up3(out)
        out = self.drop2(out)

        ## layer 3-a###
        out = self.tconv3a(out)
        out = self.bn3a(out)
        out = self.activation(out)

        ## layer 3-b###
        out = self.tconv3b(out)
        out = self.bn3b(out)
        out = self.activation(out)

        ## layer 3-c###
        out = self.tconv3c(out)
        out = self.bn3c(out)

        return out

    def forward(self, input1, input2):
        output1 = self._forward(input1)
        output2 = self._forward(input2)
        return output1, output2
