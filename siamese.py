from torch import nn


class SiameseNetwork(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # convolutional layers
        self.cnn = nn.Sequential(
            # first
            nn.Conv2d(1, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            # second
            nn.Conv2d(16, 256, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(p=0.3)
        )

        # dense layers
        self.fcn = nn.Sequential(
            # first
            nn.Linear(6400, 1024),
            nn.ReLU(inplace=True),
            # second
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            # third
            nn.Linear(128, 2),
        )

    def _forward(self, x):
        # forward pass of one element of a pair
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fcn(output)
        return output

    def forward(self, input1, input2):
        output1 = self._forward(input1)
        output2 = self._forward(input2)
        return output1, output2
