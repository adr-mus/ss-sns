import torch.nn as nn


class PseudolabelCNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # convolutional layers
        # input: 1 @ 28x28
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),  # 20 @ 24x24
            nn.MaxPool2d(kernel_size=2),  # 20 @ 12x12
            nn.ReLU(),

            nn.Conv2d(20, 40, kernel_size=5),  # 40 @ 8x8
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(kernel_size=2),  # 40 @ 4x4
            nn.ReLU(),
        )

        self.flatten = nn.Flatten() 

        # dense layers
        # input: 640
        self.dnn = nn.Sequential(
            nn.Linear(640, 150),
            nn.ReLU(),
            nn.Dropout2d(),
            
            nn.Linear(150, 10),
            nn.ReLU(),

            nn.LogSoftmax(dim=1),
        )

        # output: 10

    def forward(self, x):
        return self.dnn(self.flatten(self.cnn(x)))


class PseudolabelSNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(1, 15, 5)
        self.pool = nn.MaxPool2d(3)
        
        self.conv2 = nn.Conv2d(15, 30, 6)

        self.fc = nn.Linear(270, 128)

        self.activation = nn.ReLU()
    
    def _forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)

        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        x = self.activation(x)

        return x

    def forward(self, x1, x2):
        out1 = self._forward(x1)
        out2 = self._forward(x2)

        return out1, out2