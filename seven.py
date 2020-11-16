import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import ModifiedCrossEntropy, ReconstructionLoss


device = "cuda" if torch.cuda.is_available() else "cpu"


class DiscriminativeSNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # convolutional layers
        # input: 1 @ 28x28
        self.cnn = nn.Sequential(
            # first
            nn.Conv2d(1, 8, kernel_size=3, stride=1),  # 8 @ 26x26
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # 8 @ 13x13
            nn.Dropout2d(p=0.5),
            # second
            nn.Conv2d(8, 8, kernel_size=4, stride=1),  # 8 @ 10x10
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # 8 @ 5x5
            nn.Dropout2d(p=0.5),
            nn.Flatten(),
        )

        # dense layers
        # input: 200
        self.dnn = nn.Sequential(
            # first
            nn.Linear(200, 128),
            nn.ReLU(),
        )

        # output: 128

    def _forward(self, x):
        # forward pass of one element of a pair
        return self.dnn(self.cnn(x))

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
            # first
            nn.Linear(128, 200),
            nn.ReLU(),
        )

        # convolutional layers
        # input: 8 @ 5x5
        self.cnn = nn.Sequential(
            # first
            nn.Upsample(scale_factor=2),  # 8 @ 10x10
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=1),  # 8 @ 13x13
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            # second
            nn.Upsample(scale_factor=2),  # 8 @ 26x26
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1),  # 1 @ 28x28
            nn.Sigmoid(),
            nn.Dropout2d(p=0.5),
        )

        # output: 1 @ 28x28

    def _forward(self, x):
        # forward pass of one element of a pair
        output = self.dnn(x)
        output = output.view(output.shape[0], 8, 5, 5)
        output = self.cnn(output)
        return output

    def forward(self, input1, input2):
        output1 = self._forward(input1)
        output2 = self._forward(input2)
        return output1, output2


def train_model(trainset, testset, unlabeled, alpha, beta, lr=0.001, E=150, T=0.5):
    """ alpha - importance of reconstruction loss
        beta - l2 regularization 
        lr - learning rate 
        E - number of epochs 
        T - threshold """
    # initialize nets, criterions and optimizer
    net1, net2 = DiscriminativeSNN().to(device), GenerativeSNN().to(device)
    nets_parameters = itertools.chain(net1.parameters(), net2.parameters())
    criterion1, criterion2 = ClassificationLoss(), ReconstructionLoss()
    optimizer = torch.optim.RMSprop(nets_parameters, lr=lr, weight_decay=beta)

    # training
    print(f"alpha = {alpha}, beta = {beta}")
    for _ in range(E):  # the number of epochs
        for tensors1, tensors2 in itertools.zip_longest(trainset, unlabeled):
            optimizer.zero_grad()
            loss = 0.0

            if tensors1: # labeled data
                images1, images2, labels = (t.to(device) for t in tensors1)

                # first net
                outputs1, outputs2 = net1(images1, images2)
                loss1 = criterion1(outputs1, outputs2, labels)

                # second net
                images1_, images2_ = net2(outputs1, outputs2)
                loss2 = criterion2(images1, images1_) + criterion2(images2, images2_)

                # final loss
                loss += loss1 + alpha * loss2

            if tensors2: # unlabeled data
                images1, images2 = (t.to(device) for t in tensors2)

                # first net
                outputs1, outputs2 = net1(images1, images2)

                # second net
                images1_, images2_ = net2(outputs1, outputs2)
                loss2 = criterion2(images1, images1_) + criterion2(images2, images2_)

                # final loss
                loss += alpha * loss2

            loss.backward()
            optimizer.step()

    # evaluation - accuracy
    net1.eval()
    net2.eval()
    correct = total = 0
    with torch.no_grad():
        for tensors in testset:
            images1, images2, labels = (t.to(device) for t in tensors)
            outputs1, outputs2 = net1(images1, images2)
            dists = F.pairwise_distance(outputs1, outputs2)
            preds = (dists <= T).float()
            correct += torch.isclose(labels.squeeze(), preds).sum().item()
            total += labels.shape[0]
    accuracy = correct / total
    net1.train()
    net2.train()

    return net1, net2, accuracy


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    from data import SiameseMNIST

    # load data and prepare it
    dataset = SiameseMNIST()
    dataset.load_traintest()
    dataset.load_unlabeled()

    batch_size = 64
    trainset = torch.utils.data.DataLoader(dataset.trainset, batch_size=batch_size)
    unlabeled = torch.utils.data.DataLoader(dataset.unlabeled[:len(dataset.trainset)], batch_size=batch_size)
    testset = torch.utils.data.DataLoader(dataset.testset, batch_size=len(dataset.testset))

    # grid of paramters
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    betas = [0.0001, 0.001, 0.01, 0.1, 1]

    # grid search
    n_loops = 25
    results = {hps: [train_model(trainset, testset, unlabeled, *hps)[2] for _ in range(n_loops)] 
                    for hps in itertools.product(alphas, betas)}
    torch.save(results, "grid_search_results.ptx")

    # results
    avgs = {hps: np.round(np.mean(res), 2) for hps, res in results.items()} 
    sorted_avgs = sorted(avgs.items(), key=lambda el: el[1], reverse=True)
    print("\nalpha, beta | accuracy")
    print(*sorted_avgs, sep="\n")

    # visualization
    k = 6
    best_hps = [sorted_avgs[i][0] for i in range(k)]
    best_results = [results[hps] for hps in best_hps]
    labels = [f"$\\alpha$ = {a}\n$\\beta$ = {b}" for a, b in best_hps]

    plt.figure(figsize=(8, 4))
    plt.title(f"Summary of best hiperparamters ({n_loops} loops each)")
    plt.boxplot(best_results, labels=labels)
    plt.plot(range(1, k + 1), [sorted_avgs[i][1] for i in range(k)], "--", marker=".")
    plt.ylabel("Accuracy")
    plt.grid(axis="y")
    plt.show()

    # alpha, beta | accuracy
    # ((1, 0.1), 0.73)
    # ((1, 1), 0.72)
    # ((2, 0.0001), 0.71)
    # ((2, 0.01), 0.71)
    # ((2, 0.1), 0.71)
    # ((2, 1), 0.71)
    # ((2, 0.001), 0.7)
    # ((1, 0.001), 0.69)
    # ((0.5, 0.1), 0.67)
    # ((1, 0.0001), 0.67)
    # ((1, 0.01), 0.66)
    # ((0.5, 0.0001), 0.61)
    # ((0.5, 1), 0.61)
    # ((0.5, 0.01), 0.6)
    # ((0.5, 0.001), 0.59)
    # ((5, 0.01), 0.58)
    # ((5, 0.1), 0.58)
    # ((5, 1), 0.58)
    # ((5, 0.0001), 0.57)
    # ((5, 0.001), 0.57)
    # ((0.2, 0.1), 0.56)
    # ((0.2, 0.01), 0.55)
    # ((0.1, 0.1), 0.53)
    # ((0.2, 0.0001), 0.53)
    # ((0.1, 0.0001), 0.52)
    # ((0.1, 0.01), 0.52)
    # ((0.2, 0.001), 0.52)
    # ((0.1, 0.001), 0.51)
    # ((0.01, 0.0001), 0.5)
    # ((0.01, 0.001), 0.5)
    # ((0.01, 0.01), 0.5)
    # ((0.01, 0.1), 0.5)
    # ((0.01, 1), 0.5)
    # ((0.05, 0.0001), 0.5)
    # ((0.05, 0.001), 0.5)
    # ((0.05, 0.01), 0.5)
    # ((0.05, 0.1), 0.5)
    # ((0.05, 1), 0.5)
    # ((0.1, 1), 0.5)
    # ((0.2, 1), 0.5)
   
