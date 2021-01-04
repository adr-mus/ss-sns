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


# def train_model(dataset, batch_size, E, lr, alpha, beta, T):
#     # initialize data loaders
#     trainset = utils.data.DataLoader(dataset.trainset, batch_size=batch_size)
#     unlabeled = utils.data.DataLoader(dataset.unlabeled, batch_size=batch_size)
#     testset = utils.data.DataLoader(dataset.testset, batch_size=len(dataset.testset))

#     # initialize nets, criterions and optimizer
#     net1, net2 = DiscriminativeSNN().to(device), GenerativeSNN().to(device)
#     criterion1, criterion2 = ModifiedCrossEntropy(), ReconstructionLoss()
#     parameters = [{"params": net1.parameters()}, {"params": net2.parameters()}]
#     optimizer = torch.optim.RMSprop(parameters, lr=lr, weight_decay=beta)

#     # training
#     for _ in range(E):  # the number of epochs
#         for pos, neg, unlab in zip(it.cycle(positive), it.cycle(negative), unlabeled):
#             optimizer.zero_grad()

#             ## labeled data
#             imgs1, imgs2, labs = pos
#             imgs1_, imgs2_, labs_ = neg
#             images1 = torch.cat([imgs1, imgs1_], dim=0).to(device)
#             images2 = torch.cat([imgs2, imgs2_], dim=0).to(device)
#             labels = torch.cat([labs, labs_], dim=0).to(device)

#             # first net
#             outputs1, outputs2 = net1(images1, images2)
#             loss1 = criterion1(outputs1, outputs2, labels)

#             # second net
#             images1_, images2_ = net2(outputs1, outputs2)
#             loss2 = criterion2(images1, images1_) + criterion2(images2, images2_)

#             # final loss
#             loss = loss1 + alpha * loss2

#             ## unlabeled data
#             images1, images2 = (t.to(device) for t in unlab)

#             # first net
#             outputs1, outputs2 = net1(images1, images2)

#             # second net
#             images1_, images2_ = net2(outputs1, outputs2)
#             loss2 = criterion2(images1, images1_) + criterion2(images2, images2_)

#             # final loss
#             loss += alpha * loss2

#             ## step
#             loss.backward()
#             optimizer.step()

#     # evaluation - accuracy
#     net1.eval()
#     correct = 0
#     with torch.no_grad():
#         for tensors in testset:
#             images1, images2, labels = (t.to(device) for t in tensors)
#             outputs1, outputs2 = net1(images1, images2)
#             dists = F.pairwise_distance(outputs1, outputs2)
#             preds = (dists <= T).float()
#             correct += torch.isclose(labels.squeeze(), preds).sum().item()
#     accuracy = correct / len(dataset.testset)
#     net1.train()

#     return net1, net2, accuracy


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    from data import SiameseMNIST

    # data
    dataset = SiameseMNIST()
    dataset.load_traintest()
    dataset.load_unlabeled()

    # grid of paramters
    batch_sizes = [32]
    Es = [150]  # epochs
    lrs = [0.001]  # learning rate
    alphas = [0.05]  # importance of the reconstruction loss
    betas = [2.0]  # l2 regularization
    Ts = [0.5]  # threshold

    # grid search
    grid = itertools.product(batch_sizes, Es, lrs, alphas, betas, Ts)
    template = "batch size = {}, E = {}, lr = {}, alpha = {}, beta = {}, T = {}"
    n_loops = 1

    results = {}
    for hps in grid:
        print(template.format(*hps))
        results[hps] = []
        for _ in range(n_loops):
            results[hps].append(train_model(dataset, *hps)[2])

    # results
    avgs = {hps: np.mean(res) for hps, res in results.items()}
    sorted_avgs = sorted(avgs.items(), key=lambda el: el[1], reverse=True)
    print("\nbatch_size, E, lr, alpha, beta, T | accuracy")
    print(*sorted_avgs[:50], sep="\n")

    # visualization
    k = 1  # show k best models
    template = "batch size = {}\nE = {}\nlr = {}\n$\\alpha$ = {}\n$\\beta$ = {}\nT = {}"
    best_hps = [sorted_avgs[i][0] for i in range(k)]
    best_results = [results[hps] for hps in best_hps]
    labels = [template.format(*hps) for hps in best_hps]

    plt.figure(figsize=(4, 4))
    plt.title(f"Summary of best hiperparamters ({n_loops} loops each)")
    plt.boxplot(best_results, labels=labels)
    plt.plot(range(1, k + 1), [sorted_avgs[i][1] for i in range(k)], "--", marker=".")
    plt.ylabel("Accuracy")
    plt.grid(axis="y")
    plt.show()
