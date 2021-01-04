import math

from itertools import zip_longest

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from .mnist import DiscriminativeSNN, GenerativeSNN
from criterions import ModifiedCrossEntropy, ReconstructionLoss
from data import SiameseMNIST as SiameseDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = SiameseDataset()
batch_size = 72
alpha = 0.05  # importance of reconstruction loss
beta = 7.0  # l2 regularization
lr = 0.001  # learning rate
T = 0.5  # threshold
E = 10  # epochs
criterion1, criterion2 = ModifiedCrossEntropy(), ReconstructionLoss()

# sample data and prepare it
sizes = [180] #  [30, 60, 120, 180, 360, 540, 1080, 1800]
p = 1 / 5  # size of test set
dataset.sample_traintest(math.ceil(max(sizes) / 180))
dataset.sample_unlabeled()

fig, axs = plt.subplots(len(sizes), 2, figsize=(9, 3 * len(sizes)))
fig.suptitle(
    "Separation of classes for different models",
    y=0.99 - len(sizes) / 100,
    fontsize="x-large",
)

acc_train = []
acc_test = []
acc_train_unlab = []
acc_test_unlab = []

for k, s in enumerate(sizes):
    print(f"Number of labaled pairs: {s}")
    test_s = round(p * s)
    train_s = s - test_s
    for l in (0, 1):
        trainset = torch.utils.data.DataLoader(
            dataset.trainset[:train_s], batch_size=batch_size
        )
        unlabeled = torch.utils.data.DataLoader(
            dataset.unlabeled[:train_s] if l else [], batch_size=batch_size
        )
        testset = torch.utils.data.DataLoader(
            dataset.testset[:test_s], batch_size=test_s
        )

        # initialize nets and optimizers
        net1, net2 = DiscriminativeSNN().to(device), GenerativeSNN().to(device)
        parameters = [{"params": net1.parameters()}, {"params": net2.parameters()}]
        optimizer = torch.optim.RMSprop(parameters, lr=lr, weight_decay=beta)

        # for diagnostics
        # on training set
        train_accuracy = 0.0  # accuracy

        # on test set
        test_accuracy = 0.0  # accuracy
        pos_dists = []  # predicted distances between positive pairs
        neg_dists = []  # predicted distances between negative pairs

        # training
        for i in range(E):
            for tensors1, tensors2 in zip_longest(trainset, unlabeled):
                optimizer.zero_grad()

                ## labeled data
                images1, images2, labels = (t.to(device) for t in tensors1)

                # first net
                outputs1, outputs2 = net1(images1, images2)
                loss1 = criterion1(outputs1, outputs2, labels)

                # second net
                images1_, images2_ = net2(outputs1, outputs2)
                loss2 = criterion2(images1, images1_) + criterion2(images2, images2_)

                # final loss
                loss = loss1 + alpha * loss2

                # evaluation on training set
                if i == E - 1:
                    net1.eval()
                    with torch.no_grad():
                        outputs1, outputs2 = net1(images1, images2)
                        dists = F.pairwise_distance(outputs1, outputs2)
                        preds = (dists <= T).float()
                        train_accuracy += (
                            torch.isclose(labels.squeeze(), preds).sum().item()
                        )
                    net1.train()

                if tensors2:  ## unlabeled data
                    images1, images2 = (t.to(device) for t in tensors2)

                    # first net
                    outputs1, outputs2 = net1(images1, images2)

                    # second net
                    images1_, images2_ = net2(outputs1, outputs2)
                    loss2 = criterion2(images1, images1_) + criterion2(
                        images2, images2_
                    )

                    # total loss
                    loss += alpha * loss2

                ## step
                loss.backward()
                optimizer.step()

            # for plots
            if i == E - 1:
                train_accuracy /= train_s / 100
                if l == 0:
                    acc_train.append(train_accuracy)
                else:
                    acc_train_unlab.append(train_accuracy)

            # evaluation on test set
            if i == E - 1:
                net1.eval()
                net2.eval()
                with torch.no_grad():
                    for tensors in testset:  # one batch
                        images1, images2, labels = (t.to(device) for t in tensors)
                        outputs1, outputs2 = net1(images1, images2)
                        dists = F.pairwise_distance(outputs1, outputs2)
                        preds = (dists <= T).float()
                        test_accuracy += (
                            torch.isclose(labels.squeeze(), preds).sum().item()
                        )

                        for label, dist in zip(labels, dists):
                            if label.item() > 0.5:
                                pos_dists.append(dist.item())
                            else:
                                neg_dists.append(dist.item())
                test_accuracy /= test_s / 100
                if l == 0:
                    acc_test.append(test_accuracy)
                else:
                    acc_test_unlab.append(test_accuracy)
                net1.train()
                net2.train()

        # final model
        ax = axs[k, l]
        sns.kdeplot(neg_dists, ax=ax, fill=True, label="neg")
        sns.kdeplot(pos_dists, ax=ax, fill=True, label="pos")
        ax.vlines(T, 0, ax.get_ylim()[1], linestyles="--", label="$\\tau$")
        ax.text(
            0.01, 0.93, f"Accuracy: {test_accuracy:.2f}%", transform=ax.transAxes,
        )

        if k == 0:
            start = "With" if l else "Without"
            ax.set(title=start + " unlabeled pairs")
        elif k == len(sizes) - 1:
            ax.set(xlabel="Distance")

        if l == 1:
            ax.set(ylabel="")
        else:
            ax.text(
                -0.19,
                0.5,
                f"{s} labeled pairs",
                {"ha": "center", "va": "center"},
                fontsize="large",
                rotation=90,
                transform=ax.transAxes,
            )

        ax.tick_params("x", labelbottom=True)

        ax.legend()

plt.show()

domain = range(len(sizes))
ticks = [str(s) for s in sizes]

fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
fig.suptitle("Accuracy comparison", fontsize="x-large")

axs[0].set(
    title="Training set",
    xlabel="Number of labeled pairs",
    ylabel="Accuracy",
    xticks=domain,
    xticklabels=ticks,
)
axs[0].grid()

axs[0].plot(domain, acc_train, marker="o", label="w/o unlab.")
axs[0].plot(domain, acc_train_unlab, marker="o", label="with unlab.")

axs[1].set(
    title="Test set", xlabel="Number of labeled pairs", xticks=domain, xticklabels=ticks
)
axs[1].grid()

axs[1].plot(domain, acc_test, marker="o")
axs[1].plot(domain, acc_test_unlab, marker="o")
axs[1].yaxis.tick_right()
axs[1].tick_params("y", labelright=True)

fig.tight_layout(pad=2.5)
fig.legend(loc=(0.435, 0.7))

plt.show()
