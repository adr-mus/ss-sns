import os, itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from seven import DiscriminativeSNN, GenerativeSNN
from metrics import ContrastiveLoss, ModifiedCrossEntropy, ReconstructionLoss
from data import SiameseMNIST


device = "cuda" if torch.cuda.is_available() else "cpu"

# load data and prepare it
dataset = SiameseMNIST()
dataset.load_traintest()
dataset.load_unlabeled()
# dataset.sample_traintest()
# dataset.sample_unlabeled()

batch_size = 64
trainset = torch.utils.data.DataLoader(dataset.trainset, batch_size=batch_size)
unlabeled = torch.utils.data.DataLoader(
    dataset.unlabeled[: len(dataset.trainset)], batch_size=batch_size
)
testset = torch.utils.data.DataLoader(dataset.testset, batch_size=len(dataset.testset))
print("Data ready.")

# initialize nets and optimizers
net1, net2 = DiscriminativeSNN().to(device), GenerativeSNN().to(device)
parameters = itertools.chain(net1.parameters(), net2.parameters())
criterion1, criterion2 = ModifiedCrossEntropy(), ReconstructionLoss()
alpha = 1  # importance of reconstruction loss
beta = 0.1  # l2 regularization
lr = 0.001  # learning rate
optimizer = torch.optim.RMSprop(parameters, lr=lr, weight_decay=beta)
print("Nets ready. Beginning training.")

# training parameters
T = 0.5  # threshold
E = 150  # epochs

# for diagnostics
# on training set
train_loss_dis = []  # discriminative part of loss
train_loss_gen = []  # generative part of loss
train_loss_total = []  # total loss
train_accuracy = []  # accuracy
train_pos_distances = []  # predicted distances between positive pairs
train_neg_distances = []  # predicted distances between negative pairs

# on test set
test_loss = []  # total loss
test_accuracy = []  # accuracy
test_pos_distances = []  # predicted distances between positive pairs
test_neg_distances = []  # predicted distances between negative pairs

# training
for i in range(E):
    if i != 0 and i % 30 == 0:
        print(f"{i} epochs passed.")

    # for plots
    train_loss_dis.append(0.0)
    train_loss_gen.append(0.0)
    train_loss_total.append(0.0)
    train_accuracy.append(0.0)
    train_pos_distances.append([])
    train_neg_distances.append([])

    # actual training
    for tensors1, tensors2 in itertools.zip_longest(trainset, unlabeled):
        # reset
        optimizer.zero_grad()
        loss = 0.0

        if tensors1:  # labeled data
            images1, images2, labels = (t.to(device) for t in tensors1)

            # first net
            outputs1, outputs2 = net1(images1, images2)
            loss1 = criterion1(outputs1, outputs2, labels)

            # second net
            images1_, images2_ = net2(outputs1, outputs2)
            loss2 = criterion2(images1, images1_) + criterion2(images2, images2_)

            # final loss
            loss += loss1 + alpha * loss2

            # for plots
            train_loss_dis[-1] += loss1.item()
            train_loss_gen[-1] += loss2.item()

            # evaluation on training set
            net1.eval()
            with torch.no_grad():
                outputs1, outputs2 = net1(images1, images2)
                dists = F.pairwise_distance(outputs1, outputs2)
                preds = (dists <= T).float()
                train_accuracy[-1] += (
                    torch.isclose(labels.squeeze(), preds).sum().item()
                )

                for label, dist in zip(labels, dists):
                    if label.item() > 0.5:
                        train_pos_distances[-1].append(dist.item())
                    else:
                        train_neg_distances[-1].append(dist.item())
            net1.train()

        if tensors2:  # unlabeled data
            images1, images2 = (t.to(device) for t in tensors2)

            # first net
            outputs1, outputs2 = net1(images1, images2)

            # second net
            images1_, images2_ = net2(outputs1, outputs2)
            loss2 = criterion2(images1, images1_) + criterion2(images2, images2_)

            # total loss
            loss += alpha * loss2

            # for plots
            train_loss_gen[-1] += loss2.item()

        # for plots
        train_loss_total[-1] += loss.item()

        # step
        loss.backward()
        optimizer.step()

    # for plots
    train_accuracy[-1] /= len(dataset.trainset) / 100

    test_accuracy.append(0.0)
    test_loss.append(0.0)
    test_pos_distances.append([])
    test_neg_distances.append([])

    # evaluation on test set
    net1.eval()
    net2.eval()
    with torch.no_grad():
        for tensors in testset:  # one batch
            images1, images2, labels = (t.to(device) for t in tensors)
            outputs1, outputs2 = net1(images1, images2)
            images1_, images2_ = net2(outputs1, outputs2)
            loss = criterion1(outputs1, outputs2, labels) + alpha * (
                criterion2(images1, images1_) + criterion2(images2, images2_)
            )
            test_loss[-1] += loss.item()
            dists = F.pairwise_distance(outputs1, outputs2)
            preds = (dists <= T).float()
            test_accuracy[-1] += torch.isclose(labels.squeeze(), preds).sum().item()

            for label, dist in zip(labels, dists):
                if label.item() > 0.5:
                    test_pos_distances[-1].append(dist.item())
                else:
                    test_neg_distances[-1].append(dist.item())
    test_accuracy[-1] /= len(dataset.testset) / 100
    net1.train()
    net2.train()

print("Training finished.\n")

# summary - training
print("Summary of training")
start = 10  # the first epoch to be taken into consideration
domain = range(start, E)
ticks = range(start, E + 1, 20)

# on training set
plt.title("Discriminative loss")
plt.plot(domain, train_loss_dis[start:])
plt.xlabel("Epoch")
plt.xticks(ticks)
plt.grid()
plt.show()

plt.title("Generative loss")
plt.plot(domain, train_loss_gen[start:])
plt.xlabel("Epoch")
plt.xticks(ticks)
plt.grid()
plt.show()

plt.title("Total loss")
plt.plot(domain, train_loss_total[start:])
plt.xlabel("Epoch")
plt.xticks(ticks)
plt.grid()
plt.show()

plt.title("Train accuracy")
plt.plot(domain, train_accuracy[start:])
plt.xlabel("Epoch")
plt.xticks(ticks)
plt.grid()
plt.show()

plt.title("Distances of positive pairs on training set")
plt.plot(domain, [np.max(el) for el in train_pos_distances][start:], label="max")
plt.plot(domain, [np.mean(el) for el in train_pos_distances][start:], label="av")
plt.plot(domain, [np.min(el) for el in train_pos_distances][start:], label="min")
plt.hlines(T, start, E, label="thr", linestyles="--")
plt.xlabel("Epoch")
plt.xticks(ticks)
plt.legend()
plt.grid()
plt.show()

plt.title("Distances of negative pairs on training set")
plt.plot(domain, [np.max(el) for el in train_neg_distances][start:], label="max")
plt.plot(domain, [np.mean(el) for el in train_neg_distances][start:], label="av")
plt.plot(domain, [np.min(el) for el in train_neg_distances][start:], label="min")
plt.hlines(T, start, E, label="thr", linestyles="--")
plt.xlabel("Epoch")
plt.xticks(ticks)
plt.legend()
plt.grid()
plt.show()

plt.title("Separation of classes on training set")
plt.plot(domain, [np.median(el) for el in train_neg_distances][start:], label="med neg")
plt.plot(domain, [np.median(el) for el in train_pos_distances][start:], label="med pos")
plt.hlines(T, start, E, label="thr", linestyles="--")
plt.xlabel("Epoch")
plt.xticks(ticks)
plt.legend()
plt.grid()
plt.show()

# on test set
plt.title("Test loss")
plt.plot(domain, test_loss[start:])
plt.xlabel("Epoch")
plt.xticks(ticks)
plt.grid()
plt.show()

plt.plot(domain, test_accuracy[start:])
plt.title("Test accuracy")
plt.xlabel("Epoch")
plt.xticks(ticks)
plt.grid()
plt.show()

plt.plot(domain, [np.max(el) for el in test_pos_distances][start:], label="max")
plt.plot(domain, [np.mean(el) for el in test_pos_distances][start:], label="av")
plt.plot(domain, [np.min(el) for el in test_pos_distances][start:], label="min")
plt.hlines(T, start, E, label="thr", linestyles="--")
plt.title("Distances of positive pairs on test set")
plt.xlabel("Epoch")
plt.xticks(ticks)
plt.legend()
plt.grid()
plt.show()

plt.plot(domain, [np.max(el) for el in test_neg_distances][start:], label="max")
plt.plot(domain, [np.mean(el) for el in test_neg_distances][start:], label="av")
plt.plot(domain, [np.min(el) for el in test_neg_distances][start:], label="min")
plt.hlines(T, start, E, label="thr", linestyles="--")
plt.title("Distances of negative pairs on test set")
plt.xlabel("Epoch")
plt.xticks(ticks)
plt.legend()
plt.grid()
plt.show()

plt.plot(domain, [np.median(el) for el in test_neg_distances][start:], label="med neg")
plt.plot(domain, [np.median(el) for el in test_pos_distances][start:], label="med pos")
plt.hlines(T, start, E, label="thr", linestyles="--")
plt.title("Separation of classes on test set")
plt.xlabel("Epoch")
plt.xticks(ticks)
plt.legend()
plt.grid()
plt.show()

# final model
pos_dists = train_pos_distances[-1] + test_pos_distances[-1]
neg_dists = train_neg_distances[-1] + test_neg_distances[-1]
plt.title("Separation of classes in the final model")
sns.kdeplot(neg_dists, fill=True, label="neg")
sns.kdeplot(pos_dists, fill=True, label="pos")
plt.vlines(T, 0, 4, linestyles="--", label="thr")
plt.text(
    0.01,
    0.94,
    f"Training accuracy: {train_accuracy[-1]:.2f}%",
    transform=plt.gca().transAxes,
)
plt.text(
    0.01,
    0.87,
    f"Test accuracy: {test_accuracy[-1]:.2f}%",
    transform=plt.gca().transAxes,
)
plt.xlabel("Distances")
plt.legend()
plt.show()
