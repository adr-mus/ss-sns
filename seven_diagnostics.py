import os, itertools

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn

from seven import DiscriminativeSNN, GenerativeSNN
from metrics import ClassificationLoss, ReconstructionLoss
from data import SiameseMNIST


device = "cuda" if torch.cuda.is_available() else "cpu"

# load data and prepare it
dataset = SiameseMNIST()
# dataset.sample_traintest()
# dataset.sample_unlabeled()
dataset.load_traintest()
dataset.load_unlabeled()

trainset = torch.utils.data.DataLoader(dataset.trainset, batch_size=32, shuffle=True)
unlabeled = torch.utils.data.DataLoader(dataset.unlabeled, batch_size=128, shuffle=True)
testset = torch.utils.data.DataLoader(dataset.testset, batch_size=1)
print("Data ready.")

# initialize nets and optimizers
net1, net2 = DiscriminativeSNN().to(device), GenerativeSNN().to(device)
parameters = itertools.chain(net1.parameters(), net2.parameters())
criterion1, criterion2 = ClassificationLoss(), ReconstructionLoss()
alpha = 0.5 # importance of reconstruction loss
beta = 0.001 # l2 regularization
lr = 0.001
optimizer = torch.optim.RMSprop(parameters, lr=lr, weight_decay=beta)
print("Nets ready. Beginning training.")

# training parameters
T = 0.5 # threshold
E = 150 # epochs

# data for diagnostics
train_loss_dis = []
train_loss_gen = []
train_loss_total = []
test_loss = []
test_accuracy = []
test_pos_distances = []
test_neg_distances = []

# training
for i in range(E):
    if i != 0 and i  % 15 == 0:
        print(f"{i} epochs passed...")

    train_loss_dis.append(0.0)
    train_loss_gen.append(0.0)
    train_loss_total.append(0.0)
    # labeled data
    for tensors in trainset:
        images1, images2, labels = (t.to(device) for t in tensors)
        optimizer.zero_grad()

        # first net
        outputs1, outputs2 = net1(images1, images2)
        loss1 = criterion1(outputs1, outputs2, labels)

        # second net
        images1_, images2_ = net2(outputs1, outputs2)
        loss2 = criterion2(images1, images1_) + criterion2(images2, images2_)

        # final loss
        loss = loss1 + alpha * loss2

        loss.backward()
        optimizer.step()

        train_loss_dis[-1] += loss1.item()
        train_loss_gen[-1] += loss2.item()
        train_loss_total[-1] += loss.item()
    
    # unlabeled data
    for tensors in unlabeled:
        images1, images2 = (t.to(device) for t in tensors)
        optimizer.zero_grad()

        # first net
        outputs1, outputs2 = net1(images1, images2)

        # second net
        images1_, images2_ = net2(outputs1, outputs2)
        loss2 = criterion2(images1, images1_) + criterion2(images2, images2_)

        loss = alpha *  loss2

        loss.backward()
        optimizer.step()

        train_loss_gen[-1] += loss2.item()
        train_loss_total[-1] += loss.item()
    
    # evaluation
    with torch.no_grad():
        test_accuracy.append(0.0)
        test_loss.append(0.0)
        if i >= 10:
            test_pos_distances.append([])
            test_neg_distances.append([])
        for tensors in testset: # one-element batches
            image1, image2, label = (t.to(device) for t in tensors)
            output1, output2 = net1(image1, image2)
            image1_, image2_ = net2(output1, output2)
            loss = criterion1(output1, output2, label) + alpha * (criterion2(image1, image1_) + criterion2(image2, image2_))
            test_loss[-1] += loss
            dist = F.pairwise_distance(output1, output2)
            pred = (dist < T).float()
            test_accuracy[-1] += (label == pred).item()
            if i >= 10:
                if label.item() > 0.5:
                    test_pos_distances[-1].append(dist.item())
                else:
                    test_neg_distances[-1].append(dist.item())
        test_accuracy[-1] /= len(testset)

# summary
plt.plot(train_loss_dis)
plt.title("Discriminative loss")
plt.xlabel("Epoch")
plt.show()

plt.plot(train_loss_gen)
plt.title("Generative loss")
plt.xlabel("Epoch")
plt.show()

plt.plot(train_loss_total)
plt.title("Total loss")
plt.xlabel("Epoch")
plt.show()

plt.plot(test_loss)
plt.title("Test loss")
plt.xlabel("Epoch")
plt.show()

plt.plot(test_accuracy)
plt.title("Test accuracy")
plt.xlabel("Epoch")
plt.show()

plt.plot([max(el) for el in test_pos_distances], label="max")
plt.plot([sum(el) / len(el) for el in test_pos_distances], label="av")
plt.plot([min(el) for el in test_pos_distances], label="min")
plt.hlines(T, 0, 140, label="thr", linestyles="--")
plt.title("Distances of positive pairs")
plt.xlabel("Epoch")
plt.legend()
plt.show()

plt.plot([max(el) for el in test_neg_distances], label="max")
plt.plot([sum(el) / len(el) for el in test_neg_distances], label="av")
plt.plot([min(el) for el in test_neg_distances], label="min")
plt.hlines(T, 0, 140, label="thr", linestyles="--")
plt.title("Distances of negative pairs")
plt.xlabel("Epoch")
plt.legend()
plt.show()

plt.plot([sum(el) / len(el) for el in test_neg_distances], label="av neg")
plt.plot([sum(el) / len(el) for el in test_pos_distances], label="av pos")
plt.hlines(T, 0, 140, label="thr", linestyles="--")
plt.title("Separation of classes")
plt.xlabel("Epoch")
plt.legend()
plt.show()
