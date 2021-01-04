import itertools as it

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from pl_blog.mnist import PseudolabelCNN
from data import PseudolabelMNIST as Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def test():
    net.eval()
    with torch.no_grad():
        # evaluation on training set
        correct = 0
        for tensors in trainset:
            images, labels = (t.to(device) for t in tensors)
            outputs = net(images)

            preds = torch.argmax(outputs, dim=1)
            correct += torch.isclose(labels.squeeze(), preds).sum().item()
        accuracy = 100 * correct / len(dataset.trainset)
        train_log["accuracy"].append(accuracy)

        # evaluation on test set
        test_log["total_loss"].append(0.0)
        correct = 0
        for tensors in testset:
            images, labels = (t.to(device) for t in tensors)
            outputs = net(images)
            
            loss = criterion(outputs, labels.squeeze())
            test_log["total_loss"][-1] += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += torch.isclose(labels.squeeze(), preds).sum().item()
        accuracy = 100 * correct / len(dataset.testset)
        test_log["accuracy"].append(accuracy)


def alpha(step):
    if step < S1:
        return 0
    elif step < S2:
        return (step - S1) / (S2 - S1) * max_alpha
    else:
        return max_alpha


# load data and prepare it
dataset = Dataset()
dataset.sample_traintest()

batch_size = 64
trainset = torch.utils.data.DataLoader(dataset.trainset, batch_size=batch_size)
unlabeled = torch.utils.data.DataLoader(dataset.unlabeled, batch_size=3 * batch_size)
testset = torch.utils.data.DataLoader(dataset.testset, batch_size=len(dataset.testset))
print("Data ready.")

# initialize net and optimizer
net = PseudolabelCNN().to(device)
criterion = nn.NLLLoss()
S1, S2 = 60, 80
max_alpha = 3.0
lr = 0.01
momentum = 0.5
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
print("Net ready.")

# training parameters
E = 90  # epochs

# logs
train_log = {
    "total_loss": [],
    "accuracy": [],
}

test_log = {
    "total_loss": [],
    "accuracy": [],
}

print("Beginning training.")
# stage 1: only labeled
for i in range(min(E, S1)):
    if i != 0 and i % 30 == 0:
        print(f"{i} epochs passed.")

    train_log["total_loss"].append(0.0)
    net.train()
    for tensors in trainset:
        images, labels = (t.to(device) for t in tensors)
        outputs = net(images)
        loss = criterion(outputs, labels.squeeze())

        train_log["total_loss"][-1] += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test()

# stage 2: unlabeled + labeled for fine-tuning
step = S1
for i in range(S1, E):
    if i % 30 == 0:
        print(f"{i} epochs passed.")

    net.eval()
    with torch.no_grad():
        pseudo_labels = []
        for images in unlabeled:
            images = images.to(device)
            outputs = net(images)
            pseudo_labels.append(outputs.argmax(dim=1))
            
    train_log["total_loss"].append(0.0)
    net.train()
    count = 0
    for (images1, labels), images2, p_labels in zip(it.cycle(trainset), unlabeled, pseudo_labels):
        images1, labels, images2 = images1.to(device), labels.to(device), images2.to(device)
        
        outputs1 = net(images1)
        outputs2 = net(images2)
        loss = criterion(outputs1, labels.squeeze()) + alpha(step) * criterion(outputs2, p_labels)

        train_log["total_loss"][-1] += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        count += 1
        if count % len(trainset) == 0:
            step += 0.1
            count = 0
        
    test()

print("Training finished.\n")

# summary - training
print("Summary of training")
start = 0  # the first epoch to be taken into consideration
k = 4
domain = range(start, E)
ticks = list(range(start, E + 1, (E - start) // k)) 

# on training set
fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
fig.suptitle("Evaluation on training set", y=0.99, fontsize="x-large")

axs[0].plot(domain, train_log["total_loss"][start:])
axs[0].set(xlabel="Epoch", title="Total loss", xticks=ticks)
axs[0].grid()

axs[1].plot(domain, train_log["accuracy"][start:])
axs[1].set(xlabel="Epoch", title="Accuracy", xticks=ticks)
axs[1].grid()

plt.show()

# on test set
fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
fig.suptitle("Evaluation on test set", y=0.99, fontsize="x-large")

axs[0].plot(domain, test_log["total_loss"][start:])
axs[0].set(xlabel="Epoch", title="Total loss", xticks=ticks)
axs[0].grid()

axs[1].plot(domain, test_log["accuracy"][start:])
axs[1].set(xlabel="Epoch", title="Accuracy", xticks=ticks)
axs[1].grid()

plt.show()

print(f"Maximal accuracy: {np.max(test_log['accuracy']):.2f}% (epoch {np.argmax(test_log['accuracy'])}).")
