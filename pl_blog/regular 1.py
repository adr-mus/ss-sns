import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from pl_blog.mnist import PseudolabelCNN
from data import PseudolabelMNIST as Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_labeled():
    net.train()
    for tensors in trainset:
        images, labels = (t.to(device) for t in tensors)
        outputs = net(images)
        loss = criterion(outputs, labels.squeeze())

        train_log["total_loss"][-1] += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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
trainset = torch.utils.data.DataLoader(dataset.trainset, batch_size=batch_size, shuffle=True)
unlabeled = torch.utils.data.DataLoader(dataset.unlabeled, batch_size=batch_size, shuffle=True)
testset = torch.utils.data.DataLoader(dataset.testset, batch_size=len(dataset.testset))
print("Data ready.")

# initialize net and optimizer
net = PseudolabelCNN().to(device)
criterion = nn.NLLLoss()
S1, S2 = 100, 700
max_alpha = 3.0
lr = 0.01
momentum = 0.5
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
print("Net ready.")

# training parameters
E = 160  # epochs

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
for i in range(min(E, 100)):
    if i != 0 and i % 30 == 0:
        print(f"{i} epochs passed.")

    # logs
    train_log["total_loss"].append(0.0)
    test_log["total_loss"].append(0.0)

    train_labeled()
    test()

# stage 2: unlabeled + labeled for fine-tuning
step = 100
for i in range(100, E):
    if i != 0 and i % 30 == 0:
        print(f"{i} epochs passed.")

    # logs
    train_log["total_loss"].append(0.0)
    test_log["total_loss"].append(0.0)
    
    net.train()
    for batch_idx, images in enumerate(unlabeled):
        images = images.to(device)

        net.eval()
        outputs = net(images)
        pseudo_labels = torch.argmax(outputs, dim=1)

        net.train()          
        outputs = net(images)
        unlabeled_loss = alpha(step) * criterion(outputs, pseudo_labels)

        train_log["total_loss"][-1] += unlabeled_loss.item()
        
        optimizer.zero_grad()
        unlabeled_loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
            train_labeled()
            step += 1

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
