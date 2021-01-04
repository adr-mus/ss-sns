import itertools as it

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pl_blog.mnist import PseudolabelSNN
from data import SiameseMNIST as Dataset
from criterions import ContrastiveLoss

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(S2, max_alpha, dt):
    def alpha(step):
        if step < S1:
            return 0
        elif step < S2:
            return (step - S1) / (S2 - S1) * max_alpha
        else:
            return max_alpha

    def test():
        net.eval()
        with torch.no_grad():
            # evaluation on test set
            correct = 0
            for tensors in test_loader:
                images1, images2, labels = (t.to(device) for t in tensors)
                outputs1, outputs2 = net(images1, images2)

                dists = F.pairwise_distance(outputs1, outputs2)
                preds = (dists <= T).float()
                correct += torch.isclose(labels.squeeze(), preds).sum().item()
            accuracy = 100 * correct / len(dataset.testset)
            test_log["accuracy"].append(accuracy)

    # initialize net and optimizer
    net = PseudolabelSNN().to(device)
    net.load_state_dict(net.state_dict())
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    test_log = {
        "accuracy": [],
    }

    t = S1
    for _ in range(S1, E):
        net.eval()
        with torch.no_grad():
            pseudo_labels = []
            for images1, images2 in unlabeled_loader:
                images1, images2 = images1.to(device), images2.to(device)
                outputs1, outputs2 = net(images1, images2)
                preds = (F.pairwise_distance(outputs1, outputs2) <= T).float()
                pseudo_labels.append(preds)
                
        net.train()
        count = 0
        for (images1, images2, labels), (images3, images4), p_labels in zip(it.cycle(train_loader), unlabeled_loader, pseudo_labels):
            images1, images2, labels = images1.to(device), images2.to(device), labels.to(device)
            images3, images4 = images3.to(device), images4.to(device)
            
            outputs1, outputs2 = net(images1, images2)
            outputs3, outputs4 = net(images3, images4)
            loss = criterion(outputs1, outputs2, labels.squeeze()) + alpha(t) * criterion(outputs3, outputs4, p_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            count += 1
            if count % len(train_loader) == 0:
                t += dt
                count = 0

        test()
    
    return np.max(test_log["accuracy"])


# load data and prepare it
dataset = Dataset()
dataset.sample_traintest()
dataset.sample_unlabeled(100)

batch_size = 60
train_loader = torch.utils.data.DataLoader(dataset.trainset, batch_size=batch_size)
unlabeled_loader = torch.utils.data.DataLoader(dataset.unlabeled, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset.testset, batch_size=len(dataset.testset))
print("Data loaded and ready")

base_net = PseudolabelSNN().to(device)
criterion = ContrastiveLoss(margin=1.0)
T = 0.6
lr = 0.001
momentum = 0.9
optimizer = torch.optim.SGD(base_net.parameters(), lr=lr, momentum=momentum)
E = 100  # epochs
S1 = 50

for _ in range(min(E, S1)):
    base_net.train()
    for tensors in train_loader:
        images1, images2, labels = (t.to(device) for t in tensors)
        outputs1, outputs2 = base_net(images1, images2)
        loss = criterion(outputs1, outputs2, labels.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print("Model pretrained. Starting grid search")

# grid
grid = {"S2": [75, 100, 125, 150], 
        "max_alpha": [0.3, 0.4, 0.5, 0.6, 0.7], 
        "dt": [0.001, 0.002, 0.005, 0.01, 0.02]}

results = {}

for S2 in grid["S2"]:
    for max_alpha in grid["max_alpha"]:
        for dt in grid["dt"]:
            print(S2, max_alpha, dt)
            best_acc = train_model(S2, max_alpha, dt)
            results[S2, max_alpha, dt] = best_acc

sorted_results = sorted(results.items(), reverse=True, key=lambda el: el[1])

print(*sorted_results[:20], sep="\n")
