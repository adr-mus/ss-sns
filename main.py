import os.path

import torch

from data import SiameseNetworkDataset, show_pair
from metrics import ContrastiveLoss
from siamese import SiameseNetwork
from tools import track

device = "cuda" if torch.cuda.is_available() else "cpu"

@track
def init_dataset(*, n=18, k=4, p=1 / 36, show=True):
    """ n - the number of pairs to be chosen from each class
        k - the number of pairs to be chosen from each pair of different classes
        p - the size of the test set (0-1)
        The number of same-class pairs: 10*n, different-class pairs: 45*k. """
    dataset = SiameseNetworkDataset()
    dataset.init_traintest(n, k, p)
    if show:
        dataset.show_sample()

    trainset = torch.utils.data.DataLoader(dataset.trainset, batch_size=10)
    testset = torch.utils.data.DataLoader(dataset.testset, batch_size=1)

    return trainset, testset


@track
def init_network():
    net = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.RMSprop(
        net.parameters(),
        lr=1e-4,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0.0005,
        momentum=0.9,
    )

    return net, criterion, optimizer


@track
def train_network(net, trainset, criterion, optimizer, *, n_epochs=20, save=False):
    for _ in range(n_epochs):
        for images1, images2, labels in trainset:
            images1, images2, labels = images1.to(device), images2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs1, outputs2 = net(images1, images2)
            loss = criterion(outputs1, outputs2, labels)
            loss.backward()
            optimizer.step()

    if save:
        if "models" not in os.listdir():
            os.mkdir("models")
        torch.save(net.state_dict(), os.path.join("models", "last_model.pth"))


@track
def evaluate_network(net, testset, *, show=True):
    correct = 0
    with torch.no_grad():
        for image1, image2, label in testset:
            image1, image2, label = image1.to(device), image2.to(device), label.to(device)
            output1, output2 = net(image1, image2)
            pred = torch.max(torch.abs(output1 - output2), dim=1)[1]
            if show:
                show_pair(image1, image2, label, f"predicted {pred.item()}")
            correct += (label == pred).item()

    return correct / len(testset)


if __name__ == "__main__":
    dataset = init_dataset()
    trainset, testset = init_dataset(n=18, k=4, p=1 / 36, show=False)
    net, criterion, optimizer = init_network()
    train_network(net, trainset, criterion, optimizer)
    accuracy = evaluate_network(net, testset)
    print("Accuracy: {:.2%}".format(accuracy))
