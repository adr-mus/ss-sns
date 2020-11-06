import torch
import torch.nn.functional as F

from torch import nn

from metrics import ClassificationLoss, ReconstructionLoss


class DiscriminativeSNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # convolutional layers
        # input: 1 @ 28x28
        self.cnn = nn.Sequential(
            # first
            nn.Conv2d(1, 8, kernel_size=3, stride=1), # 8 @ 26x26
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # 8 @ 13x13 
            nn.Dropout2d(p=0.5),
            # second
            nn.Conv2d(8, 8, kernel_size=4, stride=1), # 8 @ 10x10
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # 8 @ 5x5
            nn.Dropout2d(p=0.5),
            nn.Flatten()
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
            nn.Upsample(scale_factor=2), # 8 @ 10x10
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=1), # 8 @ 13x13
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            # second
            nn.Upsample(scale_factor=2), # 8 @ 26x26
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1), # 1 @ 28x28
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


if __name__ == "__main__":
    import os, itertools

    from data import SiameseMNIST, show_pair


    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data and prepare it
    dataset = SiameseMNIST()
    # dataset.sample_traintest()
    # dataset.sample_unlabeled()
    dataset.load_traintest()
    dataset.load_unlabeled()

    trainset = torch.utils.data.DataLoader(dataset.trainset, batch_size=32)
    unlabeled = torch.utils.data.DataLoader(dataset.unlabeled, batch_size=128)
    testset = torch.utils.data.DataLoader(dataset.testset, batch_size=1)
    print("Data ready.")

    # initialize nets and optimizers
    net1, net2 = DiscriminativeSNN().to(device), GenerativeSNN().to(device)
    criterion1, criterion2 = ClassificationLoss(), ReconstructionLoss()
    alpha = 0.5 # importance of reconstruction loss
    beta = 0.001 # l2 regularization
    lr = 0.001
    optimizer = torch.optim.RMSprop(itertools.chain(net1.parameters(), net2.parameters()), lr=lr, weight_decay=beta)
    print("Nets ready. Beginning training.")

    # training
    for i in range(150): # the number of epochs
        if i != 0 and i % 15 == 0:
            print(f"{i} epochs passed.")

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

    # save model
    # if "models" not in os.listdir():
    #     os.mkdir("models")
    # torch.save(net1.state_dict(), os.path.join("models", "SEVEN_discriminative.pth"))
    # torch.save(net2.state_dict(), os.path.join("models", "SEVEN_generative.pth"))
    # print("Model saved.")

    # evaluation
    T = 0.5 # threshold
    with torch.no_grad():
        correct = 0
        for tensors in testset: # one-element batches
            image1, image2, label = (t.to(device) for t in tensors)
            output1, output2 = net1(image1, image2)
            dist = F.pairwise_distance(output1, output2)
            pred = (dist < T).float()
            # show_pair(image1, image2, label, "predicted {} ({:.4f})".format(pred.item(), dist.item()))
            correct += (label == pred).item()
    accuracy = correct / len(testset)
    print("Evaluation finished. Accuracy: {:.2%}.".format(accuracy))
