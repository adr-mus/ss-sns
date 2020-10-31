import random, os

import matplotlib.pyplot as plt
import torch, torchvision


def prepare_data():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = torchvision.datasets.MNIST("data", transform=transform, download=True)

    images = [[] for _ in range(10)]
    for image, label in dataset:
        images[label].append(image)
    torch.save(images, os.path.join("processed", "MNIST_train.pth"))

    return images


def show_pair(image1, image2, label, text=""):
    joint_image = torch.cat([image1.squeeze(), image2.squeeze()], dim=1)

    plt.figure(figsize=(4, 8))
    plt.imshow(joint_image.cpu().numpy(), cmap="gray")
    title = str(label.item())
    if text:
        title += ", " + text
    plt.title(title)
    plt.axis("off")
    plt.show()


class SiameseNetworkDataset:
    def __init__(self):
        if "processed" not in os.listdir():
            os.mkdir("processed")
        if "MNIST_train.pth" not in os.listdir("processed"):
            self.images = prepare_data()
        else:    
            self.images = torch.load(os.path.join("processed", "MNIST_train.pth"))
        
        self.trainset = None
        self.testset = None

    def init_traintest(self, n=18, k=4, p=1 / 36, *, seed=111):
        """ Fills the lists self.trainset and self.testset with tuples of form 
                            (image1, image2, sameclass?). 
            n - the number of pairs to be chosen from each class
            k - the number of pairs to be chosen from each pair of different classes
            p - the size of the test set (0-1)
            The number of same-class pairs: 10*n, different-class pairs: 45*k. """
        if seed is not None:
            random.seed(seed)

        data = []

        # same-class pairs
        for i in range(10):
            images = random.sample(self.images[i], 2 * n)
            for j in range(0, len(images), 2):
                label = torch.tensor([1])
                data.append((images[j], images[j + 1], label))

        # different-class pairs
        for i in range(10):
            for j in range(i):
                images1 = random.sample(self.images[i], k)
                images2 = random.sample(self.images[j], k)
                labels = [torch.tensor([0]) for _ in range(k)]
                data.extend(zip(images1, images2, labels))

        random.shuffle(data)

        bp = round(p * (10 * n + 45 * k))
        self.testset = data[:bp]
        self.trainset = data[bp:]

    def show_sample(self, n=4):
        """ Shows n pairs of images and the corresponding labels. """
        for i in range(n):
            show_pair(*self.trainset[i])
