import random, os

import matplotlib.pyplot as plt
import torch, torchvision


def prepare_data():
    """ Uses MNIST to create and save the following datasets:
            - MNIST_classes is a list of form:
                l[i] = images of the digit i from the MNIST train set,
            - MNIST_oneshot is a list of form:
                l[i] = an example image of the digit i. """
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
    oneshot = [digits[-1] for digits in images]

    if "processed" not in os.listdir():
        os.mkdir("processed")
    torch.save(images, os.path.join("processed", "MNIST_classes.pth"))
    torch.save(oneshot, os.path.join("processed", "MNIST_oneshot.pth"))

    return images, oneshot


def show_pair(image1, image2, label=None, text=""):
    joint_image = torch.cat([image1.squeeze(), image2.squeeze()], dim=1)

    plt.figure(figsize=(4, 8))
    plt.imshow(joint_image.cpu().numpy(), cmap="gray")
    title = "unlabeled" if label is None else str(label.item())
    if text:
        title += ", " + text
    plt.title(title)
    plt.axis("off")
    plt.show()


class SiameseMNIST:
    def __init__(self):
        self._images = None
        self._oneshot = None
        self._unlabeled = None
        self._trainset = None
        self._testset = None

        if "processed" not in os.listdir():
            self._images, self._oneshot = prepare_data()
        
    @property
    def images(self):
        if self._images is None:
            self.load_images()
        return self._images
    
    @property
    def oneshot(self):
        if self._oneshot is None:
            self.load_oneshot()
        return self._oneshot
    
    @property
    def unlabeled(self):
        if self._unlabeled is None:
            raise AttributeError("Use sample_unlabeled or load_unlabeled first.")
        return self._unlabeled

    @property
    def trainset(self):
        if self._trainset is None:
            raise AttributeError("Use sample_traintest or load_traintest first.")
        return self._trainset
    
    @property
    def testset(self):
        if self._testset is None:
            raise AttributeError("Use sample_traintest or load_traintest first.")
        return self._testset

    def sample_traintest(self, n=9, k=2, p=1 / 6, *, seed=111):
        """ Fills the lists self.trainset and self.testset with tuples of form 
                            (image1, image2, same class?)
            and returns them.
            
            Parameters:
            - n: the number of pairs to be chosen for each class,
            - k: the number of pairs to be chosen for each pair of different classes,
            - p: the size of the test set (0.0-1.0).

            The number of same-class pairs: 10*n, different-class pairs: 45*k.
            When n/k = 9/2, there are as many positive examples as negative ones. """
        if self._images is None:
            self.load_images()
        
        random.seed(seed)

        # same-class pairs
        scp = []
        for i in range(10):
            images = random.sample(self.images[i], 2 * n)
            for j in range(0, len(images), 2):
                label = torch.Tensor([1.0])
                scp.append((images[j], images[j + 1], label))

        # different-class pairs
        dcp = []
        for i in range(10):
            for j in range(i):
                images1 = random.sample(self.images[i], k)
                images2 = random.sample(self.images[j], k)
                labels = [torch.Tensor([0.0]) for _ in range(k)]
                dcp.extend(zip(images1, images2, labels))

        # controlled shuffle
        random.shuffle(scp)
        random.shuffle(dcp)
        data = scp + dcp
        l = len(data) // 2
        for i in range(0, l, 2):
            j = 2 * l - i - 1
            data[i], data[j] = data[j], data[i]

        bp = round(p * len(data))
        self._testset = data[:bp]
        self._trainset = data[bp:]

        return self._trainset, self._testset
    
    def sample_unlabeled(self, m=1, *, seed=111):
        """ Fills the list self.unlabeled with tuples of form 
                            (image1, image2)
            and returns it.
            
            Parameters:
            - m: the number of unlabeled pairs is equal to 180 * m. """
        if self._images is None:
            self.load_images()

        random.seed(seed)

        data = []

        # same-class pairs
        for i in range(10):
            images = random.sample(self.images[i], 18 * m)
            for j in range(0, len(images), 2):
                data.append((images[j], images[j + 1]))

        # different-class pairs
        for i in range(10):
            for j in range(i):
                images1 = random.sample(self.images[i], 2 * m)
                images2 = random.sample(self.images[j], 2 * m)
                data.extend(zip(images1, images2))

        random.shuffle(data)

        self._unlabeled = data

        return self._unlabeled
    
    def save_traintest(self):
        torch.save(self.trainset, os.path.join("processed", "MNIST_train_sample.pth"))
        torch.save(self.testset, os.path.join("processed", "MNIST_test_sample.pth"))
    
    def save_unlabeled(self):
        torch.save(self.unlabeled, os.path.join("processed", "MNIST_unlabeled.pth"))
    
    def load_traintest(self):
        try:
            self._trainset = torch.load(os.path.join("processed", "MNIST_train_sample.pth"))
            self._testset = torch.load(os.path.join("processed", "MNIST_test_sample.pth"))
            return self._trainset, self._testset
        except FileNotFoundError as e:
            e.args = ("Either trainset or testset cannot be loaded. Use sample_traintest and save_traintest first.",)
            raise
    
    def load_unlabeled(self):
        try:
            self._unlabeled = torch.load(os.path.join("processed", "MNIST_unlabeled.pth"))
            return self._unlabeled
        except FileNotFoundError as e:
            e.args = ("unlabeled cannot be loaded. Use sample_unlabeled and save_unlabeled first.",)
            raise
    
    def load_images(self):
        try:
            self._images = torch.load(os.path.join("processed", "MNIST_classes.pth"))
        except FileNotFoundError:
            self._images, self._oneshot = prepare_data()
        return self._images

    def load_oneshot(self):
        try:
            self._oneshot = torch.load(os.path.join("processed", "MNIST_oneshot.pth"))
            return self._oneshot
        except FileNotFoundError:
            self._images, self._oneshot = prepare_data()
        return self._oneshot

    def show_train(self, n=5):
        """ Shows n pairs of images from the train set and the corresponding labels. """
        for i in range(n):
            show_pair(*self.trainset[i])
    
    def show_test(self, n=5):
        """ Shows n pairs of images from the test set and the corresponding labels. """
        for i in range(n):
            show_pair(*self.testset[i])

    def show_oneshot(self):
        row1 = torch.cat(self.oneshot[:5], dim=2).squeeze()
        row2 = torch.cat(self.oneshot[5:], dim=2).squeeze()
        joint_image = torch.cat([row1, row2], dim=0)

        plt.figure(figsize=(8, 20))
        plt.imshow(joint_image.cpu().numpy(), cmap="gray")

        plt.title("One-shot sample")
        plt.axis("off")
        plt.show()
    
    def show_unlabeled(self, n=5):
        """ Shows n pairs of unlabeled images. """
        for i in range(n):
            show_pair(*self.unlabeled[i])



if __name__ == "__main__":
    dataset = SiameseMNIST()

    # load data or sample it if necessary
    try:
        dataset.load_traintest()
    except FileNotFoundError:
        dataset.sample_traintest()
        dataset.save_traintest()
        
    try:
        dataset.load_unlabeled()
    except FileNotFoundError:
        dataset.sample_unlabeled()
        dataset.save_unlabeled()

    print("Preview")
    dataset.show_oneshot()

    print("\nTraining set")
    dataset.show_train()

    print("\nTest set")
    dataset.show_test()

    print("\nSample of unlabeled pairs")
    dataset.show_unlabeled()
