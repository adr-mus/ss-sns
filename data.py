import random, os, math

import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision


def show_image(image, label=None, text=""):
    image = np.transpose(image.cpu().numpy(), (1, 2, 0))
    if image.shape[2] == 1:
        image = image[:, :, 0]

    plt.figure(figsize=(4, 8))
    plt.imshow(image)
    title = "unlabeled" if label is None else str(label.item())
    if text:
        title += ", " + text
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_pair(image1, image2, label=None, text=""):
    image1, image2 = image1.cpu().numpy(), image2.cpu().numpy()
    joint_image = np.concatenate([image1, image2], axis=2)
    joint_image = np.transpose(joint_image, (1, 2, 0))
    if joint_image.shape[2] == 1:
        joint_image = joint_image[:, :, 0]

    plt.figure(figsize=(4, 8))
    plt.imshow(joint_image)
    title = "unlabeled" if label is None else str(label.item())
    if text:
        title += ", " + text
    plt.title(title)
    plt.axis("off")
    plt.show()


class DataPreparation:
    @staticmethod
    def prepare_MNIST(cls):
        """ Uses MNIST to create and save the following datasets:
                - classes is a list of form:
                    l[i] = images of the digit i from the MNIST train set,
                - oneshot is a list of form:
                    l[i] = an example image of the digit i. """
        if "data" not in os.listdir():
            os.mkdir("data")
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        trainset = torchvision.datasets.MNIST(os.path.join("data", "downloaded"), transform=transform, download=True)
        train_classes = [[] for _ in range(10)]
        for image, label in trainset:
            train_classes[label].append(image)
        oneshot = [images[-1] for images in train_classes]

        testset = torchvision.datasets.MNIST(os.path.join("data", "downloaded"), transform=transform, download=True, train=False)
        test_classes = [[] for _ in range(10)]
        for image, label in testset:
            test_classes[label].append(image)

        if "processed" not in os.listdir("data"):
            os.mkdir(os.path.join("data", "processed"))
        if cls.__name__ not in os.listdir(os.path.join("data", "processed")):
            os.mkdir(os.path.join("data", "processed", cls.__name__))

        torch.save(train_classes, os.path.join("data", "processed", cls.__name__, "train_classes.pth"))
        torch.save(test_classes, os.path.join("data", "processed", cls.__name__, "test_classes.pth"))
        torch.save(oneshot, os.path.join("data", "processed", cls.__name__, "oneshot.pth"))

        return train_classes, test_classes, oneshot
    
    @staticmethod
    def prepare_CIFAR10(cls):
        if "data" not in os.listdir():
            os.mkdir("data")
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(os.path.join("data", "downloaded"), transform=train_transform, download=True)
        train_classes = [[] for _ in range(10)]
        for image, label in trainset:
            train_classes[label].append(image)
        oneshot = [images[-1] for images in train_classes]

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(os.path.join("data", "downloaded"), transform=test_transform, download=True, train=False)
        test_classes = [[] for _ in range(10)]
        for image, label in testset:
            test_classes[label].append(image)

        if "processed" not in os.listdir("data"):
            os.mkdir(os.path.join("data", "processed"))
        if cls.__name__ not in os.listdir(os.path.join("data", "processed")):
            os.mkdir(os.path.join("data", "processed", cls.__name__))

        torch.save(train_classes, os.path.join("data", "processed", cls.__name__, "train_classes.pth"))
        torch.save(test_classes, os.path.join("data", "processed", cls.__name__, "test_classes.pth"))
        torch.save(oneshot, os.path.join("data", "processed", cls.__name__, "oneshot.pth"))

        return train_classes, test_classes, oneshot


class BaseDataset:
    labels = None

    @classmethod
    def prepare_data(cls):
        raise NotImplementedError

    def __init__(self):
        self._classes = None # dataset grouped by classes
        self._test_classes = None # dedicated test set grouped by classes
        self._oneshot = None # oneshot sample

        self._unlabeled = None # unlabeled pairs
        self._trainset = None # labeled pairs in training set
        self._testset = None # labeled pairs in test set

        if not os.path.exists(os.path.join("data", "processed", self.__class__.__name__)):
            self._classes, self._test_classes, self._oneshot = self.prepare_data()

    @property
    def classes(self):
        if self._classes is None:
            self.load_classes()
        return self._classes
    
    @property
    def test_classes(self):
        if self._test_classes is None:
            self.load_classes()
        return self._test_classes

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
    
    def sample_traintest(self, *args, **kwargs):
        raise NotImplementedError

    def save_traintest(self):
        dataset_name = self.__class__.__name__
        torch.save(self.trainset, os.path.join("data", "processed", dataset_name, "train_sample.pth"))
        torch.save(self.testset, os.path.join("data", "processed", dataset_name, "test_sample.pth"))

    def save_unlabeled(self):
        dataset_name = self.__class__.__name__
        torch.save(self.unlabeled, os.path.join("data", "processed", dataset_name, "unlabeled.pth"))

    def load_traintest(self):
        dataset_name = self.__class__.__name__
        try:
            self._trainset = torch.load(
                os.path.join("data", "processed", dataset_name, "train_sample.pth")
            )
            self._testset = torch.load(
                os.path.join("data", "processed", dataset_name, "test_sample.pth")
            )
            return self._trainset, self._testset
        except FileNotFoundError as e:
            e.args = (
                "Either trainset or testset cannot be loaded. Use sample_traintest and save_traintest first.",
            )
            raise

    def load_unlabeled(self):
        dataset_name = self.__class__.__name__
        try:
            self._unlabeled = torch.load(
                os.path.join("data", "processed", dataset_name, "unlabeled.pth")
            )
            return self._unlabeled
        except FileNotFoundError as e:
            e.args = (
                "unlabeled cannot be loaded. Use sample_unlabeled and save_unlabeled first.",
            )
            raise

    def load_classes(self):
        dataset_name = self.__class__.__name__
        try:
            self._classes = torch.load(os.path.join("data", "processed", dataset_name, "train_classes.pth"))
            self._test_classes = torch.load(os.path.join("data", "processed", dataset_name, "test_classes.pth"))
        except FileNotFoundError:
            self._classes, self._test_classes, self._oneshot = self.prepare_data()
        return self._classes

    def load_oneshot(self):
        dataset_name = self.__class__.__name__
        try:
            self._oneshot = torch.load(os.path.join("data", "processed", dataset_name, "oneshot.pth"))
            return self._oneshot
        except FileNotFoundError:
            self._classes, self._oneshot = self.prepare_data()
        return self._oneshot
    
    def show_oneshot(self):
        oneshot = [image.numpy().transpose((1, 2, 0)) for image in self.oneshot]
        row1 = np.concatenate(oneshot[:5], axis=1)
        row2 = np.concatenate(oneshot[5:], axis=1)
        joint_image = np.concatenate([row1, row2], axis=0)
        if joint_image.shape[2] == 1:
            joint_image = joint_image[:, :, 0]

        plt.figure(figsize=(8, 20))
        plt.imshow(joint_image)

        plt.title("One-shot sample")
        plt.axis("off")
        plt.show()


class SimpleDataset(BaseDataset):
    def sample_traintest(self, n=100, m=None, p=None, *, seed=131):
        """ Fills the lists self.trainset and self.testset with tuples of form 
                                (image, class),
            the list self.unlabeled with unlabeled images.
            
            Parameters:
            - n: the number of examples for each class,
            - m: the number of unlabeled examples for each class
                 (None: all available),
            - p: the size of the test set as a part of labaled examples (0.0-1.0)
                 (None: use the dedicated test set). """
        random.seed(seed)

        labeled = []
        unlabeled = []

        if m is not None:
            m += n

        for i in range(len(self.labels)):
            random.shuffle(self.classes[i])
            labels = [torch.tensor([i]) for _ in range(n)]
            labeled.extend(zip(self.classes[i][:n], labels))
            unlabeled.extend(self.classes[i][n:m])

        # balanced shuffle
        random.shuffle(labeled)
        random.shuffle(unlabeled)
        if p is not None:
            bp = round((1 - p) * len(labeled))
            self._testset = labeled[bp:]
            self._trainset = labeled[:bp]
        else:
            testset = []
            for i in range(len(self.labels)):
                cls = self.test_classes[i]
                labels = [torch.tensor([i]) for _ in range(len(cls))]
                testset.extend(zip(cls, labels))

            self._trainset = labeled
            self._testset = testset
        self._unlabeled = unlabeled

        return self.trainset, self.testset, self.unlabeled
    
    def show_trainset(self, n=5):
        """ Shows n pairs of images from the train set and the corresponding labels. """
        for i in range(n):
            show_image(*self.trainset[i])

    def show_testset(self, n=5):
        """ Shows n pairs of images from the test set and the corresponding labels. """
        for i in range(n):
            show_image(*self.testset[i])

    def show_unlabeled(self, n=5):
        """ Shows n pairs of unlabeled images. """
        for i in range(n):
            show_image(self.unlabeled[i])


class PseudolabelMNIST(SimpleDataset):
    labels = list("0123456789")

    prepare_data = classmethod(DataPreparation.prepare_MNIST)


class SiameseDataset(BaseDataset):
    _k, _l = None, None # balancing_numbers(len(labels))

    def __init__(self):
        BaseDataset.__init__(self)

        self._positive = None # positive pairs in training set
        self._negative = None # negative pairs in training set
    
    @property
    def positive(self):
        if self._positive is None:
            raise AttributeError("Use sample_traintest or load_traintest first.")
        return self._positive

    @property
    def negative(self):
        if self._negative is None:
            raise AttributeError("Use sample_traintest or load_traintest first.")
        return self._negative

    def sample_traintest(self, m=1, p=None, *, seed=131):
        """ Fills the lists self.trainset and self.testset with tuples of form 
                            (image1, image2, same class?)
            and returns them.
            
            Parameters:
            - m: the number of pairs equals 180*m (90*m positive ones, 90*m 
                 negative ones; all digits are equally represented),
            - p: the size of the test set (0.0-1.0). """
        random.seed(seed)

        # same-class pairs
        scp = []
        for i in range(len(self.labels)):
            images = random.sample(self.classes[i], 2 * m * self._k)
            for j in range(0, len(images), 2):
                label = torch.Tensor([1.0])
                scp.append((images[j], images[j + 1], label))

        # different-class pairs
        dcp = []
        for i in range(len(self.labels)):
            for j in range(i):
                images1 = random.sample(self.classes[i], m * self._l)
                images2 = random.sample(self.classes[j], m * self._l)
                labels = [torch.Tensor([0.0]) for _ in range(m * self._l)]
                dcp.extend(zip(images1, images2, labels))

        # balanced shuffle
        random.shuffle(scp)
        random.shuffle(dcp)
        data = scp + dcp
        l = len(data) // 2
        for i in range(0, l, 2):
            j = 2 * l - i - 1
            data[i], data[j] = data[j], data[i]

        if p is not None:
            bp = round((1 - p) * len(data))
            self._testset = data[bp:]
            self._trainset = data[:bp]
            self._positive = self._trainset[1::2]
            self._negative = self._trainset[::2]
        else:
            self._trainset = data
            self._positive = self._trainset[1::2]
            self._negative = self._trainset[::2]
            self._sample_testset(10, seed=seed)
        
    def _sample_testset(self, m, *, seed):
        random.seed(seed)

        # same-class pairs
        data = []
        for i in range(len(self.labels)):
            images = random.sample(self.test_classes[i], 2 * m * self._k)
            for j in range(0, len(images), 2):
                label = torch.Tensor([1.0])
                data.append((images[j], images[j + 1], label))

        # different-class pairs
        for i in range(len(self.labels)):
            for j in range(i):
                images1 = random.sample(self.test_classes[i], m * self._l)
                images2 = random.sample(self.test_classes[j], m * self._l)
                labels = [torch.Tensor([0.0]) for _ in range(m * self._l)]
                data.extend(zip(images1, images2, labels))

        self._testset = data

    def sample_unlabeled(self, m=10, *, seed=131):
        """ Fills the list self.unlabeled with tuples of form 
                            (image1, image2)
            and returns it.
            
            Parameters:
            - m: the number of unlabeled pairs is equal to 180 * m. """
        random.seed(seed)

        data = []

        # same-class pairs
        for i in range(len(self.labels)):
            images = random.sample(self.classes[i], 2 * m * self._k)
            for j in range(0, len(images), 2):
                data.append((images[j], images[j + 1]))

        # different-class pairs
        for i in range(len(self.labels)):
            for j in range(i):
                images1 = random.sample(self.classes[i], m * self._l)
                images2 = random.sample(self.classes[j], m * self._l)
                data.extend(zip(images1, images2))

        random.shuffle(data)

        self._unlabeled = data
    
    def show_trainset(self, n=5):
        """ Shows n pairs of images from the train set and the corresponding labels. """
        for i in range(n):
            show_pair(*self.trainset[i])

    def show_testset(self, n=5):
        """ Shows n pairs of images from the test set and the corresponding labels. """
        for i in range(n):
            show_pair(*self.testset[i])

    def show_unlabeled(self, n=5):
        """ Shows n pairs of unlabeled images. """
        for i in range(n):
            show_pair(*self.unlabeled[i])


class SiameseMNIST(SiameseDataset):
    labels = list("0123456789")
    _k, _l = 9, 2

    prepare_data = classmethod(DataPreparation.prepare_MNIST)


class SiameseCIFAR10(SiameseDataset):
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    _k, _l = 9, 2

    prepare_data = classmethod(DataPreparation.prepare_CIFAR10)



if __name__ == "__main__":
    dataset = PseudolabelMNIST()

    # load data or sample it if necessary
    try:
        dataset.load_traintest()
    except FileNotFoundError:
        dataset.sample_traintest()
        dataset.save_traintest()

    # try:
    #     dataset.load_unlabeled()
    # except FileNotFoundError:
    #     dataset.sample_unlabeled()
    #     dataset.save_unlabeled()

    print("Preview")
    dataset.show_oneshot()

    print("\nTraining set")
    dataset.show_trainset()

    print("\nTest set")
    dataset.show_testset()

    print("\nSample of unlabeled pairs")
    dataset.show_unlabeled()
