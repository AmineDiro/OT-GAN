
# The loss is a Mini  batch energy based distance
# It is a combination between a skinhorn distance and energy distance
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
from tqdm.auto import tqdm
from torchvision.transforms.functional import to_pil_image, resize, to_tensor, normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#################### Loading datasets ####################

class DoubleBatchDataset(data.Dataset):
    """ Extension for double batch, returns the  """

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, index):
        # return the tuple of data , ignores the taget
        return self.dataset1[index][0], self.dataset2[index][0]


def load_CIFAR10(batch_size=64, img_size=32):
    transform = transforms.Compose(
        [
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data_CIFAR10", train=True, download=True, transform=transform
    )

    cifar10 = DoubleBatchDataset(trainset, trainset)

    return torch.utils.data.DataLoader(cifar10, batch_size=batch_size, shuffle=True)


def load_MNIST(batch_size=64, img_size=32):
    transform = transforms.Compose(
        [
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    trainset = torchvision.datasets.MNIST(
        root="./data_mnist", train=True, download=True, transform=transform
    )

    mnist = DoubleBatchDataset(trainset, trainset)

    return torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)
