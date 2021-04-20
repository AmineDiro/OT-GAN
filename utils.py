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
import ot
from torchvision.transforms.functional import to_pil_image, resize, to_tensor, normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################
def weights_init(m):
    """
    Taken from https://github.com/OctoberChang/MMD-GAN/blob/master/base_module.py
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


def sinkhorn_dist_log(X, Y, epsilon=0.1, max_iter=100, thresh=1e-1):
    C = 1 - X @ Y.T
    X_points = X.shape[-2]
    Y_points = Y.shape[-2]

    if X.dim() == 2:
        batch_size = 1
    else:
        batch_size = X.shape[0]

    # both marginals are fixed with equal weights
    # NOTE: change mu nu from marginal to dirac weighted sum
    # Maybe not ?
    mu = (
        torch.empty(batch_size, X_points, dtype=torch.float, requires_grad=False)
        .fill_(1.0 / X_points)
        .squeeze()
        .to(device)
    )
    nu = (
        torch.empty(batch_size, Y_points, dtype=torch.float, requires_grad=False)
        .fill_(1.0 / Y_points)
        .squeeze()
        .to(device)
    )

    u = torch.zeros_like(mu).to(device)
    v = torch.zeros_like(nu).to(device)

    actual_nits = 0

    with torch.no_grad():
        # Sinkhorn iterations , using logsumexp (fast)
        for i in range(max_iter):
            u1 = u  # useful to check the update
            u = (
                epsilon
                * (
                    torch.log(mu + 1e-8)
                    - torch.logsumexp(M_C(C, u, v, epsilon), dim=-1)
                )
                + u
            )
            v = (
                epsilon
                * (
                    torch.log(nu + 1e-8)
                    - torch.logsumexp(M_C(C, u, v, epsilon).transpose(-2, -1), dim=-1)
                )
                + v
            )
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break
        U, V = u, v
    # Transport plan (coupling) pi = diag(a)*K*diag(b) ()
    pi = torch.exp(M_C(C, U, V, epsilon))
    # Sinkhorn divergence
    divergence = torch.sum(pi * C, dim=(-2, -1))

    return divergence


def sinkhorn_loss_normal(X, Y, device=device, reg=1, maxiter=100):
    """
    Function which computes the autodiff sharp entropic OT loss.
    TODO :
    parameters:
        - a : input source measure (TorchTensor (ns))
        - b : input target measure (TorchTensor (nt))
        - C : cost between measure support (TorchTensor (ns, nt))
        - reg : entropic ragularization parameter (float)
        - maxiter : number of loop (int)

    returns:
        - sharp entropic unbalanced OT loss (float)
    """
    C = 1 - X @ Y.T

    batch_size_x = X.shape[0]
    batch_size_y = Y.shape[0]

    mu = (torch.ones(batch_size_x) / batch_size_x).to(torch.double).to(device)
    nu = (torch.ones(batch_size_y) / batch_size_y).to(torch.double).to(device)
    v = torch.ones_like(nu).to(device)
    with torch.no_grad():
        K = torch.exp(-C / reg).to(float).to(device)
        for i in range(maxiter):
            u = mu / (torch.matmul(K, v) + 1e-8)
            v = nu / (torch.matmul(K.T, u) + 1e-8)

    P = torch.matmul(torch.diag(u), torch.matmul(K, torch.diag(v)))
    return torch.sum(P * C)


def loss_func(X, Xprime, Y, Yprime, epsilon=0.1):
    loss_XY = sinkhorn_loss_normal(X, Y, reg=epsilon)
    loss_XYprime = sinkhorn_loss_normal(X, Yprime, reg=epsilon)
    loss_XprimeY = sinkhorn_loss_normal(Xprime, Y, reg=epsilon)
    loss_XprimeYprime = sinkhorn_loss_normal(Xprime, Yprime, reg=epsilon)

    loss_generator = loss_XY + loss_XYprime + loss_XprimeY + loss_XprimeYprime

    loss_XXprime = 2 * sinkhorn_loss_normal(X, Xprime, reg=epsilon)
    loss_YYprime = 2 * sinkhorn_loss_normal(Y, Yprime, reg=epsilon)

    loss_critic = loss_XXprime + loss_YYprime

    loss = loss_generator - loss_critic

    return loss, loss_generator, loss_critic


### Loading datasets
class DoubleBatchDataset(data.Dataset):
    """Extend Torch Dataset class to enable double batch streaming."""

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index]


def load_CIFAR10(batch_size=64, img_size=32):
    # TODO : Add transformatio normalize values in [-1,1]
    transform = transforms.Compose(
        [
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data_CIFAR10", train=True, download=False, transform=transform
    )

    cifar10 = DoubleBatchDataset(trainset, trainset)

    return torch.utils.data.DataLoader(cifar10, batch_size=batch_size, shuffle=True)


# def load_mnist(batch_size=64, img_size=32):
#     # TODO : Add transformatio normalize values in [-1,1]
#     transform = transforms.Compose([transforms.Resize(size=(32, 32)), transforms.ToTensor()])
#     trainset = torchvision.datasets.MNIST(root='./data_mnist', train=True,
#                                         download=False, transform=transform)

#     mnist = DoubleBatchDataset(trainset, trainset)

#     return torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)


def load_mnist(batch_size=64, img_size=32, double_batch=True):
    """Download, preprocess and load MNIST data."""
    mnist = datasets.MNIST("data_mnist", train=True, download=True).data
    # Perform transformation directly on raw data rather than in the DataLoader
    # => avoids overhead of performing transforms at each batch call
    # => much faster epochs.
    pics = []
    for pic in mnist:
        pic = to_pil_image(pic)
        if img_size != 28:
            pic = resize(pic, img_size)  # Resize image if needed
        pic = to_tensor(pic)  # Tensor conversion normalizes in [0,1]
        pic = normalize(pic, [0.5], [0.5])  # Normalize values in [-1,1]
        pics.append(pic)

    mnist = torch.stack(pics)

    if double_batch:
        mnist = DoubleBatchDataset(mnist, mnist)

    return torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)


def sample_data(n_sample, gmodel):
    """Generate synthetic images from random uniform noise."""
    z_random = (-2 * torch.rand((n_sample, 100)) + 1).to(device)
    samples = gmodel(z_random)
    samples = samples.detach().cpu().numpy()
    return samples