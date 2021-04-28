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


#################### SINKHORN #########################

def sinkhorn_dist_log(X, Y, epsilon=0.1, max_iter=100, thresh=1e-1):
    C = 1 - X @ Y.T
    X_points = X.shape[-2]
    Y_points = Y.shape[-2]

    if X.dim() == 2:
        batch_size = 1
    else:
        batch_size = X.shape[0]

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
