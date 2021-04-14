# The loss is a Mini  batch energy based distance
# It is a combination between a skinhorn distance and energy distance
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#


# def cos_dist(X,Y):
#     X_norm = torch.norm(X, dim=1)
#     Y_norm = torch.norm(Y, dim=1)
#     cos_dist = 1- torch.mm(X_norm, torch.t(Y_norm))
#     return cos_dist

# def sinkhorn_dist(C, a,b, epsilon=1, max_iters=100):
#     u = torch.ones_like(a).to(device)
#     v = torch.ones_like(b).to(device)

#     with torch.no_grad():
#         K = torch.exp(-C/epsilon)
#         for i in range(max_iters):
#             u = a / (torch.matmul(K,v))
#             v = b / (torch.matmul(K.T,u))
#     M = torch.matmul(torch.diag(u), torch.matmul(K, torch.diag(v)))
#     return M

# TODO : move to model
def weights_init(w):
    """
    Initi weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find("conv") != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find("bn") != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

def cos_dist(X, Y):
    C = torch.zeros((X.size(0), Y.size(0))).to(device)
    for i in range(X.size(0)):
        for j in range(Y.size(0)):
            C[i, j] = F.cosine_similarity(X[i], Y[j], dim=0)
    return C



def M_C(C, u, v, eps=0.1):
    "Modified cost for logarithmic updates"
    "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
    return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / eps


def sinkhorn_dist(X, Y, epsilon=0.1, max_iter=100, thresh=1e-1):
    C = cos_dist(X, Y)  # Special cost function
    X_points = X.shape[-2]
    Y_points = Y.shape[-2]

    if X.dim() == 2:
        batch_size = 1
    else:
        batch_size = X.shape[0]

    # both marginals are fixed with equal weights
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

    # Sinkhorn iterations , using logsumexp (fast)
    for i in range(max_iter):
        u1 = u  # useful to check the update
        u = (
            epsilon
            * (torch.log(mu + 1e-8) - torch.logsumexp(M_C(C, u, v, epsilon), dim=-1))
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
    # Transport plan (coupling) pi = diag(a)*K*diag(b)
    pi = torch.exp(M_C(C, U, V, epsilon))
    # Sinkhorn distance
    cost = torch.sum(pi * C, dim=(-2, -1))
    print(cost)
    return cost


def loss_cos_dist(X, Xprime, Y, Yprime, epsilon=0.1):
    loss_XY = sinkhorn_dist(X, Y, epsilon=epsilon)
    loss_XYprime = sinkhorn_dist(X, Yprime, epsilon=epsilon)
    loss_XprimeY = sinkhorn_dist(Xprime, Y, epsilon=epsilon)
    loss_XprimeYprime = sinkhorn_dist(Xprime, Yprime, epsilon=epsilon)

    loss_generator = loss_XY + loss_XYprime + loss_XprimeY + loss_XprimeYprime

    loss_XXprime = 2 * sinkhorn_dist(X, Xprime, epsilon=epsilon)
    loss_YYprime = 2 * sinkhorn_dist(Y, Yprime, epsilon=epsilon)

    loss_critic = loss_XXprime + loss_YYprime

    loss = loss_generator - loss_critic

    return loss, loss_generator, loss_critic
