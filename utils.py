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



def sample_data(n_sample, gmodel):
    """Generate synthetic images from random uniform noise."""
    z_random = (-2 * torch.rand((n_sample, 100)) + 1).to(device)
    samples = gmodel(z_random)
    # samples = samples.detach().cpu().numpy()
    return samples.detach()

def plot_losses(loss):
    G_loss, D_loss = loss
    fig, ax = plt.subplots()
    
