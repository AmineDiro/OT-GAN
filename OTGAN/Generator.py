import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_dim=100, out_channels=3):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(in_features=input_dim, out_features=32768)
        # need to define out features = 2*16384 because glu devides by 2 the number of units
        self.conv1 = nn.Conv2d(
                in_channels=2 * 512,
                out_channels=2 * 512,
                kernel_size=5,
                padding=2,
                stride=1,
            )
        
        self.conv2 = nn.Conv2d(
                in_channels=2 * 256,
                out_channels=2 * 256,
                kernel_size=5,
                padding=2,
                stride=1,
            )
        
        self.conv3 = nn.Conv2d(
                in_channels=2 * 128,
                out_channels=2 * 128,
                kernel_size=5,
                padding=2,
                stride=1,
            )
        
        self.last_conv = nn.Conv2d(
                in_channels=128,
                out_channels=out_channels,
                kernel_size=5,
                padding=2,
                stride=1,
            )
        
        self.activ_out = nn.Tanh()
    
    def generate_noise(self, batch_size):
        z = (torch.rand([batch_size, self.input_dim], requires_grad=True) * -2) + 1
        return z

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.linear(x)
        x = F.glu(x)
        x = x.reshape(batch_size, 1024, 4, 4)
        # 2x2 nearest neighbour upsampling
        x = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
        x = F.glu(x, dim=1)
        x = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
        x = self.conv2(x)
        x = F.glu(x, dim=1)
        x = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
        x = F.glu(x, dim=1)
        x = self.last_conv(x)
        x = self.activ_out(x)
        return x

