import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

# Weight initialisation
from weight_norm import weight_norm

# class Generator(nn.Module):
#     """Generator architecture from Salimans et al (2018) [see appendix B]."""
#     def __init__(self,out_channels =3, input_size=100, kernel_size=5):
#         super(Generator, self).__init__()

#         self.linear = nn.Linear(input_size, 2*512*8*8)
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

#         conv_padding_size = (kernel_size - 1) // 2

#         self.conv1 = nn.Conv2d(512, 2*256, kernel_size, padding=conv_padding_size)
#         self.conv2 = nn.Conv2d(256, 2*128, kernel_size, padding=conv_padding_size)
#         self.conv3 = nn.Conv2d(128, out_channels, kernel_size, padding=conv_padding_size)

#         self.activ_out = nn.Tanh()

#     def forward(self, x):

#         x = self.linear(x) # Shape : (n_batch, 65536)
#         # GLU activation function
#         x, l = torch.split(x, x.shape[1]//2, 1)
#         x *= torch.sigmoid(l) # Shape : (n_batch, 32768)
#         x = x.view((x.shape[0], 512, 8, 8)) # Shape : (n_batch, 512, 8, 8)

#         x = self.upsample(x) # Shape : (n_batch, 512, 16, 16)
#         x = self.conv1(x) # Shape : (n_batch, 512, 16, 16)
#         x, l = torch.split(x, x.shape[1]//2, 1)
#         x *= torch.sigmoid(l) # Shape : (n_batch, 256, 16, 16)

#         x = self.upsample(x) # Shape : (n_batch, 256, 32, 32)
#         x = self.conv2(x) # Shape : (n_batch, 256, 32, 32)
#         x, l = torch.split(x, x.shape[1]//2, 1)
#         x *= torch.sigmoid(l) # Shape : (n_batch, 128, 32, 32)

#         x = self.activ_out(self.conv3(x)) # Shape : (n_batch, 1, 32, 32)

#         return x # Shape : (batch_size, 1, 32, 32)

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

