import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from weight_norm import weight_norm

class Discriminator(nn.Module):
    def __init__(self, in_channel =3,kernel_size=5):
        super(Discriminator, self).__init__()
        conv1_channels = 64
        conv_padding_size = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(in_channel,conv1_channels , kernel_size,padding=conv_padding_size)
        self.conv2 = nn.Conv2d(conv1_channels*2, conv1_channels*2, kernel_size, 
                               stride=2, padding=conv_padding_size)
        self.conv3 = nn.Conv2d(conv1_channels*4, conv1_channels*4, kernel_size, 
                               stride=2, padding=conv_padding_size)

    def forward(self, x):
        x = self.conv1(x) #  (batch_size, 64, 32, 32)
        x = torch.cat((F.relu(x), F.relu(-x)), 1) # (batch_size, 128, 32, 32)
        x = self.conv2(x) # (batch_size, 128, 16, 16)
        x = torch.cat((F.relu(x), F.relu(-x)), 1) # (batch_size, 256, 16, 16)
        x = self.conv3(x) # (batch_size, 256, 8, 8)
        x = torch.cat((F.relu(x), F.relu(-x)), 1) # (batch_size, 512, 8, 8)
        x = nn.Flatten()(x) # (batch_size, 32768)
        x = F.normalize(x, dim=1, p=2) # (batch_size, 32768)
        return x # (batch_size, 32768)
