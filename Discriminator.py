import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from weight_norm import weight_norm

class Discriminator(nn.Module):
    """Critic architecture from Salimans et al (2018) [see appendix B]."""
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
        x = self.conv1(x) # Shape : (batch_size, 64, 32, 32)
        # CReLU : https://github.com/pytorch/pytorch/issues/1327
        x = torch.cat((F.relu(x), F.relu(-x)), 1) # Shape : (batch_size, 128, 32, 32)
        x = self.conv2(x) # Shape : (batch_size, 128, 16, 16)
        x = torch.cat((F.relu(x), F.relu(-x)), 1) # Shape : (batch_size, 256, 16, 16)
        x = self.conv3(x) # Shape : (batch_size, 256, 8, 8)
        x = torch.cat((F.relu(x), F.relu(-x)), 1) # Shape : (batch_size, 512, 8, 8)
        x = nn.Flatten()(x) # Shape : (batch_size, 32768)
        x = F.normalize(x, dim=1, p=2) # Shape : (batch_size, 32768)

        return x # Shape : (batch_size, 32768)

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.relu = nn.ReLU()
#         self.conv1 = weight_norm(
#             nn.Conv2d(
#                 in_channels=1, out_channels=256, padding=2, kernel_size=5, stride=1
#             )
#         )

#         self.conv2 = weight_norm(
#             nn.Conv2d(
#                 in_channels=256, out_channels=512, kernel_size=5, padding=2, stride=2
#             )
#         )  #
#         self.conv3 = weight_norm(
#             nn.Conv2d(
#                 in_channels=512, out_channels=1024, kernel_size=5, padding=2, stride=2
#             )
#         )  #
#         self.conv4 = nn.Conv2d(
#             in_channels=1024, out_channels=2048, kernel_size=5, padding=2, stride=2
#         )  

#     def forward(self, x):
#         #TODO : add maxpool with relu
#         batch_size = x.size()[0]
#         x = self.conv1(x)
#         x = self.relu(x)
#         # x = torch.cat((self.relu(x), self.relu(-x)), dim=1)
#         x = self.conv2(x)
#         x = self.relu(x)
#         # x = torch.cat((self.relu(x), self.relu(-x)), dim=1)
#         x = self.conv3(x)
#         x = self.relu(x)

#         # x = torch.cat((self.relu(x), self.relu(-x)), dim=1)
#         x = self.conv4(x)
#         x = self.relu(x)
#         # x = torch.cat((self.relu(x), self.relu(-x)), dim=1)
#         x_reshaped = x.reshape(batch_size, 32768)
#         x_norm = (torch.norm(x_reshaped, p=2, dim=1)).view(batch_size, 1)
#         out = x_reshaped / (x_norm)
#         return out


