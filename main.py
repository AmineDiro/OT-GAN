import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, Subset
from torchvision.utils import save_image
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim

import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm.auto import tqdm
import ot

from Generator import Generator
from Discriminator import Discriminator


from utils import *


def main():
    batch_size = 64  # selon les auteurs mais on va pas utiliser ça
    ginput_dim = 100

    # val_batch_size=128
    n_epochs = 100
    lr = 3e-4
    beta1 = 0.5
    beta2 = 0.999
    epsilon = 1
    n_gen = 3  # nupmber of generator updates / discriminator update

    output_path_imgs = "./generated_image"
    save_epoch = 2
    sample_interval = 1

    use_cuda = True
    #  Init device
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # dataloader = load_mnist()

    dataloader = load_mnist()
    # Create Generator
    gmodel = Generator(out_channels=1)
    dmodel = Discriminator(in_channel=1)

    # dmodel.apply(weights_init)
    # gmodel.apply(weights_init)

    # Load last trained model
    # saved_weights_file = glob.glob('model/*')
    # Get latest saved weights
    # model_path = max(saved_weights_file, key=os.path.getctime)
    # state_dict = torch.load(model_path,map_location='cpu')
    # gmodel.load_state_dict(state_dict['generator'])
    gmodel.to(device)

    # dmodel.load_state_dict(state_dict['critic'])
    dmodel.to(device)

    # Get optimizer
    optimizer_D = optim.Adam(dmodel.parameters(), lr=lr, betas=[beta1, beta2])
    optimizer_G = optim.Adam(gmodel.parameters(), lr=lr, betas=[beta1, beta2])

    mone = -torch.tensor(1, dtype=torch.float).to(device)

    for epoch in range(n_epochs):
        desc = "[Epoch {}]".format(epoch)
        torch.cuda.empty_cache()
        tbatch = tqdm(enumerate(dataloader), desc=desc)
        #for i, ((im1,_), (im2,_)) in tepoch:
        for i, (im1, im2) in tbatch:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            # Real images batch to GPU
            im1, im2 = im1.to(device), im2.to(device)
            # Generate batch noise
            noise = ((torch.rand([batch_size, ginput_dim]) * -2) + 1).to(device)
            prime_noise = ((torch.rand([batch_size, ginput_dim]) * -2) + 1).to(device)
            # Generated images :
            # TODO : add require grad
            gen_img =gmodel(noise)
            gen_img_prime = gmodel(prime_noise)

            # Encoded Generated images
            Y, Yprime = dmodel(gen_img), dmodel(gen_img_prime)

            # Encoded Real images
            X, Xprime = dmodel(im1), dmodel(im2)

            loss, loss_generator, loss_critic = loss_func(X, Xprime, Y, Yprime)

            loss.backward()
            # optimizer_G.step()

            if i + 1 % n_gen == 0:
                # Update critic once every 3 generator updates
                # loss.backward(mone)
                optimizer_D.step()
            else:
                # loss.backward()
                optimizer_G.step()
            
            # description tqdm
            tbatch.set_postfix(
                    loss_generator=loss_generator.item(), loss_critic=loss_critic.item()
                )

            
        # if epoch%sample_interval==0:
            save_image(
                gen_img.data[:64],
                os.path.join(output_path_imgs, "epoch_{}.png".format(epoch)),
                nrow=8,
                normalize=True,
            )


        if epoch % save_epoch == 0:
            torch.save(
                {
                    "generator": gmodel.state_dict(),
                    "critic": dmodel.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizerD": optimizer_D.state_dict(),
                },
                "model/model_mnist.pth",
            )

        # if epoch % 1 == 0 or epoch + 1 == epochs:
        #     samples = sample_data(3, gmodel) * 0.5 + 0.5
        #     for img in samples:
        #         plt.figure()
        #         plt.imshow(img[0, :, :], cmap="gray")
        #         plt.show()

if __name__ == "__main__":
    main()