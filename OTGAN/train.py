import os
import glob
import argparse
import pickle
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
from tqdm import tqdm

from OTGAN.SinkhornDistance import loss_func
from OTGAN.dataset import load_MNIST, load_CIFAR10
from OTGAN.Generator import Generator
from OTGAN.Discriminator import Discriminator

from torchvision.models.inception import inception_v3
from OTGAN.InceptionScore import inception_score_training

from OTGAN.utils import *


def train_and_evaluate(
    channels,
    batch_size,
    INCEPTION_SCORE,
    save_epoch,
    sample_interval,
    LOAD_MODEL=False,
    WEIGHT_INIT=False,
    use_cuda=True,
):
    # Hyperparameter
    # batch_size = batch_size
    ginput_dim = 100
    n_epochs = 30
    lr = 3e-4
    beta1 = 0.5
    beta2 = 0.999
    epsilon = 1
    n_gen = 3  # number of generator updates / discriminator update
    N_inception = 100  # Number of batches
    if channels == 1:
        output_path_imgs = "./generated_image_MNIST"
        output_path_model = './model_MNIST'
        if not os.path.exists(output_path_model):
            os.makedirs(output_path_model)

    else:
        output_path_imgs = "./generated_image_CIFAR"
        output_path_model = './model_CIFAR'
        if not os.path.exists(output_path_model):
            os.makedirs(output_path_model)
    
    save_epoch = 2
    sample_interval = 1

    ## Save losses
    G_loss = []
    D_loss = []

    # Inception Score while training
    imgs =[]
    icp = []

    #  Init device
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    #  Load Dataset
    if channels == 1:
        dataloader = load_MNIST()
        print("Loaded MNIST Dataset ...")
    else:
        dataloader = load_CIFAR10()
        print("Loaded CIFAR-10 Dataset ...")

    # Create Generator and Critic
    gmodel = Generator(out_channels=channels)
    dmodel = Discriminator(in_channel=channels)

    # Load inception model,pretrained for the inception score, no inception score for mnist
    if INCEPTION_SCORE and channels != 1:
        print("Loading Inception model ...")
        inception_model = inception_v3(pretrained=True, transform_input=False).to(
            device
        )
        inception_model.eval()

    # Weight init
    if WEIGHT_INIT:
        dmodel.apply(weights_init)
        gmodel.apply(weights_init)

    # Load last trained model
    if LOAD_MODEL:
        print("Loading last saved model...")
        saved_weights_file = glob.glob("model/*")
        # Get latest saved weights
        model_path = max(saved_weights_file, key=os.path.getctime)
        state_dict = torch.load(model_path, map_location="cpu")
        gmodel.load_state_dict(state_dict["generator"])
        dmodel.load_state_dict(state_dict["critic"])

    # Send models to Device
    gmodel.to(device)
    dmodel.to(device)

    # Optimizer
    optimizer_D = optim.Adam(dmodel.parameters(), lr=lr, betas=[beta1, beta2])
    optimizer_G = optim.Adam(gmodel.parameters(), lr=lr, betas=[beta1, beta2])

    # Minus one variable used for gradient ascent of critic
    mone = -torch.tensor(1, dtype=torch.float).to(device)


    for epoch in range(n_epochs):
        desc = "[Epoch {}]".format(epoch)
        torch.cuda.empty_cache()
        #tbatch = tqdm(numerate(dataloader), desc=desc)
        total = len(dataloader)
        G_epoch_loss , D_epoch_loss = 0, 0

        with tqdm(total=total, desc=desc) as tbatch:
            for i, (im1, im2) in enumerate(dataloader):
                if i ==102 :
                    break
                optimizer_D.zero_grad()
                optimizer_G.zero_grad()
                # Real images batch to GPU
                im1, im2 = im1.to(device), im2.to(device)
                # Generate batch noise
                noise = ((torch.rand([batch_size, ginput_dim]) * -2) + 1).to(device)
                prime_noise = ((torch.rand([batch_size, ginput_dim]) * -2) + 1).to(device)
                # Generated images :
                gen_img = gmodel(noise)
                gen_img_prime = gmodel(prime_noise)

                # Encoded Generated images
                Y, Yprime = dmodel(gen_img), dmodel(gen_img_prime)

                # Encoded Real images
                X, Xprime = dmodel(im1), dmodel(im2)

                loss, loss_generator, loss_critic = loss_func(X, Xprime, Y, Yprime)

                G_epoch_loss += loss_generator.detach().cpu().numpy()
                D_epoch_loss += loss_critic.detach().cpu().numpy()

                # Update critic once every "n_gen" generator updates
                if i + 1 % n_gen == 0:
                    loss.backward(mone)
                    optimizer_D.step()
                else:
                    loss.backward()
                    optimizer_G.step()

                # Save generated images for inception score
                if len(imgs) < N_inception:
                    imgs.append(gen_img.detach())

                # Modify tqdm description
                tbatch.set_postfix(
                    loss_generator=loss_generator.item(), loss_critic=loss_critic.item()
                )
                tbatch.update(1)
        
        # Save losses
        G_loss.append(G_epoch_loss)
        D_loss.append(D_epoch_loss)


        if INCEPTION_SCORE and channels != 1 :
            score = inception_score_training(
                imgs,
                inception_model,
                gmodel,
                N=N_inception,
                batch_size=batch_size,
                resize=True,
            )
            print("\n [EPOCH {}] [Inception score : {}]".format(epoch, score))
            icp.append(score)
            imgs = []


        if epoch % sample_interval == 0:
            samples = sample_data(batch_size, gmodel) * 0.5 + 0.5
            # samples = sample_data(batch_size, gmodel)
            save_image(
                samples.data[:64],
                os.path.join(output_path_imgs, "epoch_{}.png".format(epoch)),
                nrow=8,
                normalize=True,
            )


        if epoch % save_epoch == 0:
            # Save interval
            torch.save(
                {
                    "generator": gmodel.state_dict(),
                    "critic": dmodel.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizerD": optimizer_D.state_dict(),
                },
                os.path.join(output_path_model, "model_{}.pth".format(batch_size)),
            )
            # save losses and icp score
            with open('results/pickled_result_{}.pkl'.format(batch_size),'wb') as f :
                pickle.dump( (icp, (G_loss, D_loss)), f )
 

