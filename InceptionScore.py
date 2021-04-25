import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

import numpy as np
from scipy.stats import entropy
from utils import *


def inception_score(imgs, cuda=False, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
        print('Im using cuda')
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        print('Im not using cuda')
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):

        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        print(str(i)+'/'+str(len(dataloader)))

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)



def inception_score_training(imgs,inception_model, gmodel, batch_size=32, N =100 , resize=True, splits=1):
    """Computes the inception score of the generated images imgs
    #generate images
    # compute
    #  N : number of batches to generate
    """    

    up = nn.Upsample(size=(299, 299), mode='bilinear',align_corners=False ).to(device)
    
    # Get predictions
    preds = np.zeros((batch_size*N, 1000))

    with torch.no_grad():
        for i, samples in enumerate(imgs):
            pred = F.softmax(inception_model(up(samples)),dim=1).data.cpu().numpy()
            preds[i*batch_size:i*batch_size + batch_size] = pred
            torch.cuda.empty_cache()

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores) #, np.std(split_scores)




# if __name__ == '__main__':
#     class IgnoreLabelDataset(torch.utils.data.Dataset):
#         def __init__(self, orig):
#             self.orig = orig

#         def __getitem__(self, index):
#             return self.orig[index][0]

#         def __len__(self):
#             return len(self.orig)

#     import torchvision.datasets as dset
#     import torchvision.transforms as transforms


#     '''
#     cifar = dset.CIFAR10(root='data/', download=True,
#                              transform=transforms.Compose([
#                                  transforms.Scale(32),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                              ])
#     )

#     IgnoreLabelDataset(cifar)
#     '''
#     #data_dir = 'Imagenet'
#     #dataset = 'cifar'
#     batch_size = 128
#     #train_loader = data_loader.fetch_dataloader(data_dir, batch_size, dataset)

#     train_transformer = transforms.Compose([
#             transforms.Resize((64,64)),        # resize the image to 64x64 (remove if images are already 64x64)
#             transforms.RandomHorizontalFlip(),   # randomly flip image horizontally
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # transform it into a torch tensor


#     print ("Calculating Inception Score...")
#     #print (inception_score(data_loader.ImagenetDataset(data_dir, train_transformer), cuda=False, batch_size=32, resize=True, splits=10))
#     #print (inception_score(data_loader.ChairsDataset('clean_chairs', train_transformer), cuda=False, batch_size=32, resize=True, splits=10))
#     print (inception_score(data_loader.ChairsDataset('clean_chairs', train_transformer), cuda=False, batch_size=32, resize=True, splits=10))
