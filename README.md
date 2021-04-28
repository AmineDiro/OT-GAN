[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rFNs_dYyU_gTCL216TvN8WTFCRAlKmHR?usp=sharing)

# Optimal Transport applied to GANs 

Implementation of GAN variants that built on optimal transport, using a different approach than the Wasserstein-GAN paper, as proposed in the OT-Gan paper (you can also start using this variant of the same idea).
* [Improving GANs Using Optimal Transport](https://arxiv.org/abs/1803.05573).
* [Learning Generative Models with Sinkhorn Divergences](https://arxiv.org/abs/1706.00292)

# Usage

To test training implementation of the GAN model with the Mini batch Energy distance loss presented in the paper you should clone the project. 


Project Organization
------------

`OTGAN` directory is structured as follows  

   
    ├── OTGAN
    │   ├── dataset.py          <- Loading the two batch datasets : MNIST or CIFAR10
    │   ├── Discriminator.py    <- Critic architecture (1 or 3 input channels)
    │   ├── Generator.py        <- Generator architecture  (1 or 3 output channels)
    │   ├── InceptionScore.py   <- Returns inception score for batchs or for dataset
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── SinkhornDistance.py <- Computes the Minibatch Energy distance
    │   ├── train.py            <- Main training loop
    │   └── utils.py            <- Weight init, plotting and sampling



To run the training follow these steps :
1. Clone the repository and cd to the directory
    ```bash
    git clone https://github.com/AmineDiro/OT-GAN.git && cd OT-GAN/
    ```
2. The training has different arguments , run  the command `python -m OTGAN` with the proper arguments : 

    | Short                | Long         | Description                                                       | Default |
    |----------------------|--------------|-------------------------------------------------------------------|---------|
    | -c                   | --channels   | Nb of channels 1 for MNIST,3 for CIFAR , 3 by default             | 3       |
    | -b                   | --batch_size | Batch size for training (default: 64)                             | 64      |
    | -se                  | --save_epoch | Saving model every N epoch                                        | 2       |
    | -si                  | --sample_interval| Interval number for sampling image from generator and saving them | 1       |
    | --score / --no-score |              | Boolean args to get Inception score or not                        | True        | 

**NOTE :** The Notebook `Results.ipynb`  presents the main results from training on the CIFAR10 dataset. We plot the generated images from training, the loss of generator, critic and the inception score while training. You can click on the **[Open In Colab]** to access the notebook on google collab.

