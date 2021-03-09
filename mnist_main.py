import os
import sys
import torch
import torchvision
import numpy as np
import wandb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import networks
from data import load_mnist_data
from config import config


class GAN:
    """
    Generative Adversarial Network. Encapsulates a generator network, a discriminator network,
    functionality for generating with the generator, and losses for training both networks.
    """

    def __init__(self, hparams):
        self.discriminator = networks.Discriminator(in_dim=hparams['networks']['discriminator']['in_dim'],
                                                    h_dims=hparams['networks']['discriminator']['h_dims'])
        self.generator = networks.Generator(in_dim=hparams['networks']['generator']['in_dim'],
                                            h_dims=hparams['networks']['generator']['h_dims'],
                                            out_dim=hparams['networks']['generator']['out_dim'])
        if hparams['device'] == 'gpu':
            self.generator.cuda()
            self.discriminator.cuda()
        self.hparams = hparams

    def generate(self, num_samples):
        z = torch.randn((num_samples, self.hparams['networks']['generator']['z_dim']))
        if self.hparams['device'] == 'gpu':
            z = z.cuda()
        return self.generator(z)

    def loss_d(self, x, x_g):
        real_logits = self.discriminator(x)
        fake_logits = self.discriminator(x_g)
        logits = torch.vstack((real_logits, fake_logits))
        targets = torch.vstack((torch.zeros((real_logits.shape[0], 1)),
                                torch.ones((fake_logits.shape[0], 1))))
        if self.hparams['device'] == 'gpu':
            targets = targets.cuda()
        accuracy = accuracy_score(self.discriminator.out_activation(targets).detach().cpu().numpy().round(),
                                  self.discriminator.out_activation(logits).detach().cpu().numpy().round())
        return torch.nn.BCEWithLogitsLoss()(logits, targets), accuracy

    def loss_g(self, x_g):
        fake_logits = self.discriminator(x_g)
        targets = torch.zeros((fake_logits.shape[0], 1))
        if self.hparams['device'] == 'gpu':
            targets = targets.cuda()
        return torch.nn.BCEWithLogitsLoss()(fake_logits, targets)


def train(hparams):
    # Setup experiment.
    wandb.init(project='EECS 545 - GAN', name=hparams['exp'], config=hparams)

    # Load model.
    model = GAN(hparams)
    wandb.watch(model.generator)
    wandb.watch(model.discriminator)

    # Load data.
    data = load_mnist_data(batch_size=hparams['batch_size'] // 2)

    # Optimizers.
    optimizer = getattr(torch.optim, hparams['optim'])
    optimizer_d = optimizer(params=model.discriminator.parameters(), lr=hparams['lr'])
    optimizer_g = optimizer(params=model.generator.parameters(), lr=hparams['lr'])

    # Train.
    for epoch in range(hparams['epochs']):
        for batch_idx, (x, _) in enumerate(data):
            # Flatten and migrate to GPU
            x = torch.flatten(x, start_dim=1)
            if hparams['device'] == 'gpu':
                x = x.cuda()

            # Train discriminator.
            x_g = model.generate(num_samples=x.shape[0]).detach()
            loss_d, acc_d = model.loss_d(x, x_g)
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # Train generator.
            x_g = model.generate(num_samples=x.shape[0])
            loss_g = model.loss_g(x_g)
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            # Logging.
            print("epoch: {}/{} | batch: {}/{} | loss_g: {} | loss_d: {} | acc_d: {}%".format(
                epoch, hparams['epochs'], batch_idx, len(data), loss_g, loss_d, acc_d * 100
            ))
            wandb.log({
                "loss_g": loss_g,
                "loss_d": loss_d,
                "acc_d": acc_d,
            })

        # Checkpointing and saving samples.
        if epoch % hparams['save_interval'] == 0:
            with torch.no_grad():
                examples = model.generate(num_samples=64)
            examples = (examples.reshape(-1, 28, 28) + 1) * 255 / 2
            examples = np.round(examples.detach().cpu().numpy())[:64]
            fig = plt.figure(figsize=(64, 64))
            grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=1)
            for i, (ax, im) in enumerate(zip(grid, examples)):
                ax.imshow(np.reshape(im, (28, 28)), cmap=plt.get_cmap('gray'))
            wandb.log({'Generated samples': plt})


def main():
    train(config)


if __name__ == '__main__':
    main()
