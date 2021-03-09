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
from data import *
from db_config import config
import seaborn as sns


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
    wandb.init(project='EECS 545 - Diabetes GAN', name=hparams['exp'], config=hparams)

    # Load model.
    model = GAN(hparams)
    wandb.watch(model.generator)
    wandb.watch(model.discriminator)

    # Load data.
    data = load_diabetes_data(batch_size=hparams['batch_size'] // 2)

    # Optimizers.
    optimizer = getattr(torch.optim, hparams['optim'])
    optimizer_d = optimizer(params=model.discriminator.parameters(), lr=hparams['lr'])
    optimizer_g = optimizer(params=model.generator.parameters(), lr=hparams['lr'])

    # Train.
    for epoch in range(hparams['epochs']):
        for batch_idx, x in enumerate(data):
            # Flatten and migrate to GPU
            x = torch.flatten(x[0], start_dim=1).float()
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
                examples = model.generate(num_samples=hparams['batch_size'])
            examples = examples.detach().cpu().numpy()
            sns.heatmap(examples)
            plt.savefig('test.png')
            plt.clf()
    torch.save(model.generator.state_dict(), 'ckpts/{}'.format(args['ckpt_name']))


def generate_synthetic_data(model_pos, model_neg, samples):
    """
    model_pos:  model that generates the postive samples (path)
    model_neg: model that generates the negative samples (path)
    samples: this is the numner of samples that are generated
    """
    with torch.no_grad():
        model_p = GAN(config)
        model_p.generator.load_state_dict(torch.load(model_pos))
        model_p.generator.eval()
        pos = model_p.generate(samples)

        model_n = GAN(config)
        model_n.generator.load_state_dict(torch.load(model_neg))
        model_n.generator.eval()
        neg = model_n.generate(samples)

    pos = pos.detach().cpu().numpy()
    neg = neg.detach().cpu().numpy()

    pos_subsample = np.where(pos[:, 1:] < 0, -1, 1)
    neg_subsample = np.where(neg[:, 1:] < 0, -1, 1)

    pos = np.hstack((pos[:, 0:1],
                     pos_subsample,
                     np.ones((samples, 1))))

    neg = np.hstack((neg[:, 0:1],
                     neg_subsample,
                     np.zeros((samples, 1))))

    pos = pd.DataFrame(pos)
    neg = pd.DataFrame(neg)

    # merge into one file with labels and denormalize age.
    merge_synthetic_data(pos, neg)


def main():
    generate_synthetic_data('ckpts/saved_generator_positive0.5.pt', 'ckpts/saved_generator_negative0.5.pt',
                            1000)


if __name__ == '__main__':
    main()
