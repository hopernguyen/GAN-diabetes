import torch
import torchvision


class Discriminator(torch.nn.Module):
    """
    Discriminator.
    """

    def __init__(self, in_dim, h_dims):
        super(Discriminator, self).__init__()
        self.fcs = torch.nn.ModuleList([])
        self.input = torch.nn.Linear(in_dim, h_dims[0])
        for i in range(len(h_dims) - 1):
            self.fcs.append(torch.nn.Linear(h_dims[i], h_dims[i + 1]))
        self.fcs.append(torch.nn.Linear(h_dims[-1], 1))
        self.out_activation = torch.nn.Sigmoid()
        self.hidden_activation = torch.nn.ReLU()

    def forward(self, x):
        z = self.input(x)
        z = self.hidden_activation(z)
        for fc in self.fcs[:-1]:
            z = fc(z)
            z = self.hidden_activation(z)
        z = self.fcs[-1](z)
        return z


class Generator(torch.nn.Module):
    """
    Generator.
    """

    def __init__(self, in_dim, h_dims, out_dim):
        super(Generator, self).__init__()
        self.fcs = torch.nn.ModuleList([])
        self.input = torch.nn.Linear(in_dim, h_dims[0])
        for i in range(len(h_dims) - 1):
            self.fcs.append(torch.nn.Linear(h_dims[i], h_dims[i + 1]))
        self.fcs.append(torch.nn.Linear(h_dims[-1], out_dim))
        self.out_activation = torch.nn.Tanh()
        self.hidden_activation = torch.nn.ReLU()

    def forward(self, x):
        z = self.input(x)
        z = self.hidden_activation(z)
        for fc in self.fcs[:-1]:
            z = fc(z)
            z = self.hidden_activation(z)
        z = self.fcs[-1](z)
        z = self.out_activation(z)
        return z
