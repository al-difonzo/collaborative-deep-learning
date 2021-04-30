import torch
import torch.nn as nn
from torch.autograd import Variable

from sdae import StackedDenoisingAutoencoder


class CollaborativeDeepLearning(nn.Module):
    def __init__(self, in_features, num_users, num_items, layer_sizes, corruption, dropout):
        super().__init__()

        self.sdae = StackedDenoisingAutoencoder(in_features, layer_sizes, corruption, dropout)

        latent_size = layer_sizes[-1]

        # Not parameters; we train these manually with coordinate descent.
        self.U = nn.Parameter(torch.Tensor(num_users, latent_size), requires_grad=False)
        self.V = nn.Parameter(torch.Tensor(num_items, latent_size), requires_grad=False)
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

    def forward(self, x):
        # Second input is the ratings matrix; ignore it.
        x, _ = x
        reconstruction = self.sdae(x)
        ratings_pred = self.U @ self.V.t()
        return reconstruction, ratings_pred
