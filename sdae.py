import torch.nn as nn


class StackedDenoisingAutoencoder(nn.Module):
    def __init__(self, in_features, layer_sizes, corruption, dropout):
        super().__init__()

        dims = zip([in_features] + layer_sizes[:-1], layer_sizes)
        self.autoencoders = [
            DenoisingAutoencoder(rows, cols, corruption, dropout)
            for rows, cols in dims
        ]

        self.encode = nn.Sequential(*[autoencoder.encode for autoencoder in self.autoencoders])
        self.decode = nn.Sequential(*[autoencoder.decode for autoencoder in reversed(self.autoencoders)])

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed

    @property
    def weights(self):
        return [ae.weight for ae in self.autoencoders]

    @property
    def biases(self):
        return [bias for ae in self.autoencoders for bias in ae.bias]


class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_features, latent_size, corruption, dropout):
        """
        Instantiates a DenoisingAutoencoder.

        :param in_features: The number of features (rows) of the input.
        :param latent_size: The size of the latent representation.
        :param corruption: The probability of corrupting (zeroing) each element of the input.
        :param dropout: The dropout probability before decoding.
        """
        super().__init__()

        encode = nn.Linear(in_features, latent_size)
        decode = nn.Linear(latent_size, in_features)
        decode.weight.data = encode.weight.t()

        self.weight = encode.weight
        self.biases = [encode.bias.data, decode.bias.data]

        self.encode = nn.Sequential(
            nn.Dropout(corruption),
            encode,
            nn.Sigmoid(),
        )
        self.decode = nn.Sequential(
            nn.Dropout(dropout),
            decode,
        )

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed
