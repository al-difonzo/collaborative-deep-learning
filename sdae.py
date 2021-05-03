import torch.nn as nn


class StackedAutoencoder(nn.Module):
    def __init__(self, autoencoder_stack):
        super().__init__()

        self.autoencoders = autoencoder_stack

        self.encode = nn.Sequential(*[autoencoder.encode for autoencoder in self.autoencoders])
        self.decode = nn.Sequential(*[autoencoder.decode for autoencoder in reversed(self.autoencoders)])

        self.weights = [weight for ae in self.autoencoders for weight in ae.weights]
        self.biases = [bias for ae in self.autoencoders for bias in ae.biases]

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed

    def regularization_term(self, reg):
        s = 0
        s += reg * sum(weight.square().sum() for weight in self.weights)
        s += reg * sum(bias.square().sum() for bias in self.biases)
        return s


class Autoencoder(nn.Module):
    def __init__(self, in_features, latent_size, dropout=0, activation=nn.Sigmoid(), tie_weights=True):
        """
        Instantiates a Autoencoder.

        :param in_features: The number of features (rows) of the input.
        :param latent_size: The size of the latent representation.
        :param dropout: The dropout probability before decoding.
        :param activiation: The activation function.
        :param tie_weights: Whether to use the same weight matrix in the encoder and decoder.
        """
        super().__init__()

        encode = nn.Linear(in_features, latent_size)
        decode = nn.Linear(latent_size, in_features)

        if tie_weights:
            decode.weight.data = encode.weight.t()
            self.weights = [encode.weight]
        else:
            self.weights = [encode.weight, decode.weight]

        self.biases = [encode.bias.data, decode.bias.data]

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

        for bias in self.biases:
            nn.init.zeros_(bias)

        self.encode = nn.Sequential(
            encode,
            activation,
        )
        self.decode = nn.Sequential(
            nn.Dropout(dropout),
            decode,
        )

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed
