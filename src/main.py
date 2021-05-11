import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim

import data
import evaluate
from autoencoder import Autoencoder, StackedAutoencoder
from cdl import LatentFactorModel
from train import train_stacked_autoencoders, train_isolated_autoencoder, train_model


def load_model(filename, sdae, lfm, map_location=None):
    checkpoint = torch.load(filename, map_location=map_location)
    sdae.load_state_dict(checkpoint['sdae'])
    lfm.U = checkpoint['U']
    lfm.V = checkpoint['V']


def save_checkpoint(filename, sdae, lfm):
    torch.save({
        'sdae': sdae.cpu().state_dict(),
        'U': lfm.U,
        'V': lfm.V,
    }, filename)


sdae_activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
}

recon_losses = {
    'mse': nn.MSELoss(),
    'cross-entropy': nn.BCEWithLogitsLoss(),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collaborative Deep Learning implementation.')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--embedding', choices=['bert', 'bow'], default='bert')
    parser.add_argument('--recall', type=int, default=300)
    parser.add_argument('--out', default='model.pt')

    parser.add_argument('--conf_a', type=float, default=1.0)
    parser.add_argument('--conf_b', type=float, default=0.01)

    parser.add_argument('--lambda_u', type=float, default=13.9)
    parser.add_argument('--lambda_v', type=float, default=25.0)
    parser.add_argument('--lambda_w', type=float, default=1e-4)
    parser.add_argument('--lambda_n', type=float, default=4.5e4)

    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=150)

    # SDAE hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--corruption', type=float, default=0.3)
    parser.add_argument('--activation', choices=sdae_activations.keys(), default='sigmoid')
    parser.add_argument('--recon_loss', choices=recon_losses.keys(), default='mse')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[200])
    parser.add_argument('--latent_size', type=int, default=50)

    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.verbose:
        logging.basicConfig(format='%(asctime)s  %(message)s', datefmt='%I:%M:%S', level=logging.INFO)

    # Note: SDAE inputs and parameters will use the GPU if desired, but U and V
    # matrices of CDL do not go on the GPU (and therefore nor does the ratings
    # matrix).
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    logging.info(f'Loading content dataset ({args.embedding})')
    if args.embedding == 'bert':
        content_dataset = torch.load('data/processed/citeulike-a/content-bert.pt', map_location=device)
    else:
        content_dataset = torch.load('data/processed/citeulike-a/content-bag.pt', map_location=device).to_dense()
    num_items, in_features = content_dataset.shape
    # content_dataset.shape: (16980, 8000)

    logging.info('Loading ratings datasets')
    ratings_training_dataset = torch.load('data/processed/citeulike-a/cf-train-1-users.pt')
    ratings_test_dataset = torch.load('data/processed/citeulike-a/cf-test-1-users.pt')

    config = {
        'conf_a': args.conf_a,
        'conf_b': args.conf_b,
        'lambda_u': args.lambda_u,
        'lambda_v': args.lambda_v,
        'lambda_w': args.lambda_w,
        'lambda_n': args.lambda_n,
        'dropout': args.dropout,
        'corruption': args.corruption,
    }

    recon_loss_fn = recon_losses[args.recon_loss]
    activation = sdae_activations[args.activation]

    layer_sizes = [in_features] + args.hidden_sizes + [args.latent_size]
    logging.info(f'Using autoencoder architecture {"x".join(map(str, layer_sizes))}')

    autoencoders = [
        Autoencoder(in_features, out_features, args.dropout, activation, tie_weights=True)
        for in_features, out_features in zip(layer_sizes, layer_sizes[1:])
    ]
    sdae = StackedAutoencoder(autoencoder_stack=autoencoders)
    sdae.to(device)

    lfm = LatentFactorModel(target_shape=ratings_training_dataset.shape, latent_size=args.latent_size)

    logging.info(f'Config: {config}')
    optimizer = optim.AdamW(sdae.parameters(), lr=args.lr, weight_decay=args.lambda_w)

    content_training_dataset = data.random_subset(content_dataset, int(num_items * 0.8))

    logging.info(f'Pretraining SDAE with {args.recon_loss} loss')
    train_stacked_autoencoders(autoencoders, args.corruption, content_training_dataset, optimizer, recon_loss_fn, epochs=args.pretrain_epochs, batch_size=args.batch_size)
    train_isolated_autoencoder(sdae, content_training_dataset, args.corruption, args.pretrain_epochs, args.batch_size, recon_loss_fn, optimizer)

    logging.info(f'Training with recon loss {args.recon_loss}')
    train_model(sdae, lfm, content_dataset, ratings_training_dataset, optimizer, recon_loss_fn, config, epochs=args.epochs, batch_size=args.batch_size, device=device)

    logging.info(f'Saving model to {args.out}')
    torch.save({
        'autoencoder': sdae.state_dict(),
        'latent_factor_model': lfm.state_dict(),
    }, args.out)

    logging.info(f'Predicting')
    pred = lfm.predict()

    logging.info(f'Calculating recall@{args.recall}')
    recall = evaluate.recall(pred, ratings_test_dataset.to_dense(), args.recall)

    print(f'recall@{args.recall}: {recall.item()}')
