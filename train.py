import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim

import optuna
import time

from cdl import data
from cdl.autoencoder import Autoencoder, StackedAutoencoder
from cdl.cdl import train_model, train_stacked_autoencoder
from cdl.mf import MatrixFactorizationModel
from cdl import constants
from cdl import hyper


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collaborative Deep Learning training')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--embedding', choices=['bert', 'bow'], default=None)
    parser.add_argument('--embedding_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='citeulike-a')
    parser.add_argument('--train_dataset_path', type=str, default=None)
    parser.add_argument('--test_dataset_path', type=str, default=None)
    parser.add_argument('--model_path', default='model.pt')
    parser.add_argument('--user_rec_path', type=str, default=None)
    parser.add_argument('--topk', type=int, default=300)
    parser.add_argument('--run_optuna', action='store_true')
    parser.add_argument('--optuna_n_trials', type=int, default=30)
    parser.add_argument('--optuna_timeout', type=int, default=600)
    parser.add_argument('--hyperopt', action='store_true')

    parser.add_argument('--conf_a', type=float, default=1.0)
    parser.add_argument('--conf_b', type=float, default=0.01)

    parser.add_argument('--lambda_u', type=float, default=13.9)
    parser.add_argument('--lambda_v', type=float, default=25.0)
    parser.add_argument('--lambda_w', type=float, default=1e-4)
    parser.add_argument('--lambda_n', type=float, default=4.5e4)

    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=150)

    # SDAE hyperparameters
    parser.add_argument('--activation', choices=constants.SDAE_ACTIVATIONS.keys(), default='sigmoid')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--corruption', type=float, default=0.3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_sizes', nargs='*', type=int, default=[200])
    parser.add_argument('--latent_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--recon_loss', choices=constants.RECON_LOSSES.keys(), default='mse')

    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    rng = torch.Generator()

    if args.verbose:
        logging.basicConfig(format='%(asctime)s  %(message)s', datefmt='%I:%M:%S', level=logging.INFO)

    # Create directories for artifacts
    for path in [args.embedding_path, args.train_dataset_path, args.test_dataset_path, args.user_rec_path, args.model_path]: 
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Note: SDAE inputs and parameters will use the GPU (if available), 
    # but U and V matrices of CDL do not go on the GPU (and therefore nor does the ratings
    # matrix).
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    logging.info(f'Loading content dataset ({args.embedding})')
    content_dataset = data.load_content_embeddings(args.dataset, args.embedding, args.embedding_path, device=device)
    num_items, in_features = content_dataset.shape
    logging.info(f'Number of items: {num_items}, Number of item features: {in_features}')

    logging.info('Loading ratings datasets')
    ratings_training_dataset = data.load_cf_train_data(args.dataset, args.train_dataset_path)
    logging.info(f'Size of ratings_training_dataset: {ratings_training_dataset.size()}')
    ratings_valid_dataset = data.load_cf_valid_data(args.dataset, args.train_dataset_path)
    logging.info(f'Size of ratings_valid_dataset: {ratings_valid_dataset.size()}')

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

    recon_loss_fn = constants.RECON_LOSSES[args.recon_loss]
    activation = constants.SDAE_ACTIVATIONS[args.activation]

    layer_sizes = [in_features] + args.hidden_sizes + [args.latent_size]
    logging.info(f'Using autoencoder architecture {"x".join(map(str, layer_sizes))}')

    autoencoders = [
        Autoencoder(in_features, out_features, args.dropout, activation, tie_weights=True)
        for in_features, out_features in zip(layer_sizes, layer_sizes[1:])
    ]
    sdae = StackedAutoencoder(autoencoders)
    sdae.to(device)

    mfm = MatrixFactorizationModel(target_shape=ratings_training_dataset.shape, latent_size=args.latent_size)
    if args.run_optuna:
        optuna_wrapper = hyper.OptunaWrapper(args, sdae, mfm, 
                            ratings_training_dataset, ratings_valid_dataset, content_dataset, 
                            recon_loss_fn, activation)
        
        study_name = os.path.basename(args.model_path)
        storage_name = f"sqlite:///{os.path.splitext(args.model_path)[0]}.db"
        # Optuna will timeout after `args.optuna_timeout` seconds, if not yet stopped due to `args.optuna_n_trials`
        study = optuna_wrapper.optimize(n_trials=args.optuna_n_trials, timeout=args.optuna_timeout, study_name=study_name, storage=storage_name)
    else:
        optimizer = optim.AdamW(sdae.parameters(), lr=args.lr, weight_decay=args.lambda_w)

        content_training_dataset = data.random_subset(content_dataset, int(num_items * 0.8), rng=rng)

        logging.info(f'Pretraining SDAE with {args.recon_loss} loss')
        train_stacked_autoencoder(sdae, content_training_dataset, args.corruption, args.pretrain_epochs, args.batch_size, recon_loss_fn, optimizer)

        logging.info(f'Training with recon loss {args.recon_loss}')
        train_model(sdae, mfm, content_dataset, ratings_training_dataset, optimizer, recon_loss_fn, config, epochs=args.epochs, batch_size=args.batch_size, device=device)

        logging.info(f'Saving model to {args.model_path}')
        data.save_model(sdae, mfm, args.model_path)