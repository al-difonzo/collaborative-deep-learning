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
    parser.add_argument('--user_rec_path', type=str, default=None)
    parser.add_argument('--topk', type=int, default=300)
    parser.add_argument('--out', default='model.pt')
    parser.add_argument('--hyperopt', action='store_true')
    parser.add_argument('--optuna', action='store_true')

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

    # Note: SDAE inputs and parameters will use the GPU if desired, but U and V
    # matrices of CDL do not go on the GPU (and therefore nor does the ratings
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
    ratings_test_dataset = data.load_cf_test_data(args.dataset, args.test_dataset_path)
    logging.info(f'Size of ratings_test_dataset: {ratings_test_dataset.size()}')

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
    if args.optuna:
        optuna_hyper = hyper.OptunaWrapper(args, sdae, mfm, 
                            ratings_training_dataset, ratings_valid_dataset, content_dataset, 
                            recon_loss_fn, activation)
        
        study = optuna_hyper.optimize(n_trials=2, study_name=f'CDL_{int(time.time())}')

        logging.info(f'Saving best model to {args.out}')
        data.save_model(study.best_trial.user_attrs['sdae'], 
                        study.best_trial.user_attrs['mfm'],
                        args.out)
    else:
        optimizer = optim.AdamW(sdae.parameters(), lr=args.lr, weight_decay=args.lambda_w)

        content_training_dataset = data.random_subset(content_dataset, int(num_items * 0.8), rng=rng)

        logging.info(f'Pretraining SDAE with {args.recon_loss} loss')
        train_stacked_autoencoder(sdae, content_training_dataset, args.corruption, args.pretrain_epochs, args.batch_size, recon_loss_fn, optimizer)

        logging.info(f'Training with recon loss {args.recon_loss}')
        train_model(sdae, mfm, content_dataset, ratings_training_dataset, optimizer, recon_loss_fn, config, epochs=args.epochs, batch_size=args.batch_size, device=device)

        logging.info(f'Saving model to {args.out}')
        data.save_model(sdae, mfm, args.out)

    logging.info(f'Loading trained model from {args.out}')
    data.load_model(sdae, mfm, args.out)
    
    if args.user_rec_path is None: args.user_rec_path = f'{args.dataset}_{args.embedding}_user_recommendations_{args.topk}.csv'
    logging.info(f'Saving user recommendations to {args.user_rec_path}')
    user_rec_df = mfm.get_user_recommendations(ratings_test_dataset.to_dense(), args.topk)
    os.makedirs(os.path.dirname(args.user_rec_path), exist_ok=True)
    user_rec_df.to_csv(args.user_rec_path)

    logging.info(f'Calculating recall@{args.topk}')
    recall = mfm.compute_recall(ratings_test_dataset.to_dense(), args.topk)
    logging.info(f'Recall@{args.topk}: {recall.item()}')
