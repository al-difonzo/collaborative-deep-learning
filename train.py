import logging
from collections import namedtuple

import torch
import torch.cuda
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import DataLoader

import cdl
import data
from cdl import LatentFactorModelOptimizer


def train_model(sdae, lfm, content, ratings, optimizer, recon_loss_fn, config, epochs, batch_size, device=None, max_iters=10):
    """
    Trains the CDL model. For best results, the SDAE should be pre-trained.

    Each training epoch consists of the following steps: (1) update V, (2)
    update U, (3) update W+ and b. In each step, we hold the parameters not
    being updated constant. When all but V is held constant, we can minimize V
    exactly. Similarly, when all but U is constant, we can minimize U exactly.
    In the last step, we batch the input and update W+ and b on each batch with
    one step of a gradient-based iterative algorithm.
    """
    lfm_optim = LatentFactorModelOptimizer(lfm, ratings, config['conf_a'], config['conf_b'], config['lambda_u'], config['lambda_v'])
    latent_loss_fn = cdl.AutoencoderLatentLoss(config['lambda_v'], config['lambda_n'])

    for epoch in range(epochs):
        logging.info(f'Starting epoch {epoch + 1}/{epochs}')

        # Update U and V.
        with autograd.no_grad():
            # Don't use dropout here.
            sdae.eval()
            latent_items_target, recon = sdae(content)
            latent_items_target = latent_items_target.cpu()
            sdae.train()

        lfm_optim.step(latent_items_target)

        if epoch % 3 == 0:
            loss = lfm_optim.loss(latent_items_target).item()
            loss += config['lambda_n'] / 2 * F.mse_loss(recon, content, reduction='sum').item()
            loss += config['lambda_w'] / 2 * (sum(w.square().sum() for w in sdae.weights) + sum(b.square().sum() for b in sdae.biases)).item()
            logging.info(f'  neg_likelihood: {loss}')

        # Update SDAE weights. Loss here only depends on SDAE outputs.
        train_cdl_autoencoder(sdae, content, lfm.V.to(device), config['corruption'], batch_size, recon_loss_fn, latent_loss_fn, optimizer)

    sdae.eval()
    latent_items_target = sdae.encode(content).cpu()

    # Now optimize U and V completely holding the SDAE latent layer fixed.
    prev_loss = None
    for i in range(max_iters):
        lfm_optim.step(latent_items_target)
        loss = lfm_optim.loss(latent_items_target)
        if prev_loss is not None and (prev_loss - loss) / loss < 1e-4:
            break

        prev_loss = loss


def train_stacked_autoencoders(autoencoders, corruption, dataset, optimizer, loss_fn, epochs, batch_size):
    cur_dataset = dataset

    # Layer-wise pretraining.
    for i, autoencoder in enumerate(autoencoders):
        logging.info(f'Training autoencoder {i + 1}/{len(autoencoders)}')

        train_isolated_autoencoder(autoencoder, cur_dataset, corruption, epochs, batch_size, loss_fn, optimizer)

        with torch.no_grad():
            autoencoder.eval()
            cur_dataset = autoencoder.encode(cur_dataset[:])
            autoencoder.train()


def train_isolated_autoencoder(autoencoder, content, corruption, epochs, batch_size, loss_fn, optimizer):
    for epoch in range(epochs):
        logging.info(f'Starting epoch {epoch + 1}/{epochs}')

        dataset = data.TransformDataset(
            content,
            lambda x: (F.dropout(x, corruption), x),
        )
        train(lambda x: autoencoder(x)[1], dataset, loss_fn, batch_size, optimizer)


def train_cdl_autoencoder(autoencoder, content, latent_items, corruption, batch_size, recon_loss_fn, latent_loss_fn, optimizer):
    # Input to autoencoder is add_noise(item); target is (latent_item, item).
    dataset = data.TransformDataset(
        torch.utils.data.TensorDataset(latent_items, content),
        lambda x: (F.dropout(x[1], corruption), x),
    )

    def loss_fn(pred, target):
        latent_pred, recon_pred = pred
        latent_target, recon_target = target
        return recon_loss_fn(recon_pred, recon_target) + latent_loss_fn(latent_pred, latent_target)

    train(autoencoder, dataset, loss_fn, batch_size, optimizer)


def train(model, dataset, loss_fn, batch_size, optimizer):
    dataloader = DataLoader(dataset, batch_size)
    size = len(dataset)

    for i, (xb, yb) in enumerate(dataloader):
        yb_pred = model(xb)
        loss = loss_fn(yb_pred, yb)

        if i % 100 == 0:
            current = i * batch_size
            logging.info(f'  loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
