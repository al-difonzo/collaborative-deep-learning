import logging

import torch
from torch import linalg
import torch.nn.functional as F


class LatentRepresentationOptimizer:
    """
    Computes latent user and latent item representations given a ratings target and a latent item representation target.
    """

    def __init__(self, mf, ratings, conf_a, conf_b, lambda_u, lambda_v):
        self.mf = mf
        self.ratings = ratings

        self.conf_a = conf_a
        self.conf_b = conf_b

        self.lambda_u = lambda_u
        self.lambda_v = lambda_v

        # Precompute the nonzero entries of each row and column from the ratings matrix.
        self._ratings_nonzero_rows = [row.nonzero().squeeze(1) for row in ratings]
        self._ratings_nonzero_cols = [col.nonzero().squeeze(1) for col in ratings.t()]

    def step(self, latent_item_target):
        self.step_users()
        self.step_items(latent_item_target)

    def step_users(self):
        """Minimize the loss holding all but the latent users matrix constant."""
        # We compute Vt @ Ci @ V with the following optimization. Recall
        # that Ci = diag(C_i1, ..., C_iJ) where C_ij is a if R_ij = 1 and
        # b otherwise. So we have
        #
        #   Vt @ Ci @ V = Vt @ diag((a - b) * Ri + b * ones) @ V
        #               = (a - b) Vt @ diag(Ri) @ V + b * Vt @ V
        #
        # Notice that since Ri is a zero-one matrix, diag(Ri) simply kills
        # the items of V that user i has does not have in her library; indeed,
        #                Vt @ diag(Ri) @ V = Wt @ Wr,
        # where W is the submatrix restricted to rows j with R_ij != 0.
        # Since W will be *much* smaller than V, it is much more efficient to
        # first extract this submatrix.

        U = self.mf.U
        V = self.mf.V

        conf_a = self.conf_a
        conf_b = self.conf_b

        latent_size = self.mf.latent_size

        A_base = conf_b * V.t() @ V + self.lambda_u * torch.eye(latent_size)
        for j in range(len(U)):
            rated_idx = self._ratings_nonzero_rows[j]
            W = V[rated_idx, :]
            A = (conf_a - conf_b) * W.t() @ W + A_base
            # R[j, rated_idx] is just an all-ones vector
            b = conf_a * W.t().sum(dim=1)

            U[j] = linalg.solve(A, b)

    def step_items(self, latent_items_target):
        """Minimize the loss holding all but the latent items matrix constant."""

        latent_items_target = latent_items_target * self.lambda_v

        U = self.mf.U
        V = self.mf.V

        conf_a = self.conf_a
        conf_b = self.conf_b

        latent_size = self.mf.latent_size

        A_base = conf_b * U.t() @ U + self.lambda_v * torch.eye(latent_size)
        for j in range(len(V)):
            rated_idx = self._ratings_nonzero_cols[j]
            if len(rated_idx) == 0:
                A = A_base
                b = latent_items_target[j]
            else:
                W = U[rated_idx, :]
                A = (conf_a - conf_b) * W.t() @ W + A_base
                # R[rated_idx, j] is just an all-ones vector
                b = conf_a * W.t().sum(dim=1) + latent_items_target[j]

            V[j] = linalg.solve(A, b)

    def loss(self, latent_items_target):
        conf_mat = (self.conf_a - self.conf_b) * self.ratings + self.conf_b * torch.ones_like(self.ratings)

        loss_v = self.lambda_v * F.mse_loss(self.mf.V, latent_items_target, reduction='sum') / 2
        loss_u = self.lambda_u * self.mf.U.square().sum() / 2
        loss_r = (conf_mat * (self.ratings - self.mf.predict()).square()).sum() / 2

        loss = loss_v + loss_u + loss_r
        logging.info(
            f'  neg_likelihood={loss:>5f}'
            f'  v={loss_v:>5f}'
            f'  u={loss_u:>5f}'
            f'  r={loss_r:>5f}'
        )
        return loss
