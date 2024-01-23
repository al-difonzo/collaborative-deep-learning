import torch
from torch import nn
import pandas as pd

class MatrixFactorizationModel:
    def __init__(self, target_shape, latent_size):
        super().__init__()

        self.U = torch.empty((target_shape[0], latent_size))
        self.V = torch.empty((target_shape[1], latent_size))
        self.latent_size = latent_size

        nn.init.normal_(self.U, 0, 0.1)
        nn.init.normal_(self.V, 0, 0.1)

    def estimate(self):
        return self.U @ self.V.t()

    def state_dict(self):
        return {'U': self.U, 'V': self.V}

    def update_state_dict(self, d):
        self.U = d['U']
        self.V = d['V']
        assert self.U.shape[1] == self.V.shape[1]
        self.latent_size = self.U.shape[1]

    def get_user_recommendations(self, test, k):
        scores, indices = torch.topk(self.estimate(), k)
        data = {'itemIds': indices.tolist()}
        user_rec_df = pd.DataFrame(data, columns=['itemIds'])
        user_rec_df['scores'] = scores.tolist()
        user_rec_df.index.name = 'userId'
        return user_rec_df

    def compute_recall(self, test, k):
        _, indices = torch.topk(self.estimate(), k)
        gathered = test.gather(1, indices)
        recall = gathered.sum(dim=1) / test.sum(dim=1)
        # We use nanmean because there may be some users with 0 ratings in test set, thus test.sum(dim=1) may contain some 0s
        return recall.nanmean()
