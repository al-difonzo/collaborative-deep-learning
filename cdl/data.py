import torch


class TransformDataset:
    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, i):
        return self._transform(self._dataset[i])

    def __len__(self):
        return len(self._dataset)


def random_subset(x, k):
    idx = torch.randperm(len(x))[:k]
    return torch.utils.data.Subset(x, idx)


def bernoulli_corrupt(x, p):
    mask = torch.rand_like(x) > p
    return x * mask


def load_content_embeddings(dataset_name, embedding=None, path=None, device=None):
    if embedding is None and path is None: 
        raise Exception('Please specify either embedding model with --embedding,' 
        'or custom path of precomputed embeddings with --embedding_path')
    if path is None: path = f'data/processed/{dataset_name}/content-{embedding}.pt'
    # Make sure dtype of loaded embeddings is double (float64)
    x = torch.load(path, map_location=device).to(torch.double)
    if x.is_sparse:
        x = x.to_dense()
    return x


def load_cf_train_data(dataset_name, path=None):
    if path is None: path = f'data/processed/{dataset_name}/cf-train-1.pt'
    return torch.load(path)


def load_cf_test_data(dataset_name, path=None):
    if path is None: path = f'data/processed/{dataset_name}/cf-test-1.pt'
    return torch.load(path)


def save_model(sdae, mfm, filename):
    torch.save({
        'autoencoder': sdae.state_dict(),
        'matrix_factorization_model': mfm.state_dict(),
    }, filename)


def load_model(sdae, mfm, filename):
    d = torch.load(filename)

    if sdae is not None:
        sdae.update_state_dict(d['autoencoder'])

    if mfm is not None:
        mfm.update_state_dict(d['matrix_factorization_model'])

# AMZ_CHOICES = [
#     'Amazon Fashion'
#     'All Beauty'
#     'Appliances'
#     'Arts, Crafts and Sewing'
#     'Automotive'
#     'Books'
#     'CDs and Vinyl'
#     'Cell Phones and Accessories'
#     'Clothing, Shoes and Jewelry'
#     'Digital Music'
#     'Electronics'
#     'Gift Cards'
#     'Grocery and Gourmet Food'
#     'Home and Kitchen'
#     'Industrial and Scientific'
#     'Kindle Store'
#     'Luxury Beauty'
#     'Magazine Subscriptions'
#     'Movies and TV'
#     'Musical Instruments'
#     'Office Products'
#     'Patio, Lawn and Garden'
#     'Pet Supplies'
#     'Prime Pantry'
#     'Software'
#     'Sports and Outdoors'
#     'Tools and Home Improvement'
#     'Toys and Games'
#     'Video Games'
# ]