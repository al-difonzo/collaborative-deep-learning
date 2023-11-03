import sys
import os

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from importlib.machinery import SourceFileLoader
constants = SourceFileLoader("constants","cdl/constants.py").load_module()
import constants

def embed_and_save(content, path, check_for_nan=False):
    embeddings = model.encode(content, convert_to_tensor=True)
    if check_for_nan:
        assert type(content)==pd.Series, f'Not applicable to {type(content)}'
        nan_mask = pd.isna(content)
        if all(nan_mask): raise ValueError("Please provide an iterable with at least 1 non-empty element!")
        mask_ = torch.tensor(nan_mask).unsqueeze(-1).expand(embeddings.size())
        embeddings = embeddings.masked_fill_(mask_, 0)
    
    print(f'Saving embeddings with size {embeddings.shape} to {path}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(embeddings, path)
    return embeddings


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    in_path = sys.argv[2] if len(sys.argv)>=3 else None
    out_path = sys.argv[3] if len(sys.argv)>=4 else None
    ST_MODEL = 'allenai-specter'
    model = SentenceTransformer(ST_MODEL)

    if dataset_name == 'citeulike-a':
        df = pd.read_csv(f'data/raw/{dataset_name}/raw-data.csv')
        content = df['title'] + ' ' + df['raw.abstract']
        embed_and_save(content, f'data/processed/{dataset_name}/content-bert.pt')

    elif dataset_name == 'citeulike-t':
        with open(f'data/raw/{dataset_name}/rawtext.dat') as f:
            lines = f.readlines()
        content = lines[1::2]
        embed_and_save(content, f'data/processed/{dataset_name}/content-bert.pt')

    elif dataset_name in constants.AMZ_CHOICES_:
        df = pd.read_csv(in_path)
        for col in constants.AMZ_EMBEDDED_COLS:
            print(f'Embedding column {col}...')
            if out_path is None: out_path = f'data/processed/{dataset_name}/{col}_embedded_{ST_MODEL}.pt'
            embed_and_save(df[col], out_path, check_for_nan=True)


