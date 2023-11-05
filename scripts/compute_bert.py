import sys
import os

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from importlib.machinery import SourceFileLoader
constants = SourceFileLoader("constants","cdl/constants.py").load_module()
import constants
import time

def embed_and_save(content, model, path, check_empty=False):
    start = time.time()
    if any(pd.isna(content)): print("Warning: there are NaNs in input content!")
    BATCH_SIZE = 32
    embeddings = model.encode(content, convert_to_tensor=True, show_progress_bar=True, batch_size=BATCH_SIZE)
    if check_empty:
        assert type(content)==pd.Series, f'Not applicable to {type(content)}'
        empty_mask = content == ''
        if any(empty_mask): 
            mask_ = torch.tensor(empty_mask).unsqueeze(-1).expand(embeddings.size())
            embeddings = embeddings.masked_fill_(mask_, 0)
    print('Elapsed time (seconds):', time.time() - start)

    print(f'Saving embeddings with size {embeddings.shape} to {path}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(embeddings, path)
    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Computing BERT embeddings with SentenceTransformer model')
    parser.add_argument('--dataset', type=str, default='citeulike-a')
    parser.add_argument('--in_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--st_model', type=str, default='allenai-specter')
    # parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.st_model
    model = SentenceTransformer(model_name)

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
        df = pd.read_csv(args.in_path)
        for col in constants.AMZ_EMBEDDED_COLS[:1]:
            df_col = df[col].fillna('')
            print(f'Embedding column {col}...')
            path = (f'data/processed/{dataset_name}/{col}_embedded_{model_name}.pt'
                    if args.out_path is None else args.out_path)
            embed_and_save(df_col, model, path, check_empty=True)


