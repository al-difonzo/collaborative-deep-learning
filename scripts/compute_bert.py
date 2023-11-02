import sys

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import path
directory = path.path(__file__).abspath()
sys.path.append(directory.parent.parent)
import constants

def embed_and_save(content, path, check_for_nan=False):
    embeddings = model.encode(content, convert_to_tensor=True)
    if check_for_nan:
        assert type(content)==pd.Series, f'Not applicable to {type(content)}'
        empty_mask = [x=='' for x in content]
        idx_first_non_empty_elem = content[~empty_mask].index[0]
        zero_filler = torch.zeros_like(embeddings[idx_first_non_empty_elem])
        embeddings = embeddings.masked_fill_(empty_mask, zero_filler)
    
    torch.save(embeddings, path)
    return embeddings


if __name__ == '__main__':
    dataset_name = sys.argv[1]
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
        df = pd.read_csv(f'data/raw/{dataset_name}/preprocessed_CF.csv')
        for col in constants.AMZ_EMBEDDED_COLS:
            df_col = df[col].fillna('')
            embed_and_save(df[col], f'data/processed/{dataset_name}/{col}_embedded_{ST_MODEL}.pt', check_for_nan=True)


