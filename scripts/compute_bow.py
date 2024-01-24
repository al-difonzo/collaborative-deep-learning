import sys

import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def compute_bow(infile, outfile, shape):
    indices = [[], []]
    values = []

    with open(infile) as f:
        for doc_id, line in enumerate(f):
            tokens = line.split()[1:]

            for token in tokens:
                word_id, cnt = tuple(map(int, token.split(':')))
                indices[0].append(doc_id)
                indices[1].append(word_id)
                values.append(cnt)

    x = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32).to_dense()

    maxes, _ = x.max(dim=1, keepdim=True)
    torch.clamp_min_(maxes, 1)
    x /= maxes

    torch.save(x.to_sparse(), outfile)

def compute_amz_bow(content, out_path):
    # Preprocess the text (e.g., remove stopwords, punctuation, etc.)
    # You can use libraries like NLTK or spaCy for text preprocessing.

    # Create a CountVectorizer instance
    vectorizer = CountVectorizer()

    # Fit the vectorizer to the reviews
    X = vectorizer.fit_transform(content)

    # Convert the BoW vectors to an array
    bow_embeddings = X.toarray()

    # The 'bow_embeddings' variable now contains the BoW representations of the reviews
    # You can use these embeddings for further analysis or NLP tasks.


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    shape = {
        'citeulike-a': (16980, 8000),
        'citeulike-t': (25975, 20000),
    }
    compute_bow(f'data/raw/{dataset_name}/mult.dat', f'data/processed/{dataset_name}/content-bow.pt', shape[dataset_name])
    elif dataset_name in constants.AMZ_CHOICES_:
        df = pd.read_csv(args.in_path)
        for col in constants.AMZ_EMBEDDED_COLS[:1]:
            df_col = df[col].fillna('')
            print(f'Embedding column {col}...')
            path = (f'data/processed/{dataset_name}/{col}_embedded_{st_model}.pt'
                    if args.out_path is None else args.out_path)
            compute_amz_bow(df_col, path)
