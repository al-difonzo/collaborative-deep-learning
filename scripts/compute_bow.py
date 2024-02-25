import sys
import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

from cdl import constants


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


def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove punctuation
    tokens = [word for word in tokens if word.isalnum()]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def compute_ml_bow(rich_df, out_path):
    rich_df['combined_info_cleaned'] = rich_df['combined_info'].map(preprocess_text)

    # Compute TF-IDF scores
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(rich_df['combined_info_cleaned'])

    # Calculate average TF-IDF score for each word
    word_tfidf = dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_matrix.mean(axis=0).tolist()[0]))

    # Select top discriminative words based on TF-IDF scores
    VOCAB_SIZE = 8000 # from 31116 words
    top_words = dict(Counter(word_tfidf).most_common(VOCAB_SIZE))
    vocabulary = list(top_words.keys())
    assert len(vocabulary) == VOCAB_SIZE

    rich_df['combined_info_cleaned'] = rich_df['combined_info_cleaned'].map(lambda text: ' '.join([w for w in word_tokenize(text.lower()) if w in vocabulary]))

    # Initialize CountVectorizer to tokenize words and create bag of words representation
    vectorizer = CountVectorizer()
    # Fit the CountVectorizer on movie combined_info and transform them to bag of words representation
    bow_matrix = vectorizer.fit_transform(rich_df['combined_info_cleaned']) # a 'scipy.sparse.csr.csr_matrix' one-hot encoded
    # Convert bag of words matrix to DataFrame for easier manipulation
    # bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names())

    rich_df['computed_info_embedded_TFIDF_BOW'] = bow_matrix.toarray().tolist()
    print(rich_df.head())

    tensor_to_save = torch.tensor(rich_df['computed_info_embedded_TFIDF_BOW'].tolist()).type(torch.CharTensor)
    print('Shape of saved tensor:', tensor_to_save.shape)
    torch.save(tensor_to_save, out_path)
    # Little consistency check
    loaded_embd_tensor = torch.load(out_path)
    assert loaded_embd_tensor.shape == tensor_to_save.shape, f'Loaded: {loaded_embd_tensor.shape} vs Saved:{tensor_to_save.shape}'


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
    elif dataset_name in constants.MOVIELENS_CHOICES:
        # path = (f'data/processed/{dataset_name}/{col}_embedded_{st_model}.pt'
        #         if args.out_path is None else args.out_path)
        out_path = 'combined_info_embedded_TFIDF_BOW.pt'
        rich_df = pd.read_csv(args.in_path, index_col=0)
        # rich_df = pd.read_csv('data/preprocessed/ml-latest-small/rich_ml-latest-small_ready_to_embed.csv', index_col=0)
        compute_ml_bow(rich_df, out_path)
