import os
import numpy as np
import pandas as pd
import urllib.request
import zipfile
from importlib.machinery import SourceFileLoader
constants = SourceFileLoader("constants","cdl/constants.py").load_module()

def get_movielens_dataset(ml_choice):
    filename = ml_choice['source_zip']
    url = f'{constants.ML_BASE_URL}/{filename}'
    if not os.path.exists(filename): urllib.request.urlretrieve(url, filename)
    # Unzip the file
    try:
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall()
        print('Content extracted successfully')
        os.remove(filename)  # Remove the zip file after successful extraction
    except Exception as e:
        print(f'Failed to extract content from {filename}. Error: {e}')
        raise e
    
    return pd.read_csv(ml_choice['UIMat_file'])

def enrich_from_tmdb(row:pd.Series):
    '''
    Function to get additional information from TMDb and IMDb
    '''
    import requests
    tmdb_id = row['tmdbId']
    if pd.notna(tmdb_id):
        # Get additional info from TMDb
        url = f"{constants.TMDB_BASE_URL}/{tmdb_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            row['tmdb_title'] = data.get('title', '')
            row['tmdb_overview'] = data.get('overview', '')
            row['tmdb_release_date'] = data.get('release_date', '')
            row['tmdb_vote_average'] = data.get('vote_average', '')
    return row

def enrich_movielens(df, output_path=None):
    # You can put your TMDb API key in your environment with `export TMDB_API_KEY=<api_key>` 
    TMDB_API_KEY = os.environ['TMDB_API_KEY']
    # Retrieve additional information for each movie from The Movie DB
    df = df.apply(enrich_from_tmdb, axis=1)
    print(IFeat_df.head())
    if output_path is not None: df.to_csv(output_path, index=False, encoding='utf-8-sig')    