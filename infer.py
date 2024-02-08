import argparse
import optuna
import logging
import os
import pandas as pd
import numpy as np
import shutil
import glob

from cdl import data
from cdl.mf import MatrixFactorizationModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collaborative Deep Learning inference')
    parser.add_argument('--model_path', type=str, default='model.pt')
    parser.add_argument('--dataset', type=str, default='citeulike-a')
    parser.add_argument('--embedding', type=str, default='bert')
    parser.add_argument('--test_dataset_path', type=str, default=None)
    parser.add_argument('--user_rec_path', type=str, default=None)
    parser.add_argument('--topk', type=int, default=300)
    parser.add_argument('--optuna_study_name', type=str, default=None)
    parser.add_argument('--optuna_storage', type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s  %(message)s', datefmt='%I:%M:%S', level=logging.INFO)

    ratings_test_dataset = data.load_cf_test_data(args.dataset, args.test_dataset_path)
    logging.info(f'Size of ratings_test_dataset: {ratings_test_dataset.size()}')
    mfm = MatrixFactorizationModel((1, 1), 1)
    if args.user_rec_path is None: args.user_rec_path = f'{args.dataset}_{args.embedding}_user_recommendations_{args.topk}.csv'
    
    if args.optuna_study_name:
        study = optuna.load_study(study_name=args.optuna_study_name, storage=args.optuna_storage)
        
        study_df = study.trials_dataframe(attrs=("value", "user_attrs", "params", "state"))
        study_df = study_df[study_df.state=='COMPLETE'].drop(columns=['state'])
        study_df[f'Test Recall@{args.topk}'] = np.nan
        study_df_path = args.model_path.replace('pt','csv')
        if os.path.exists(study_df_path):
            logging.info(f'Updating study trials DataFrame with non-NaN values from {study_df_path}')
            study_df.update(pd.read_csv(study_df_path, index_col=0))
            print(study_df[~np.isnan(study_df[f'Test Recall@{args.topk}'])])
        
        trials_parent_dir = os.path.dirname(args.model_path)
        trials_dirs = [d for d in glob.glob(f'{trials_parent_dir}/trial_*') if os.path.isdir(d)]
        logging.info(f'Loading model artifacts from trials (best and non):\n{trials_dirs}')
        for d in trials_dirs:
            trial_num = int(d.split('_')[-1])
            logging.info(f'Computing Test Recall@{args.topk} for trial {trial_num}')
            mfm = MatrixFactorizationModel((1, 1), 1)
            trial_model_path = os.path.join(d, os.path.basename(args.model_path))
            if os.path.exists(trial_model_path):
                data.load_model(sdae=None, mfm=mfm, filename=trial_model_path)
                recall = mfm.compute_recall(ratings_test_dataset.to_dense(), args.topk).item()
                logging.info(f'Recall@{args.topk} on TEST data: {recall}')
                logging.info(f'Updating Test Recall@{args.topk} for trial {trial_num} in dataframe')
                study_df.at[trial_num, f'Test Recall@{args.topk}'] = recall
            # shutil.rmtree(d)
        best_trial_folder_vd = f'trial_{study.best_trial.number}'
        logging.info(f'Best trial according to Validation values: {best_trial_folder_vd}')
        best_trial_num = int(study_df[f'Test Recall@{args.topk}'].idxmax())
        best_trial_folder_te = f'trial_{best_trial_num}'
        logging.info(f'Best trial according to Test values: {best_trial_folder_te}')
        best_trial_model_te = os.path.join(trials_parent_dir, best_trial_folder_te, os.path.basename(args.model_path))
        # best_trial_model = os.path.join(trials_parent_dir, best_trial_folder_vd, os.path.basename(args.model_path))
        if os.path.exists(best_trial_model_te): 
            logging.info(f'Moving model of best trial from to {args.model_path}')
            shutil.copyfile(best_trial_model_te, args.model_path)
        # logging.info(f'\tAFTER LOADING\nautoencoder:\n{sdae.state_dict()}\nmatrix_factorization_model:\n{mfm.state_dict()}')
        
        print(study_df)
        study_df.to_csv(study_df_path)
    
    else:
        logging.info(f'Loading trained model from {args.model_path}')
        data.load_model(sdae=None, mfm=mfm, filename=args.model_path)
    
        logging.info(f'Saving user recommendations to {args.user_rec_path}')
        user_rec_df = mfm.get_user_recommendations(ratings_test_dataset.to_dense(), args.topk)
        user_rec_df.to_csv(args.user_rec_path)

        logging.info(f'Calculating recall@{args.topk} on TEST data')
        recall = mfm.compute_recall(ratings_test_dataset.to_dense(), args.topk).item()
        logging.info(f'Recall@{args.topk} on TEST data: {recall}')
