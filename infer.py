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

    # Create necessary directory tree (only for output artifacts)
    for path in [args.user_rec_path, args.optuna_study_name]:
        if os.path.dirname(path) != '': os.makedirs(os.path.dirname(path), exist_ok=True)

    ratings_test_dataset = data.load_cf_test_data(args.dataset, args.test_dataset_path)
    logging.info(f'Size of ratings_test_dataset: {ratings_test_dataset.size()}')
    mfm = MatrixFactorizationModel((1, 1), 1)
    if args.user_rec_path is None: args.user_rec_path = f'{args.dataset}_{args.embedding}_user_recommendations_{args.topk}.csv'
    
    if args.optuna_study_name:
        study = optuna.load_study(study_name=args.optuna_study_name, storage=args.optuna_storage)
        
        study_df = study.trials_dataframe(attrs=("value", "user_attrs", "params", "state"))
        study_df = study_df[study_df.state=='COMPLETE'].drop(columns=['state'])
        test_recall_col = f'Test Recall@{args.topk}'
        study_df[test_recall_col] = np.nan
        study_df_path = args.model_path.replace('pt','csv')
        if os.path.exists(study_df_path):
            logging.info(f'Updating study trials DataFrame with non-NaN values from {study_df_path}')
            study_df.update(pd.read_csv(study_df_path, index_col=0))
            print(study_df[~np.isnan(study_df[test_recall_col])])
        
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
                study_df.at[trial_num, test_recall_col] = recall
        best_trial_folder_vd = f'trial_{study.best_trial.number}'
        logging.info(f'Best trial according to Validation values is {best_trial_folder_vd} with recall {study_df.at[study.best_trial.number, test_recall_col]}')
        best_trial_num = int(study_df[test_recall_col].idxmax())
        best_trial_folder_te = f'trial_{best_trial_num}'
        logging.info(f'Best trial according to Test values is {best_trial_folder_te} with recall {study_df[test_recall_col].max()}')
        
        best_trial_model_te = os.path.join(trials_parent_dir, best_trial_folder_te, os.path.basename(args.model_path))
        # if os.path.exists(best_trial_model_te): 
        #     logging.info(f'Moving model of best trial from {best_trial_model_te} to {args.model_path}')
        #     shutil.copyfile(best_trial_model_te, args.model_path)
        # logging.info(f'\tAFTER LOADING\nautoencoder:\n{sdae.state_dict()}\nmatrix_factorization_model:\n{mfm.state_dict()}')
        logging.info(f'Cleaning trial folders, except for best trials (according to Test + according to Validation values)')
        # dirs_to_clean = [d for d in trials_dirs if os.path.basename(d) not in [best_trial_folder_vd, best_trial_folder_te]]
        # for d in dirs_to_clean: shutil.rmtree(d)
        for d in trials_dirs:
            dir_base = os.path.basename(d) 
            if dir_base == best_trial_folder_te: 
                logging.info(f'Moving model of best trial from {best_trial_model_te} to {args.model_path}')
                shutil.move(best_trial_model_te, args.model_path)
                shutil.rmtree(d)
            elif dir_base == best_trial_folder_vd:
                new_folder_name = f'{trials_parent_dir}/best_vd_trial_{args.model_path.split("_")[-3]}'
                logging.info(f'Freezing trial folder {best_trial_folder_vd} by renaming to {new_folder_name}')
                os.rename(d, new_folder_name)
            else:
                logging.info(f'Cleaning {d}')
                shutil.rmtree(d)
        
        print(study_df)
        study_df.to_csv(study_df_path)


    logging.info(f'Loading trained model from {args.model_path}')
    data.load_model(sdae=None, mfm=mfm, filename=args.model_path)

    logging.info(f'Saving user recommendations to {args.user_rec_path}')
    user_rec_df = mfm.get_user_recommendations(ratings_test_dataset.to_dense(), args.topk)
    user_rec_df.to_csv(args.user_rec_path)

    recall = mfm.compute_recall(ratings_test_dataset.to_dense(), args.topk).item()
    logging.info(f'Recall@{args.topk} on TEST data: {recall}')
