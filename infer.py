import argparse
import optuna
import logging

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

    mfm = MatrixFactorizationModel((1, 1), 1)
    logging.info(f'Loading trained model from {args.model_path}')
    data.load_model(sdae=None, mfm=mfm, filename=args.model_path)
    # logging.info(f'\tAFTER LOADING\nautoencoder:\n{sdae.state_dict()}\nmatrix_factorization_model:\n{mfm.state_dict()}')
    
    ratings_test_dataset = data.load_cf_test_data(args.dataset, args.test_dataset_path)
    logging.info(f'Size of ratings_test_dataset: {ratings_test_dataset.size()}')
    if args.user_rec_path is None: args.user_rec_path = f'{args.dataset}_{args.embedding}_user_recommendations_{args.topk}.csv'
    logging.info(f'Saving user recommendations to {args.user_rec_path}')
    user_rec_df = mfm.get_user_recommendations(ratings_test_dataset.to_dense(), args.topk)
    user_rec_df.to_csv(args.user_rec_path)

    logging.info(f'Calculating recall@{args.topk} on TEST data')
    recall = mfm.compute_recall(ratings_test_dataset.to_dense(), args.topk).item()
    logging.info(f'Recall@{args.topk} on TEST data: {recall}')
    if args.optuna_study_name:
        study = optuna.load_study(study_name=args.optuna_study_name, storage=args.optuna_storage)
        study_df = study.trials_dataframe(attrs=("value", "user_attrs", "params", "state"))
        study_df = study_df[study_df.state=='COMPLETE'].drop(columns=['state'])
        study_df.loc[study.best_trial.number, f'Test Recall@{args.topk}'] = recall
        print(study_df)
        study_df.to_csv(args.model_path.replace('pt','csv'))