from cdl import data
from cdl import cdl
from cdl import constants
import logging
import torch
import optuna
import shutil
import glob
import os

class OptunaWrapper:
    def __init__(self, args, sdae, mfm, train_data, valid_data, content_data, recon_loss_fn, activation):
        self.args = args
        self.sdae = sdae
        self.mfm = mfm
        self.train_data = train_data
        self.valid_data = valid_data
        self.content_data = content_data
        self.num_items, self.in_features = self.content_data.shape

        self.recon_loss_fn = constants.RECON_LOSSES[args.recon_loss]
        self.activation = constants.SDAE_ACTIVATIONS[args.activation]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    def get_hyper_combo(self, trial):
        hypers = {
            'lambda_u': trial.suggest_float("lambda_u", 1e-2, 1e2, log=True),
            'lambda_v': trial.suggest_float("lambda_v", 1e-2, 1e2, log=True),
            'lambda_w': trial.suggest_float("lambda_w", 1e-2, 1e2, log=True),
            'lambda_n': trial.suggest_float("lambda_n", 1e-2, 1e2, log=True),
        }
        dangerous_ratio = hypers['lambda_n'] / hypers['lambda_v']
        return hypers, dangerous_ratio < 1e-2 or dangerous_ratio > 1e2

    
    def objective(self, trial):
        config, extreme_case = self.get_hyper_combo(trial)
        if extreme_case:
            raise optuna.exceptions.TrialPruned()
        config.update({ # non-hyper parameters
            'conf_a': self.args.conf_a,
            'conf_b': self.args.conf_b,
            'dropout': self.args.dropout,
            'corruption': self.args.corruption,
        })
        logging.info(f'Config: {config}')

        optimizer = torch.optim.AdamW(self.sdae.parameters(), lr=self.args.lr, weight_decay=config['lambda_w'])

        content_training_dataset = data.random_subset(self.content_data, int(self.num_items * 0.75))
        
        EPOCHS = self.args.epochs # or trial.suggest_int('epochs', 5, 20)
        logging.info(f'Pretraining SDAE with {self.args.recon_loss} loss for {EPOCHS} epochs')
        cdl.train_stacked_autoencoder(self.sdae, content_training_dataset, self.args.corruption, EPOCHS, self.args.batch_size, self.recon_loss_fn, optimizer)
        
        # Train the model
        cdl.train_model(self.sdae, self.mfm, self.content_data, self.train_data, optimizer, self.recon_loss_fn, config, 
                        epochs=EPOCHS, batch_size=self.args.batch_size, device=self.device, trial=trial)
        recall = self.mfm.compute_recall(self.valid_data.to_dense(), self.args.topk).item()
        trial.set_user_attr(f"Validation Recall@{self.args.topk}", recall)
        trial_dir = f'{os.path.dirname(self.args.model_path)}/trial_{trial.number}'
        os.makedirs(trial_dir, exist_ok=True)
        # logging.info(f'\tWHEN SAVING\nautoencoder:\n{self.sdae.state_dict()}\nmatrix_factorization_model:\n{self.mfm.state_dict()}')
        torch.save({'autoencoder': self.sdae.state_dict(),
                    'matrix_factorization_model': self.mfm.state_dict()
                    }, f'{trial_dir}/{os.path.basename(self.args.model_path)}')

        return 1 - recall # Optuna minimizes the objective function, whereas recall should be maximized

    
    def optimize(self, n_trials=10, timeout=600, study_name=None, storage=None):
        study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage, load_if_exists=True)
        
        # study.enqueue_trial({"lambda_n": 0.2560861375505139, "lambda_u": 0.09906732983615665, "lambda_v": 14.116461137607097, "lambda_w": 0.031627597935146495})
        
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        
        # ratings_test_dataset = data.load_cf_test_data(self.args.dataset, self.args.test_dataset_path)
        # logging.info(f'Size of ratings_test_dataset: {ratings_test_dataset.size()}')
        # if self.args.user_rec_path is None: self.args.user_rec_path = f'{self.args.dataset}_{self.args.embedding}_user_recommendations_{self.args.topk}.csv'
        # logging.info(f'Saving user recommendations to {self.args.user_rec_path}')
        # user_rec_df = self.mfm.get_user_recommendations(ratings_test_dataset.to_dense(), self.args.topk)
        # user_rec_df.to_csv(self.args.user_rec_path)
        # logging.info(f'Calculating recall@{self.args.topk} on TEST data')
        # recall = self.mfm.compute_recall(ratings_test_dataset.to_dense(), self.args.topk).item()
        # logging.info(f'Recall@{self.args.topk} on TEST data: {recall}')
        # study_df = study.trials_dataframe(attrs=("value", "user_attrs", "params", "state"))
        # study_df = study_df[study_df.state=='COMPLETE'].drop(columns=['state'])
        # import math
        # assert math.isclose(study_df.loc[study_df.index[-1], 'params_lambda_n'], 0.2560861375505139, abs_tol=0.000003), study_df.loc[study_df.index[-1], 'params_lambda_n']
        # import pandas as pd
        # import numpy as np
        # study_df[f'Test Recall@{self.args.topk}'] = np.nan
        # # study_df.loc[study_df.index[-1], f'Test Recall@{args.topk}'] = 0.25922495126724243
        # study_df_path = self.args.model_path.replace('pt','csv')
        # if os.path.exists(study_df_path):
        #     logging.info(f'Updating study trials DataFrame with non-NaN values from {study_df_path}')
        #     study_df.update(pd.read_csv(study_df_path, index_col=0))
        #     print(study_df[~np.isnan(study_df[f'Test Recall@{self.args.topk}'])])
        # if np.isnan(study_df.at[study_df.index[-1], f'Test Recall@{self.args.topk}']):
        #     logging.info(f'Updating Test Recall@{self.args.topk} for {study_df.index[-1]}')
        #     study_df.at[study_df.index[-1], f'Test Recall@{self.args.topk}'] = recall
        # print(study_df)
        # study_df.to_csv(study_df_path)

        logging.info('Cleaning artifacts from non-best trials')
        trials_parent_dir = os.path.dirname(self.args.model_path)
        best_trial_folder = f'trial_{study.best_trial.number}'
        dirs_to_clean = [d for d in glob.glob(f'{trials_parent_dir}/trial_*') if os.path.isdir(d) and os.path.basename(d)!=best_trial_folder]
        for d in dirs_to_clean: shutil.rmtree(d)
        best_trial_model = os.path.join(trials_parent_dir, best_trial_folder, os.path.basename(self.args.model_path))
        if os.path.exists(best_trial_model): 
            logging.info(f'Moving model from best trial to {self.args.model_path}')
            shutil.move(best_trial_model, self.args.model_path)

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])))
        print("  Number of complete trials: ", len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])))
        print("Best trial:")
        print("  Number: ", study.best_trial.number)
        print(f"  Recall@{self.args.topk} (validation): ", study.best_trial.user_attrs[f"Validation Recall@{self.args.topk}"])
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print("    {}: {}".format(key, value))
        
        return study
