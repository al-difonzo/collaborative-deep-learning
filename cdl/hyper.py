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
            'lambda_u': trial.suggest_float("lambda_u", 2e-2, 2e1, log=True),
            'lambda_v': trial.suggest_float("lambda_v", 2e-2, 2e1, log=True),
            'lambda_w': trial.suggest_float("lambda_w", 2e-2, 2e1, log=True),
            'lambda_n': trial.suggest_float("lambda_n", 2e-2, 2e1, log=True),
        }
        dangerous_ratio = hypers['lambda_n'] / hypers['lambda_v']
        return hypers, dangerous_ratio < 1e-2 or dangerous_ratio > 1e2

    
    def objective(self, trial):
        config, extreme_case = self.get_hyper_combo(trial)
        config.update({ # non-hyper parameters
            'conf_a': self.args.conf_a,
            'conf_b': self.args.conf_b,
            'dropout': self.args.dropout,
            'corruption': self.args.corruption,
        })
        logging.info(f'Config: {config}')
        if extreme_case: raise optuna.exceptions.TrialPruned()

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
        
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        
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
