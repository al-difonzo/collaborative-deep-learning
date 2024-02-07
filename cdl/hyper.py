from cdl import data
from cdl import cdl
from cdl import constants
import logging
import torch
import optuna
import shutil

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
        return {
            'lambda_u': trial.suggest_float("lambda_u", 1e-2, 1e4, log=True),
            'lambda_v': trial.suggest_float("lambda_v", 1e-2, 1e4, log=True),
            'lambda_w': trial.suggest_float("lambda_w", 1e-2, 1e4, log=True),
            'lambda_n': trial.suggest_float("lambda_n", 1e-2, 1e4, log=True),
        }

    def objective(self, trial):
        config = self.get_hyper_combo(trial)
        config.update({ # non-hyper parameters
            'conf_a': self.args.conf_a,
            'conf_b': self.args.conf_b,
            'dropout': self.args.dropout,
            'corruption': self.args.corruption,
        })
        logging.info(f'Config: {config}')

        optimizer = torch.optim.AdamW(self.sdae.parameters(), lr=self.args.lr, weight_decay=config['lambda_w'])

        content_training_dataset = data.random_subset(self.content_data, int(self.num_items * 0.75))
        
        # EPOCHS = trial.suggest_int('epochs', 5, 20)
        EPOCHS = 2
        logging.info(f'Pretraining SDAE with {self.args.recon_loss} loss for {EPOCHS} epochs')
        cdl.train_stacked_autoencoder(self.sdae, content_training_dataset, self.args.corruption, EPOCHS, self.args.batch_size, self.recon_loss_fn, optimizer)
        
        # Train the model
        cdl.train_model(self.sdae, self.mfm, self.content_data, self.train_data, optimizer, self.recon_loss_fn, config, epochs=EPOCHS, batch_size=self.args.batch_size, device=self.device)
        recall = self.mfm.compute_recall(self.valid_data.to_dense(), self.args.topk).item()
        trial.set_user_attr(f"Recall@{self.args.topk}", recall)
        trial_dir = f'{os.path.dirname(self.args.out_model_path)}/trial_{trial.number}'
        os.makedirs(trial_dir, exist_ok=True)
        torch.save({'autoencoder': self.sdae.state_dict(),
                    'matrix_factorization_model': self.mfm.state_dict()
                    }, f'{trial_dir}/{os.path.basename(self.args.out_model_path)}')
        # trial.set_user_attr("sdae", self.sdae)
        # trial.set_user_attr("mfm", self.mfm)
        
        # for epoch in range(EPOCHS):
        #     logging.info(f'Training with recon loss {self.args.recon_loss}')
        #     # Evaluate the model on validation set
        #     recall = self.mfm.compute_recall(self.valid_data.to_dense(), self.args.topk)

        #     trial.report(recall, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return 1 - recall  # Optuna minimizes the objective function, whereas recall should be maximized

    def optimize(self, n_trials=10, study_name=None, storage=None):
        study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage, load_if_exists=True)
        study.optimize(self.objective, n_trials=n_trials, timeout=1800) # timeouts after 30 minutes, if not yet stopped due to n_trials

        # Clean artifacts from non-best trials
        trials_parent_dir = os.path.dirname(self.args.out_model_path)
        best_trial_folder = f'trial_{study.best_trial.number}'
        dirs_to_clean = [d for d in os.listdir(trials_parent_dir) if os.path.isdir() and d.startswith('trial_') and d != best_trial_folder]
        for d in dirs_to_clean: shutil.rmtree(os.path.join(trials_parent_dir, d))
        # Move artifacts of best trial to args.out_model_path
        shutil.move(os.path.join(trials_parent_dir, best_trial_folder, '*'), args.out_model_path)

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])))
        print("  Number of complete trials: ", len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])))

        print("Best trial:")
        print("  Value: ", study.best_trial.value)
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print("    {}: {}".format(key, value))
        
        return study
