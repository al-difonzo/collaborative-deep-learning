from cdl import data
from cdl import constants
import logging
import torch

class OptunaWrapper:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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

    def get_hyper_combo(self, trial):
        return {
            'lambda_u': trial.suggest_float("lambda_u", 1e-2, 1e4, log=True),
            'lambda_v': trial.suggest_float("lambda_v", 1e-2, 1e4, log=True),
            'lambda_w': trial.suggest_float("lambda_w", 1e-2, 1e4, log=True),
            'lambda_n': trial.suggest_float("lambda_n", 1e-2, 1e4, log=True),
        }

    def objective(self, trial):
        config = self.get_hyper_combo(trial)
        logging.info(f'Config: {config}')
        
        optimizer = optim.AdamW(self.sdae.parameters(), lr=config['lr'], weight_decay=config['lambda_w'])

        content_training_dataset = data.random_subset(self.content_data, int(self.num_items * 0.75))

        logging.info(f'Pretraining SDAE with {self.args.recon_loss} loss')
        train_stacked_autoencoder(self.sdae, content_training_dataset, self.args.corruption, self.args.pretrain_epochs, self.args.batch_size, self.recon_loss_fn, optimizer)

        for epoch in range(trial.suggest_int('epochs', 5, 20)):
            # Train the model
            logging.info(f'Training with recon loss {self.args.recon_loss}')
            train_model(self.sdae, self.mfm, self.content_data, self.train_data, optimizer, self.recon_loss_fn, config, epochs=self.args.epochs, batch_size=self.args.batch_size, device=device)
            # Evaluate the model on the validation set
            recall = self.mfm.compute_recall(self.valid_data.to_dense(), self.args.topk)

            trial.report(recall, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return 1 - recall  # Optuna minimizes the objective function, and higher is better for recall

    def optimize(self, n_trials=10, study_name=None):
        study = optuna.create_study(direction='minimize', study_name=study_name)
        study.optimize(self.objective, n_trials=n_trials)

        # Print the best hyperparameters
        best_params = study.best_params
        logging.info("Best Hyperparameters:", best_params)
        
        return study
