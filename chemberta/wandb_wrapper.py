import wandb
from tensorflow import keras
from training import prepare_training

# to turn off warning that model needs to be pretrained (and other warnings..)
from transformers import logging
logging.set_verbosity_error()

class LRLogger(keras.callbacks.Callback):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs):
        lr = self.optimizer.learning_rate(epoch)
        wandb.log({"lr": lr}, commit=False)


def train(config=None):
    """Wrapper to pass to wandb agent."""
    with wandb.init(config=config):
        config = wandb.config

        model, datasets = prepare_training(**config)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.patience,
                restore_best_weights=True,
            ),
            LRLogger(model.optimizer),
            wandb.keras.WandbCallback(
                monitor='val_root_mean_squared_error',
                save_model=True,
                save_weights_only=True,
            ),
        ]

        model.fit(
            datasets['train'], 
            validation_data=datasets['val'],
            epochs=config.epochs,
            callbacks=callbacks,
        )

def main(config):
    train(config)


if __name__ == '__main__':
    defaults = {
        'checkpoint': 'DeepChem/ChemBERTa-77M-MTR',
        'filename':  '../data/all_carboxylics.csv',
        'task_id': 0,
        'epochs': 3,
        'learning_rate': 1e-4,
        'decay_rate': 1.,
        'batch_size': 256,
        'weight_decay': 0.,
        'patience': 5
    }
    main(config=defaults)
