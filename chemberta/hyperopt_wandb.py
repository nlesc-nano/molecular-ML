from training import fine_tune

import wandb

# to turn off warning that model needs to be pretrained (and other warnings..)
from transformers import logging
logging.set_verbosity_error()

MODEL_CHECKPOINT = 'DeepChem/ChemBERTa-77M-MTR'
WANDB_USERNAME = 'apjansen'
NUM_RUNS = 25
FILENAME = '../data/all_carboxylics.csv'
  
def train(config=None):
    """Wrapper to pass to wandb agent."""
    with wandb.init(config=config):
        config = wandb.config
        
        wandb_callback = wandb.keras.WandbCallback(
            monitor='val_root_mean_squared_error',
            save_model=True,
            save_weights_only=True,
        )

        fine_tune(**config, wandb_callback=wandb_callback)

        return


sweep_config = {'method': 'bayes'}
sweep_config['metric'] = {'name': 'best_val_root_mean_squared_error', 'goal': 'minimize'}

sweep_config['parameters'] = {
    'checkpoint': {'value': MODEL_CHECKPOINT},
    'filename': {'value': FILENAME},
    'task_id': {'value': 0},
    'epochs': {'value': 20},
    'learning_rate': {'values': [3e-6, 1e-5, 2e-5, 5e-5, 1e-4]},
    'decay_rate': {'values': [1., .95, .9]},  # each epoch decay lr by this factor
    'batch_size': {'values': [64, 128, 256]}, # batch size 512 can't be handled by colab
    'weight_decay': {'value': 0.},
    'patience': {'value': 5},
}

def main():
    sweep_id = wandb.sweep(
        sweep_config,
        project='chemberta',
        entity=WANDB_USERNAME)
    wandb.agent(sweep_id, train, count=NUM_RUNS)

if __name__ == '__main__':
    main()
