from transformers import AdamWeightDecay
import tensorflow as tf
from tensorflow import keras

from data_loader import load_data
from model_loader import load_model_and_tokenizer

import wandb



def fine_tune(
        checkpoint: str,
        filename: str,
        task_id: int,
        batch_size: int,
        patience: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        decay_rate: float,
        wandb_callback=None):

    model, tokenizer = load_model_and_tokenizer(checkpoint)

    datasets = load_data(
        filename=filename,
        tokenizer=tokenizer,
        task_id=task_id,
        batch_size=batch_size)

    num_batches = len(datasets['train'])  # use to decay per epoch
    schedule = keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=num_batches,
        decay_rate=decay_rate,
    )
    optimizer = AdamWeightDecay(
        learning_rate=schedule,
        weight_decay_rate=weight_decay)

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=patience, restore_best_weights=True),
        LRLogger(optimizer),
    ]
    if wandb_callback:
        callbacks.append(wandb_callback)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

    model.fit(
        datasets['train'], 
        validation_data=datasets['val'],
        epochs=epochs,
        callbacks=callbacks
    )

    return model

# to log the learning rate
class LRLogger(keras.callbacks.Callback):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs):
        lr = self.optimizer.learning_rate(epoch)
        wandb.log({"lr": lr}, commit=False)

def main():
    FILENAME = '../data/all_carboxylics.csv'
    CHECKPOINT = 'DeepChem/ChemBERTa-77M-MTR'
    model = fine_tune(
        checkpoint=CHECKPOINT,
        filename=FILENAME,
        task_id=0,
        batch_size=32,
        patience=3,
        epochs=1,
        learning_rate=1e-3,
        decay_rate=1.,
        weight_decay=0.,
    )
    print('fine tuned a model')

if __name__ == '__main__':
    main()
