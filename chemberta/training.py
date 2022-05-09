from transformers import AdamWeightDecay
import tensorflow as tf
from tensorflow import keras

from data_loader import load_data
from model_loader import load_model_and_tokenizer


def fine_tune(
        checkpoint: str,
        filename: str,
        task_id: int,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        decay_rate: float,
        callbacks,
        single_batch: bool = False,
        classification: bool = False,
        freeze_base: bool = False,
    ):

    model, datasets = prepare_training(
        checkpoint, filename, task_id, batch_size, learning_rate, weight_decay,
        decay_rate, single_batch, classification, freeze_base
    )

    model.fit(
        datasets['train'],
        validation_data=datasets['val'],
        epochs=epochs,
        callbacks=callbacks,
    )

def prepare_training(
        checkpoint: str,
        filename: str,
        task_id: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        decay_rate: float,
        single_batch: bool,
        classification: bool,
        freeze_base: bool,
        **kwargs,
    ):

    model, tokenizer = load_model_and_tokenizer(
            checkpoint,
            classification=classification,
            freeze_base=freeze_base,
            )

    datasets = load_data(
        filename=filename,
        tokenizer=tokenizer,
        task_id=task_id,
        batch_size=batch_size,
        single_batch=single_batch,
        classification=classification,
    )

    optimizer = create_optimizer(learning_rate, decay_rate, weight_decay)

    if classification:
        metric = tf.keras.metrics.Accuracy()
    else:
        metric = tf.keras.metrics.RootMeanSquaredError()

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[metric],
    )

    return model, datasets

def create_optimizer(learning_rate, decay_rate, weight_decay):
    schedule = keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=1,  # every epoch
        decay_rate=decay_rate,
    )
    return  AdamWeightDecay(
        learning_rate=schedule,
        weight_decay_rate=weight_decay,
    )


def main():
    fine_tune(
        filename='../data/cone_angle_carbox_11k.csv',
        checkpoint='DeepChem/ChemBERTa-77M-MTR',
        task_id=0,
        batch_size=512,
        epochs=100,
        learning_rate=3e-4,
        decay_rate=1.,
        weight_decay=0.,
        callbacks=[],
        single_batch=True,
        classification=False,
        freeze_base=True,
    )

if __name__ == '__main__':
    main()
