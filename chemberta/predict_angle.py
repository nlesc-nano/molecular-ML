import pandas as pd
import wandb
from datasets import Dataset
from transformers import DefaultDataCollator
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
import tensorflow as tf

from model_loader import load_model_and_tokenizer
from model_evaluation import get_best_sweep, restore_model_wandb

MODEL_CHECKPOINT = 'DeepChem/ChemBERTa-77M-MTR'
SWEEP_ID  = 'apjansen/chemberta/5ble16ck/'
DATA_FILE = '../data/carbox_fulldatabase.csv'


class TQDMPredictCallback(Callback):
    def __init__(self, custom_tqdm_instance=None, tqdm_cls=tqdm, **tqdm_params):
        super().__init__()
        self.tqdm_cls = tqdm_cls
        self.tqdm_progress = None
        self.prev_predict_batch = None
        self.custom_tqdm_instance = custom_tqdm_instance
        self.tqdm_params = tqdm_params

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        self.tqdm_progress.update(batch - self.prev_predict_batch)
        self.prev_predict_batch = batch

    def on_predict_begin(self, logs=None):
        self.prev_predict_batch = 0
        if self.custom_tqdm_instance:
            self.tqdm_progress = self.custom_tqdm_instance
            return

        total = self.params.get('steps')
        if total:
            total -= 1

        self.tqdm_progress = self.tqdm_cls(total=total, **self.tqdm_params)

    def on_predict_end(self, logs=None):
        if self.tqdm_progress and not self.custom_tqdm_instance:
            self.tqdm_progress.close()

def load_best_model(model_checkpoint, sweep_id):
    model, tokenizer = load_model_and_tokenizer(model_checkpoint)
    best_run_id = SWEEP_ID + get_best_sweep(sweep_id)
    model = restore_model_wandb(best_run_id, model)
    return model, tokenizer

def load_test_data(df, tokenizer, batch_size=2024):
    def tokenize(batch):
        return tokenizer(batch['name'], padding=True, truncation=True)

    dataset = Dataset.from_pandas(df, preserve_index=False)
    print("Tokenizing SMILES strings...")
    dataset = dataset.map(tokenize, batched=True, batch_size=batch_size)
    dataset.set_format('torch', columns=tokenizer.model_input_names)

    data_collator = DefaultDataCollator(return_tensors='tf')
    tf_dataset = dataset.to_tf_dataset(
                            columns=tokenizer.model_input_names,
                            shuffle=False,
                            batch_size=batch_size,
                            collate_fn=data_collator)
    return tf_dataset

def main():
    model, tokenizer = load_best_model(MODEL_CHECKPOINT, SWEEP_ID)

    df = pd.read_csv(DATA_FILE)
    df = df.head(1_700)
    dataset = load_test_data(df, tokenizer)

    model.compile(
        optimizer='Adam',
        loss=tf.keras.losses.MeanSquaredError(),
    )

    print("Computing predictions...")
    predictions = model.predict(
            dataset,
            verbose=2,
            batch_size=512,
            #callbacks=[TQDMPredictCallback()],
            )

    df['predicted_cone_angle'] = predictions['logits']

    df.to_csv('../data/carbox_predictions.csv')


if __name__ == '__main__':
    main()

