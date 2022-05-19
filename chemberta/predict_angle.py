import pandas as pd
import h5py
import os

import wandb
from datasets import Dataset
from transformers import DefaultDataCollator
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
import tensorflow as tf

from model_loader import load_model_and_tokenizer
from model_evaluation import get_best_sweep, restore_model_wandb

MODEL_CHECKPOINT = 'DeepChem/ChemBERTa-77M-MTR'
SWEEP_ID  = 'apjansen/chemberta/5pw5mcyz/'
DATA_FILE = '../data/carbox_fulldatabase.csv'
TOKENIZED_DATA = '../data/carbox_full_tokenized'
PREDICTIONS_PATH = '../data/carbox_predictions.hdf5'
BATCH_SIZE = 512


def load_best_model(model_checkpoint, sweep_id):
    model, tokenizer = load_model_and_tokenizer(model_checkpoint)
    best_run_id = SWEEP_ID + get_best_sweep(sweep_id)
    model = restore_model_wandb(best_run_id, model)
    return model, tokenizer

def load_test_data(df, tokenizer, batch_size):
    def tokenize(batch):
        return tokenizer(batch['name'], padding=True, truncation=True)

    dataset = Dataset.from_pandas(df, preserve_index=False)
    print("Tokenizing SMILES strings...")
    dataset = dataset.map(tokenize, batched=True, batch_size=batch_size)
    dataset.set_format('tensorflow', columns=tokenizer.model_input_names)

    data_collator = DefaultDataCollator(return_tensors='tf')
    tf_dataset = dataset.to_tf_dataset(
                            columns=tokenizer.model_input_names,
                            shuffle=False,
                            batch_size=batch_size,
                            collate_fn=data_collator)
    return tf_dataset

def predict_to_file(model, dataset, filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

    f = h5py.File(filepath, 'a')
    name = 'predicted_cone_angle'
    f.create_dataset(name, shape=(0, 1), maxshape=(None, 1))
    for i, batch in enumerate(dataset):
        preds = model.predict_on_batch(batch).logits
        f[name].resize(len(f[name]) + preds.shape[0], axis=0)
        f[name][-preds.shape[0]:] = preds
        if i % 10 == 0:
            print(f'Batch {i} out of {len(dataset)}...')

    return f


def main():
    model, tokenizer = load_best_model(MODEL_CHECKPOINT, SWEEP_ID)
    def tokenize(batch):
        return tokenizer(batch, padding=True, truncation=True)

    df = pd.read_csv(DATA_FILE)
    dataset = load_test_data(df, tokenizer, BATCH_SIZE)
    tf.data.experimental.save(dataset, TOKENIZED_DATA)

    print("Computing predictions...")
    f = predict_to_file(model, dataset, PREDICTIONS_PATH)
    # predictions = model.predict(
    #         dataset,
    #         batch_size=BATCH_SIZE,
    #         )

    df['predicted_cone_angle'] = f['predicted_cone_angle'][:]

    df.to_csv('../data/carbox_predictions.csv')


if __name__ == '__main__':
    main()

