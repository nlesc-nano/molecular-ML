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
SWEEP_ID  = 'apjansen/chemberta/5pw5mcyz/'
DATA_FILE = '../data/carbox_fulldatabase.csv'


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
    df = df.head(1_000)
    dataset = load_test_data(df, tokenizer)

    print("Computing predictions...")
    predictions = model.predict(
            dataset,
            batch_size=256,
            )

    df['predicted_cone_angle'] = predictions['logits']

    df.to_csv('../data/carbox_predictions.csv')


if __name__ == '__main__':
    main()

