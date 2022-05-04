import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import DefaultDataCollator

from utils import set_seed


def load_data(
    filename: str,
    tokenizer,
    task_id,
    batch_size: int = 64,
    seed: int = 42,
    train_frac: float = 0.9,
    val_frac: float = 0.1,
    single_batch: bool = False,
    ):
    """
    Load data from filename, extracting a single task, splitting into
    training, validation and test data, and putting it in a format that is
    compatible with tensorflow based transformers models.
    """
    carboxylics_frame = pd.read_csv(filename, index_col='Unnamed: 0')
    all_tasks = list(carboxylics_frame.columns[2:])
    task = all_tasks[task_id]
    carboxylics_frame = carboxylics_frame[['smiles', task]]

    carboxylics_frame=carboxylics_frame.dropna(axis=0) #Delete rows containing any Nan(s)
    carboxylics_frame[task]=carboxylics_frame[task] / 180. #normalizing the cone angles

    df = carboxylics_frame[['smiles', task]]
    df = df.rename(columns={task: 'label', 'smiles': 'text'})

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.map(tokenize, batched=True, batch_size=None)

    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    datasets = split_dataset(dataset, train_frac, val_frac, seed)

    if single_batch:
        datasets = DatasetDict({key: _first_batch(val, batch_size)
            for key, val in datasets.items()})

    data_collator = DefaultDataCollator(return_tensors='tf')
    set_seed(seed)
    tf_dataset = {k: v.to_tf_dataset(
                            columns=tokenizer.model_input_names,
                            label_cols=['label'],
                            shuffle=True,
                            batch_size=batch_size,
                            collate_fn=data_collator)
                    for k, v in datasets.items()
    }
    return tf_dataset

def _first_batch(dataset, batch_size):
    """Take single batch from arrow_dataset."""
    return Dataset.from_dict(dataset[:batch_size])

def split_dataset(dataset, train_frac, val_frac, seed):
    """Split dataset into train, validation and test splits with specified ratios."""
    train_testvalid = dataset.train_test_split(test_size=(1 - train_frac), seed=seed)

    return DatasetDict({
        'train': train_testvalid['train'],
        'val': train_testvalid['test']
    })

