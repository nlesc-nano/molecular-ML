from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from utils import set_seed


def load_model_and_tokenizer(
        checkpoint,
        seed: int = 42,
        num_labels: int = 1,
        from_pt: bool = True):
    set_seed(seed)

    model = TFAutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        problem_type='regression',
        from_pt=from_pt  # original model is in pytorch
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    return model, tokenizer
