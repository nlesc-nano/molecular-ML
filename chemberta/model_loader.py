from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from utils import set_seed


def load_model_and_tokenizer(
        checkpoint,
        seed: int = 42,
        num_labels: int = 1,
        from_pt: bool = True,
        classification: bool = False,
        ):
    set_seed(seed)
    problem_type = 'classification' if classification else 'regression'
    model = TFAutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        problem_type=problem_type,
        from_pt=from_pt  # original model is in pytorch
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    return model, tokenizer
