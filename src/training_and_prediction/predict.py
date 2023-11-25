import numpy as np
from src.preprocess import text_prep
from src.preprocess import ner_prep


def predict_intent(text, model, tokenizer):
    preprocessed_text = text_prep.preprocess_for_intent(text, intents=None, tokenizer=tokenizer, train=False)
    probas = model.predict(preprocessed_text)
    return np.argmax(probas)


def predict_entities(text: str, model, tokenizer, label_map: dict, max_seq_length: int):
    # Preprocess the text
    prediction_input = ner_prep.preprocess_for_prediction(text, tokenizer, label_map, max_seq_length)

    # Predict using the model
    prediction_output = model.predict(prediction_input)

    # Extract logits and get the highest probability labels
    logits = prediction_output[0]
    label_indices = np.argmax(logits, axis=-1)

    # Reverse the label_map to get labels from indices
    reverse_label_map = {v: k for k, v in label_map.items()}

    # Tokens and labels
    tokens = tokenizer.tokenize(tokenizer.decode(prediction_input['input_ids'][0]))
    predicted_labels = [reverse_label_map[idx] for idx in label_indices[0][:len(tokens)]]

    # Filter out padding tokens
    token_label_pairs = [(token, label) for token, label in zip(tokens, predicted_labels) if token != '[PAD]']

    return token_label_pairs
