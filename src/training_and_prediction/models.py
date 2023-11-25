import tensorflow as tf
from transformers import TFDistilBertModel, BertConfig, TFBertForTokenClassification
from tensorflow import keras
from src.preprocess import ner_prep


def create_intent_classifier(compile=False, init_lr=0.00001, dropout=0.3):
    # Define input layers
    input_ids = tf.keras.layers.Input(shape=(40,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(40,), dtype=tf.int32, name='attention_mask')

    # Load the DistilBERT model
    # Ensure that you have already loaded or downloaded the model as needed
    model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

    # Use the DistilBERT model
    distilbert_output = model(input_ids, attention_mask=attention_mask)[0]

    # Get the output for the [CLS] token (first token)
    pooled_output = distilbert_output[:, 0]

    # Additional dropout layer for regularization
    dropout = tf.keras.layers.Dropout(dropout)(pooled_output)

    # Classifier layer for your 3 classes
    classifier = tf.keras.layers.Dense(3, activation='softmax')(dropout)

    # Final model
    final_model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=classifier)

    if compile:
        optimizer = keras.optimizers.Adam(learning_rate=init_lr)
        final_model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), ner_prep.F1Score()])

    return final_model


def trained_intent_classifier():
    model = create_intent_classifier()
    model.load_weights('resources/bert/saved/ir_trained_weights.h5')
    return model


def create_entity_classifier(compile=False, lr=5e-5):
    num_labels = 15
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model = TFBertForTokenClassification.from_pretrained('bert-base-uncased', config=config)

    if compile:
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        loss = keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = [keras.metrics.Precision(), keras.metrics.Recall(), ner_prep.F1Score()]
        model.compile(optimizer, loss=loss, metrics=metrics)

    return model


def trained_entity_classifier():
    model = create_entity_classifier()
    model.load_weights('resources/bert/saved/ner_trained_weights.h5')
    return model
