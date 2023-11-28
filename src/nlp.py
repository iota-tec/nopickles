import pickle
import threading
from openai import OpenAI
from src.training_and_prediction import predict, models

bert_intent_tokenizer, bert_intent_model, bert_ner_tokenizer, bert_ner_model, label_map = None, None, None, None, None


# Function to load NER model and tokenizer
def load_ner_model():
    global bert_ner_tokenizer, bert_ner_model, label_map
    with open('resources/bert/saved/ner_tokenizer.pkl', 'rb') as tn:
        bert_ner_tokenizer = pickle.load(tn)
    with open('resources/bert/data/label_map.pkl', 'rb') as lm:
        label_map = pickle.load(lm)
    bert_ner_model = models.trained_entity_classifier()
    bert_ner_model.load_weights('resources/bert/saved/ner_trained_weights.h5')


# Function to load Intent model and tokenizer
def load_intent_model():
    global bert_intent_tokenizer, bert_intent_model
    with open('resources/bert/saved/intent_tokenizer.pkl', 'rb') as ti:
        bert_intent_tokenizer = pickle.load(ti)
    bert_intent_model = models.trained_intent_classifier()
    bert_intent_model.load_weights('resources/bert/saved/ir_trained_weights.h5')


OPENAI_API_KEY = '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'
OPENAI_JOB = "ftjob-NOjJ5NxYigdba5FCHz8GXwQo"
GPT3_MODEL = "ft:gpt-3.5-turbo-0613:personal::8PhccnUL"

client = OpenAI(api_key=OPENAI_API_KEY)
completion = client.fine_tuning.jobs.retrieve(OPENAI_JOB)

# Creating threads
thread1 = threading.Thread(target=load_ner_model)
thread2 = threading.Thread(target=load_intent_model)

# Starting threads
thread1.start()
thread2.start()

# Joining threads back to the main thread
thread1.join()
thread2.join()

