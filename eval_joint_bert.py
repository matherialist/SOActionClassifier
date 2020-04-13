from src.JointBertModel import BERTVectorizer
from src.JointBertModel import JointBertModel
import os
import pickle
import tensorflow as tf
from sklearn import metrics
from itertools import chain


def flatten(y):
    return list(chain.from_iterable(y))


def read_goo(dataset_folder_path):
    with open(os.path.join(dataset_folder_path, 'label'), encoding='utf-8') as f:
        labels = f.readlines()

    with open(os.path.join(dataset_folder_path, 'seq.in'), encoding='utf-8') as f:
        text_arr = f.readlines()

    with open(os.path.join(dataset_folder_path, 'seq.out'), encoding='utf-8') as f:
        tags_arr = f.readlines()

    assert len(text_arr) == len(tags_arr) == len(labels)
    return text_arr, tags_arr, labels


load_folder_path = 'files'
data_folder_path = 'data/test'
batch_size = 128


sess = tf.compat.v1.Session()

bert_model_hub_path = 'https://tfhub.dev/google/albert_base/1'

    
bert_vectorizer = BERTVectorizer(sess, bert_model_hub_path)

# loading models
print('Loading models ...')
if not os.path.exists(load_folder_path):
    print('Folder `%s` not exist' % load_folder_path)

with open(os.path.join(load_folder_path, 'tags_vectorizer.pkl'), 'rb') as handle:
    tags_vectorizer = pickle.load(handle)
    slots_num = len(tags_vectorizer.label_encoder.classes_)
with open(os.path.join(load_folder_path, 'intents_label_encoder.pkl'), 'rb') as handle:
    intents_label_encoder = pickle.load(handle)
    intents_num = len(intents_label_encoder.classes_)
    
model = JointBertModel.load(load_folder_path, sess)


data_text_arr, data_tags_arr, data_intents = read_goo(data_folder_path)
data_input_ids, data_input_mask, data_segment_ids, data_valid_positions, data_sequence_lengths = bert_vectorizer.transform(data_text_arr)


def get_results(input_ids, input_mask, segment_ids, valid_positions, sequence_lengths, tags_arr, 
                intents, tags_vectorizer, intents_label_encoder):
    predicted_tags, predicted_intents = model.predict_slots_intent(
            [input_ids, input_mask, segment_ids, valid_positions], 
            tags_vectorizer, intents_label_encoder, remove_start_end=True)
    gold_tags = [x.split() for x in tags_arr]
    print(metrics.classification_report(flatten(gold_tags), flatten(predicted_tags), digits=3))
    f1_score = metrics.f1_score(flatten(gold_tags), flatten(predicted_tags), average='micro')
    acc = metrics.accuracy_score(intents, predicted_intents)
    return f1_score, acc


print('==== Evaluation ====')
f1_score, acc = get_results(data_input_ids, data_input_mask, data_segment_ids, data_valid_positions,
                            data_sequence_lengths, data_tags_arr, data_intents, tags_vectorizer, intents_label_encoder)
print('Slot f1_score = %f' % f1_score)
print('Intent accuracy = %f' % acc)

tf.compat.v1.reset_default_graph()
