import tensorflow as tf
from os import path
from src.JointBertModel import JointBertModel

config_path = path.join(path.dirname(__file__), 'files')

graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)


model = JointBertModel.train_model(config_path, sess)
f1_score, acc = model.evaluate_model(config_path, sess)
