from os import path
from src.JointBertModel import JointBertModel

config_path = path.join(path.dirname(__file__), 'files')

model = JointBertModel.train_model(config_path, is_bert=True)
f1_score, acc = model.evaluate_model(config_path, is_bert=True)
