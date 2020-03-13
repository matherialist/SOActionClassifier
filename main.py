import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from os import path
from flask import Flask, request
from src.ActionClassifer import ActionClassifier

app = Flask(__name__)

config_relative_path = "files"
config_path = path.join(path.dirname(__file__), config_relative_path)

graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)
ac = ActionClassifier(config_path, sess)


@app.route('/get-intent', methods=['POST'])
def get_intent():
    with graph.as_default():
        set_session(sess)
        data = request.get_json()
        result = ac.make_prediction(data['text'])
    return result


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
