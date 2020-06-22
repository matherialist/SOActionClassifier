from os import path
from flask import Flask, request
from src.ActionClassifer import ActionClassifier
from src.JointBertModel import JointBertModel

app = Flask(__name__)

config_relative_path = 'files'
config_path = path.join(path.dirname(__file__), config_relative_path)

ac = ActionClassifier(config_path, model_hub_path="https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2", is_bert=True)


@app.route('/get-intent', methods=['POST'])
def get_intent():
    data = request.get_json()
    address = request.remote_addr
    result = ac.make_prediction(data['text'])
        # if result['command']['device'] == 'timer':
        #     tr = TimerReminder()
        #     tr.set_timer(result['command']['value'], address)
        # elif result['command']['device'] == 'reminder':
        #     tr = TimerReminder()
        #     tr.set_reminder(result['command']['value'], address, result['command']['value'])
    return result


# @app.route('/train', methods=['POST'])
# def train():
#     with graph.as_default():
#         set_session(sess)
#         model = JointBertModel.train(config_path, sess)
#         f1_score, acc = model.evaluate('data', sess)
#     return {"f1_score": f1_score, "acc": acc}


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
