import os
import pickle
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from src.JointBertModel import JointBertModel, BERTVectorizer, TagsVectorizer


class ActionClassifier:

    def __init__(self, load_folder_path, sess):
        self.sess = tf.compat.v1.Session()
        self.bert_vectorizer = BERTVectorizer(sess)
        self.tags_vectorizer = TagsVectorizer()
        self.intents_label_encoder = LabelEncoder()
        with open(os.path.join(load_folder_path, 'tags_vectorizer.pkl'), 'rb') as handle:
            self.tags_vectorizer = pickle.load(handle)
        with open(os.path.join(load_folder_path, 'intents_label_encoder.pkl'), 'rb') as handle:
            self.intents_label_encoder = pickle.load(handle)
        self.model = JointBertModel.load_model(load_folder_path, sess)

    def make_prediction(self, utterance):
        intent_slots = self.__predict(utterance)
        if intent_slots['intent']['name'] == 'no_intent':
            response = "i don't understand you"
        else:
            response = self.__construct_phrase(intent_slots)
        return {"intent": intent_slots, "response": response}

    def __construct_phrase(self, intent_slots):
        commands = {'action.switch_on': 'switch on', 'action.switch_off': 'switch off', 'action.set': 'set',
                    'action.open': 'open', 'action.close': 'close', 'action.mute': 'mute', 'action.unmute': 'unmute',
                    'brightness.increase': 'brightness', 'brightness.decrease': 'brightness',
                    'brightness.value': 'brightness', 'color': 'color', 'temperature': 'temperature',
                    'action.increase_temp': 'increase temperature', 'action.decrease_temp': 'decrease temperature',
                    'increase': 'increase', 'decrease': 'decrease', 'sound.decrease': 'sound',
                    'sound.increase': 'sound', 'sound.value': 'sound'}
        res = 'okay, i will '
        words = []
        for slot in intent_slots['slots']:
            if slot['name'] in commands.keys():
                words.append(commands[slot['name']])
        if intent_slots['intent']['name'] == 'air':
            res = 'the air status follows'
        else:
            if len(words) > 0:
                intent = intent_slots['intent']['name']
                if intent == 'implicit_light':
                    intent = 'brightness'
                if intent == 'implicit_conditioner':
                    intent = 'temperature'
                if intent == 'light':
                    apropriate_slots = ['switch on', 'switch off', 'set', 'brightness', 'color']
                    words = [word for word in words if word in apropriate_slots]
                if intent == 'conditioner' and ('increase temperature' in words or 'decrease temperature' in words):
                    res += words[0]
                else:
                    res += words[0] + ' the ' + intent
            if len(words) > 1:
                res += ' ' + words[1]
        return res

    def __predict(self, utterance):
        tokens = utterance.split()
        input_ids, input_mask, segment_ids, valid_positions, data_sequence_lengths = \
            self.bert_vectorizer.transform([utterance])
        predicted_tags, predicted_intents = self.model.predict_slots_intent(
            [input_ids, input_mask, segment_ids, valid_positions],
            self.tags_vectorizer, self.intents_label_encoder, remove_start_end=True,
            include_intent_prob=True)
        slots = self.__fill_slots(predicted_tags[0])
        slots = [{"name": name, "value": ' '.join([tokens[i] for i in slots[name]])} for name in slots.keys()]
        predictions = {
            "intent": {
                "name": predicted_intents[0][0].strip(),
                "confidence": predicted_intents[0][1]
            },
            "slots": slots
        }
        return predictions

    def __fill_slots(self, slots_arr, no_class_tag='O', begin_prefix='B-', in_prefix='I-'):
        slots = {}
        for i, slot in enumerate(slots_arr):
            if slot == no_class_tag:
                continue
            if slot.startswith(begin_prefix):
                name = slot[len(begin_prefix):]
                slots[name] = [i]
            elif slot.startswith(in_prefix):
                name = slot[len(in_prefix):]
                if name in slots.keys():
                    slots[name].append(i)
                else:
                    slots[name] = [i]
        return slots

    def __generate_command(self, intent_slots):
        command = {}
        intent = intent_slots['intent']['name']
        slots = intent_slots['slots']
        command['device'] = intent
        # light
        if intent == 'light':
            for slot in slots:
                if slot['name'] in ['action.switch_on', 'action.switch_off', 'action.set']:
                    command['action'] = slot['name'].split('.')[1]
                if slot['name'] in ['brightness.increase', 'brightness.decrease']:
                    command['action'] = slot['name'].split('.')[1]
                    command['parameter'] = slot['name'].split('.')[0]
                if slot['name'] == 'brightness.value':
                    command['parameter'] = slot['name'].split('.')[0]
                    command['value'] = slot['value']
                if slot['name'] == 'color':
                    command['parameter'] = 'color'
                    command['value'] = slot['value']

        # implicit_light
        if intent == 'implicit_light':
            for slot in slots:
                if slot['name'] in ['increase', 'decrease']:
                    command['action'] = slot['name']
                    command['parameter'] = 'brightness'
                    command['value'] = 'default'

        # conditioner
        if intent == 'conditioner':
            for slot in slots:
                if slot['name'] in ['action.switch_on', 'action.switch_off', 'action.set']:
                    command['action'] = slot['name'].split('.')[1]
                if slot['name'] == 'action.set':
                    command['action'] = 'set'
                    command['parameter'] = 'temperature'
                if slot['name'] in ['action.increase_temp', 'action.decrease_temp']:
                    command['action'] = slot['name'].split('.')[1].split('_')[0]
                    command['parameter'] = 'temperature'
                if slot['name'] == 'temperature':
                    command['parameter'] = 'temperature'
                    command['value'] = slot['value']

        # implicit_conditioner
        if intent == 'implicit_conditioner':
            for slot in slots:
                if slot['name'] in ['increase', 'decrease']:
                    command['action'] = slot['name']
                    command['parameter'] = 'temperature'
                    command['value'] = 'default'

        # curtains
        if intent == 'curtains':
            for slot in slots:
                if slot['name'] in ['open', 'close']:
                    command['action'] = slot['name']

        # tv
        if intent == 'tv':
            for slot in slots:
                if slot['name'] in ['action.switch_on', 'action.switch_off', 'action.set']:
                    command['action'] = slot['name'].split('.')[1]
                if slot['name'] in ['action.mute', 'action.unmute']:
                    command['action'] = slot['name'].split('.')[1]
                if slot['name'] in ['sound.increase', 'sound.decrease']:
                    command['action'] = slot['name'].split('.')[1]
                    command['parameter'] = slot['name'].split('.')[0]
                if slot['name'] == 'sound.value':
                    command['parameter'] = 'sound'
                    command['value'] = slot['value']

        # air
        if intent == 'air':
            for slot in slots:
                if slot['name'] in ['parameter.temperature', 'parameter.humidity', 'parameter.CO2', 'parameter.all']:
                    command['action'] = 'get_info'
                    command['parameter'] = slot['name'].split('.')[1]

        return command
