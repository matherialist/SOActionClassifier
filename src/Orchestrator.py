from src.ActionClassifer import ActionClassifier
from src.SpeechToText import SpeechToText
from src.TextToSpeech import TextToSpeech


class Orchestrator:

    def __init__(self):
        self.action_classifier = ActionClassifier("files")
