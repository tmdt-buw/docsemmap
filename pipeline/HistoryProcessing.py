import codecs
import json
import os


class HistoryProcessing:
    HISTORY_PATH = '../data/vc-slam/history'

    mappings = None

    def __init__(self, id):
        self.mappings = {}
        self.load_history(id)

    def reset(self):
        self.mappings = None

    def load_history(self, id):
        filename = os.path.join(self.HISTORY_PATH, 'history_' + id + '.json')
        with codecs.open(filename, encoding='utf8', mode='r') as f:
            self.mappings = json.load(f)

    def get_history_mapping(self, attribute):
        if attribute in self.mappings:
            return [[att, 1] for att in self.mappings[attribute]]
        return []
