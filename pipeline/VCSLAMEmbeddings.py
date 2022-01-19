import gensim
import codecs
import json

class VCSLAMEmbeddings:

    texts = []
    text_based_model = None

    def __init__(self, excludeId=''):
        with codecs.open('../data/vc-slam/models/vcslam_models.json', encoding='utf8', mode='r') as f:
            vcslam = json.load(f)
            for model in vcslam['models']:
                if model['id'] != excludeId:
                    self.texts.append(gensim.utils.simple_preprocess(model['description']))

        #self.text_based_model = gensim.models.Word2Vec(
        #    self.texts,
        #    size=100,
        #    window=5,
        #    min_count=1,
        #    workers=4,
        #    iter=1000)
        self.text_based_model = gensim.models.Word2Vec(self.texts, workers=4)

    def reset(self):
        self.texts = []
        self.text_based_model = None

    def get_similar_words(self, word):
        try:
            return [(x[0], x[1]) for x in self.text_based_model.wv.most_similar(positive=[word], topn=100)]
        except:
            return []



