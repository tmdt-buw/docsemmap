import re
import codecs
import json
from Levenshtein import distance

delimiters = ['.', ',', ' ', '_', '-', ';', ':']
regexPattern = '|'.join(map(re.escape, delimiters))


def parse_relation(relation):
    return {
        'source': relation['source'].split('#')[1].lower(),
        'relation': relation['relation'].split('#')[1].lower(),
        'target': relation['target'].split('#')[1].lower(),
    }


class OntologyProcessing:
    ontology = {}
    concepts = {}
    relations = {}
    lemmas = []
    te_spacy = None
    te_bert = None
    descriptions = {}

    def __init__(self):
        self.load_ontology()
        for c in self.ontology["concepts"]:
            self.concepts[c] = {
                'lemma': '',
                'components': [],
                'component-lemmas': [],
                'description': '',
                'descriptionVector': None
            }
        for r in self.ontology["relations"]:
            self.relations[r['relation']] = {}
        self.relations = self.ontology["relations"]
        #self.te_spacy = te_spacy
        #self.te_bert = te_bert
        #self.preprocess()
        with codecs.open('../data/vc-slam/descriptions/ontology_descriptions.json', encoding='utf8', mode='r') as f:
            self.descriptions = json.load(f)
            for key in self.descriptions.keys():
                if key in self.concepts:
                    self.concepts[key]['description'] = self.descriptions[key]
        #            self.concepts[key]['descriptionVector'] = te_bert.get_vector(self.descriptions[key])

    def get_concepts(self):
        return self.concepts

    def get_relations(self):
        return self.relations

    def get_lemmas(self):
        return self.lemmas

    def preprocess(self):
        for concept in self.concepts.keys():
            doc = self.te_spacy(concept)
            if doc[0].pos_ == 'NOUN':
                self.concepts[concept]['lemma'] = doc[0].lemma_
                self.lemmas.append(doc[0].lemma_)
            else:
                self.concepts[concept]['lemma'] = None
            self.concepts[concept]['components'] = re.split(regexPattern, concept)
            doc = self.te_spacy(' '.join(self.concepts[concept]['components']))
            for token in doc:
                if token.pos_ == 'NOUN':
                    self.concepts[concept]['component-lemmas'].append(token.lemma_)

    def load_ontology(self):
        with codecs.open('../data/vc-slam/models/vcslam_models.json', encoding='utf8', mode='r') as f:
            all_models = json.load(f)
            self.ontology['concepts'] = [c['label'].lower() for c in all_models['ontology']['concepts']]
            self.ontology['relations'] = [parse_relation(r) for r in all_models['ontology']['links']]

    def reduce_to_known_concepts(self, list_of_candidates):
        known_concepts_and_lemmas = list(self.concepts.keys())
        for key in self.concepts.keys():
            known_concepts_and_lemmas.append(self.concepts[key]['lemma'])
            known_concepts_and_lemmas += (self.concepts[key]['components'])
            known_concepts_and_lemmas += (self.concepts[key]['component-lemmas'])
        known_concepts_and_lemmas = list(set([k.lower() for k in known_concepts_and_lemmas if k is not None]))
        return list(set(known_concepts_and_lemmas) & set(list_of_candidates))

    def get_concept_with_lowest_lhevenstein_distance(self, candidate):
        lowest_distance = 99999
        lowest_candidate = None
        known_concepts_and_lemmas = list(self.concepts.keys())
        for key in self.concepts.keys():
            known_concepts_and_lemmas.append(self.concepts[key]['lemma'])
            known_concepts_and_lemmas += (self.concepts[key]['components'])
            known_concepts_and_lemmas += (self.concepts[key]['component-lemmas'])
        known_concepts_and_lemmas = list(set([k.lower() for k in known_concepts_and_lemmas if k is not None]))

        for conc in known_concepts_and_lemmas:
            dist = distance(candidate, conc)
            if dist < lowest_distance:
                lowest_candidate = candidate
                lowest_distance = dist
        return lowest_candidate


    def reduce_to_known_concepts_with_ratings(self, list_of_candidates):
        known_concepts_and_lemmas = list(self.concepts.keys())
        for key in self.concepts.keys():
            known_concepts_and_lemmas.append(self.concepts[key]['lemma'])
            known_concepts_and_lemmas += (self.concepts[key]['components'])
            known_concepts_and_lemmas += (self.concepts[key]['component-lemmas'])
        known_concepts_and_lemmas = list(set(known_concepts_and_lemmas))
        result_list = []
        for entry in list_of_candidates:
            if entry[0] in known_concepts_and_lemmas:
                result_list.append(entry)
        return result_list


    def get_description_of_concept(self, concept):
        if concept in self.descriptions:
            return self.descriptions[concept]
        return ''

    def reduce_to_known_concepts_partial_matchs(self, list_of_candidates):

        lookup = {}

        final_set = []

        for key in self.concepts.keys():
            lookup[key] = [key]
            if 'lemma' in self.concepts[key] and self.concepts[key]['lemma'] is not None:
                lookup[key].append(self.concepts[key]['lemma'])
            if 'components' in self.concepts[key] and self.concepts[key]['components'] is not None:
                lookup[key] += self.concepts[key]['components']
            if 'component-lemmas' in self.concepts[key] and self.concepts[key]['component-lemmas'] is not None:
                lookup[key] += self.concepts[key]['component-lemmas']
            lookup[key] = [x.lower() for x in lookup[key]]


        for key in self.concepts.keys():
            for conc in lookup[key]:
                for att in list_of_candidates:
                    if conc.lower() in att.lower() or att.lower() in conc.lower():
                        final_set.append(key)

        return list(set(final_set))
