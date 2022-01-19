import os
from json import JSONEncoder

import spacy
from wikipedia2vec import Wikipedia2Vec
from FlairProcessing import FlairProcessing
from VCSLAMEmbeddings import VCSLAMEmbeddings
from OntologyProcessing import OntologyProcessing
from DataProcessing import DataProcessing
from HistoryProcessing import HistoryProcessing
import wikipediaapi
import logging
import numpy as np
import operator
import codecs
import json

from Levenshtein import distance

# Matching weights, evalutated by grid search
matching_weights = {
        'close_aix': 1.05,
        'similar_vc_aix': 0.25,
        'similar_wiki_aix': 0.5,
        'partial_match_aix': 1.45,
        'similar_desc_aix': 1.95,
        'similar_wiki_sec_aix': 1.0,
        'history': 1.25
    }

# parse numpy values for export
class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return JSONEncoder.default(self, obj)

# Persists the candidate lists of a data source
def persist_candidates(candidates, results, id_str):

    for r in results.keys():
        candidates[r]['prediction'] = results[r]
    content_str = json.dumps(candidates, cls=NumpyFloatValuesEncoder)

    with codecs.open('../candidates/' + id_str + '.json', 'w', 'utf8') as f:
        f.write(content_str)

# Prepare all Pipelne steps
def global_preprocesing(exclude_id):
    te_vc = train_vcslam_embeddings(exclude_id)
    te_wiki = load_wiki_embeddings()
    te_spacy = load_spacy_embeddings()
    te_bert = load_pretrained_bert_embeddings()
    ontology = preprocess_ontology()
    history_processing = HistoryProcessing(exclude_id)
    return te_vc, te_wiki, te_spacy, te_bert, ontology, history_processing

# Process a complete data set with the given id
def dataset_processing(id):
    te_vc, te_wiki, te_spacy, te_bert, ontology, history_processing = global_preprocesing(id)

    data_processing = DataProcessing(te_spacy, id)

    data = data_processing.get_data()
    labels = data.keys()
    candidates = {}
    results = {}

    for aix in labels:
        candidates[aix] = label_processing(aix, data_processing, te_vc, te_wiki, te_bert, ontology, history_processing)
        results[aix] = matching(candidates[aix])

    persist_candidates(candidates, results, id)
    return results

# Checks if a concept is in the candidate list
def concept_in_candidates(concept, candidates):
    for c in candidates.keys():
        if concept in [x[0] for x in candidates[c]]:
            return True
    return False

# perform matching for all candidates of one label
def matching(candidates_aix):
    match_dict = {}

    for type in candidates_aix.keys():
        update_match_by_list(match_dict, candidates_aix[type], type)

    try:
        return max(match_dict.items(), key=operator.itemgetter(1))[0]
    except:
        return ''

# Updates the match list with the related weights of a type
def update_match_by_list(match_dict, lst, type_name: str):
    for entry in lst:
        if entry[0] in match_dict:
            match_dict[entry[0]] += entry[1] * matching_weights[type_name]
        else:
            match_dict[entry[0]] = entry[1] * matching_weights[type_name]


# Pipepeline processing for one label
def label_processing(aix, data_processing, te_vc, te_wiki, te_bert, ontology: OntologyProcessing, history_processing: HistoryProcessing):
    log('Identifying Candidates for --- ' + aix + ' ---')

    candidates = {}

    close_aix = data_processing.get_close_words(aix, 8)

    close_aix = [(x, 0.5) for x in close_aix]

    similar_vc_aix = te_vc.get_similar_words(aix)
    similar_wiki_aix = get_similar_by_wiki(aix, 20, te_wiki)
    history_aix = history_processing.get_history_mapping(aix)

    partial_match_aix = perform_partial_match(aix, data_processing, ontology)

    similar_desc_aix = []

    close_sentences = [sent for sent in data_processing.get_close_sentences(aix) if len(sent.split(' ')) > 4]
    for con in ontology.get_concepts().keys():
        for sent in close_sentences:
            vector_sentence = te_bert.get_vector(sent)
            if 'descriptionVector' in ontology.get_concepts()[con]:
                similarity = te_bert.get_cosine_similarity_of_vectors(vector_sentence,
                                                                      ontology.get_concepts()[con]['descriptionVector'])
                if similarity >= 0.9:
                    similar_desc_aix.append((con, similarity))

    # prepare result dictionary

    similar_wiki_sec_aix = []

    wiki_desc = get_wikipedia_description(aix)
    if wiki_desc != '':
        vector_wiki_desc = te_bert.get_vector(wiki_desc)
        for con in ontology.get_concepts().keys():
            if 'descriptionVector' in ontology.get_concepts()[con]:
                similarity = te_bert.get_cosine_similarity_of_vectors(vector_wiki_desc,
                                                                      ontology.get_concepts()[con]['descriptionVector'])
                if similarity >= 0.85:
                    similar_wiki_sec_aix.append((con, similarity))

    candidates['close_aix'] = ontology.reduce_to_known_concepts_with_ratings(close_aix)
    candidates['similar_vc_aix'] = ontology.reduce_to_known_concepts_with_ratings(similar_vc_aix)
    candidates['similar_wiki_aix'] = ontology.reduce_to_known_concepts_with_ratings(similar_wiki_aix)
    candidates['partial_match_aix'] = ontology.reduce_to_known_concepts_with_ratings(partial_match_aix)
    candidates['similar_desc_aix'] = ontology.reduce_to_known_concepts_with_ratings(similar_desc_aix)
    candidates['similar_wiki_sec_aix'] = ontology.reduce_to_known_concepts_with_ratings(similar_wiki_sec_aix)
    candidates['history'] = ontology.reduce_to_known_concepts_with_ratings(history_aix)

    return candidates

# Initializes the Flair BERT model
def load_pretrained_bert_embeddings():
    return FlairProcessing()

# trains the vcslam embeddings and excludes the data source with the given id from training
def train_vcslam_embeddings(exclude_id):
    return VCSLAMEmbeddings(exclude_id)

# Initializes the wiki embeddings
def load_wiki_embeddings():
    return Wikipedia2Vec.load('../models/enwiki_20180420_100d.pkl')

# Initializes the spacy embeddings
def load_spacy_embeddings():
    return spacy.load('en_core_web_trf')

# Initializes the Ontology
def preprocess_ontology():
    return OntologyProcessing()

# Return a list of tuples (concept, similarity) using the spacy similarity function
def get_similar_by_spacy(word, top_n=10, te_spacy=None):
    try:
        ms = te_spacy.vocab.vectors.most_similar(
            te_spacy(word).vector.reshape(1, te_spacy(word).vector.shape[0]), n=top_n)
        words = [
            (te_spacy.vocab.strings[w].lower(), te_spacy(te_spacy.vocab.strings[w].lower()).similarity(te_spacy(word)))
            for w in ms[0][0]]
        return list(set(words))
    except:
        return []


# Calculate the cosine similarity of two vectors
def cosine_similarity(u: np.ndarray, v: np.ndarray):
    assert (u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = uv / np.sqrt(uu * vv)
    return cos_theta

# Get top_n most similar words to term using wikipedia embeddings
def get_similar_by_wiki(term, top_n=10, te_wiki=None):
    try:
        return [(w[0].text, w[1]) for w in te_wiki.most_similar(te_wiki.get_word(term), 20) if hasattr(w[0], 'text')]
    except:
        return []

# Returns a list of lemmas of the words provided using the spacy model
def get_lemmas_of_list(terms, te_spacy):
    doc = te_spacy(' '.join(terms))
    lemmas = []
    for token in doc:
        if token.pos_ == 'NOUN':
            lemmas.append(token.lemma_)
    return list(set(lemmas))

# Returns a list of lemmas and original score of the words provided using the spacy model
def get_lemmas_of_list_with_rating(terms, te_spacy):
    lemmas = []
    for term in terms:
        doc = te_spacy(term[0])
        for token in doc:
            if token.pos_ == 'NOUN':
                lemmas.append((token.lemma_, term[1]))
    return list(set(lemmas))

# Returns the Wikipedia description of the term provided, if existing. otherwise returns an empty string
def get_wikipedia_description(term):
    try:
        wiki_wiki = wikipediaapi.Wikipedia('en')
        page = wiki_wiki.page(term)
        return page.summary
    except Exception as ex:
        logging.getLogger('manager').warning("Could not get Wiki Description for term " + term)
        return ''

# Calculates and scores partial matches with concepts from the ontology
def perform_partial_match(aix, data_processing, ontology):
    attribute_components = [aix]

    for split_pattern in ['_', '-', '#', ':']:
        splitted = aix.split(split_pattern)
        if len(splitted) > 1:
            attribute_components += splitted

    attribute_components += data_processing.get_data()[aix]['component-lemmas']

    # todo: check if component part of attribute

    attribute_components = list(set([a.lower() for a in attribute_components]))
    print('Attribute components: ', attribute_components)

    results = ontology.reduce_to_known_concepts_partial_matchs(attribute_components)
    return [[x, score_by_levenshtein_and_length(aix.lower(), x.lower())] for x in results]

# Calculates the length-dependent levenshtein distance between two strings
def score_by_levenshtein_and_length(s1, s2):
    l = max(len(s1), len(s2))
    dist = float(float(l - distance(s1, s2)) / float(l))
    return dist

# logs a message to the console
def log(message):
    logging.getLogger('manager').info(message)

# persists the results of one data source to a result file
def persist_results(id, distribution, correct, possible, wrong, candidates, predictions):
    filename = '../results/' + id + '.json'

    accuracy_correct = float(float(len(correct)) / float(len(distribution)))
    accuracy_possible = float(float(len(correct) + len(possible)) / float(len(distribution)))

    peristed_result = {
        'id': id,
        'accuracy_possible': accuracy_possible,
        'accuracy_correct': accuracy_correct,
        'distribution': distribution,
        'correct': correct,
        'possible': possible,
        'wrong': wrong,
        'candidates': str(candidates),
        'predictions': str(predictions)
    }

    with codecs.open(filename, encoding='utf8', mode='w') as f:
        json.dump(peristed_result, f, sort_keys=True, indent=4)

# entry point of the pipeline
if __name__ == '__main__':

    FORMAT = '%(asctime)-15s \t %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger('manager').setLevel(logging.INFO)

    # iterate through all vcslam data sources
    for id in range(1, 101):
        id_str = str(id).zfill(4)
        predictions = dataset_processing(id_str)
        print(predictions)
