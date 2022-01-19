class TextConceptExtraction:

    text = ''
    ontology = {}
    nlp = None
    nouns = []
    attributes = []
    attribute_lemmas = {}

    def __init__(self, text, attributes, ontology, nlp):
        self.text = text
        self.ontology = ontology
        self.ontology['lemmas'] = []
        self.nouns = []
        self.attributes = attributes
        self.attribute_lemmas = {}
        self.nlp = nlp

    def get_text(self):
        return self.text

    def get_ontology(self):
        return self.ontology

    def get_nouns(self):
        return self.nouns

    def get_attributes(self):
        return self.attributes

    def get_attribute_lemmas(self):
        return self.attribute_lemmas

    def execute(self):
        doc = self.nlp(self.text)
        for token in doc:
            if token.pos_ == 'NOUN':
                self.nouns.append(token.lemma_)
        concepts_str = ' '.join(self.ontology['concepts'])
        concept_doc = self.nlp(concepts_str)
        for token in concept_doc:
            if token.pos_ == 'NOUN':
                self.ontology['lemmas'].append(token.lemma_)

        attributes_str = ' '.join(self.attributes)
        attributes_doc = self.nlp(attributes_str)
        for token in attributes_doc:
            if token.pos_ == 'NOUN':
                self.attribute_lemmas[str(token)] = token.lemma_



