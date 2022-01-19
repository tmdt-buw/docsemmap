from flair.embeddings import DocumentPoolEmbeddings, TransformerWordEmbeddings
from flair.data import Sentence
import flair
import torch
from torch import nn


class FlairProcessing:
    document_embeddings = None
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def __init__(self):
        if torch.cuda.is_available():
            flair.device = torch.device('cuda:0')
        embedding = TransformerWordEmbeddings('bert-base-uncased')
        self.document_embeddings = DocumentPoolEmbeddings([embedding])

    def get_vector(self, text):
        try:
            s = Sentence(text)
            self.document_embeddings.embed(s)
            return s.embedding
        except:
            return None

    def get_cosine_similarity_of_documents(self, doc1, doc2):
        s1 = Sentence(doc1)
        s2 = Sentence(doc2)
        self.document_embeddings.embed(s1)
        self.document_embeddings.embed(s2)
        return float(self.cos(s1.embedding, s2.embedding).data.item())

    def get_cosine_similarity_of_vectors(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return 0
        return float(self.cos(vec1, vec2).data.item())
