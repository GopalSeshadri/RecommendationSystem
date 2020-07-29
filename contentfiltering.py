import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utilities import Util

class ContentFilter:
    def __init__(self, embeddings_matrix, current):
        self.embeddings_matrix = embeddings_matrix
        self.current = current

    def calculateCosineSimilarity(self):
        similarity_scores = cosine_similarity(X = self.current, Y = self.embeddings_matrix)
        return similarity_scores[0]

    def getTop10(self):
        similarity_scores = self.calculateCosineSimilarity()
        top10 = np.argpartition(similarity_scores, -10)[-10:]
        return top10

# tfidf_embeddings = Util.loadObj('tfidf_reduced_matrix')
# ## Have to append other embeddings
# embeddings_matrix =  tfidf_embeddings.to_numpy()
# current = [embeddings_matrix[0]]
# cf = ContentFilter(embeddings_matrix, current)
# print(tfidf_embeddings.index[cf.getTop10()])
