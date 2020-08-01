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
        print(similarity_scores.shape)
        top10 = np.argpartition(similarity_scores, -10)[-10:]
        return top10

embeddings = Util.loadObj('final_vector_df')
embeddings_matrix =  embeddings.loc[:, embeddings.columns != 'movieId']
embedding_movie_list = embeddings['movieId'].tolist()
print(embeddings.shape)
print(embeddings_matrix.shape)
# print(embeddings.isna().sum())
current = [embeddings_matrix.iloc[0]]
cf = ContentFilter(embeddings_matrix, current)
top10_list = cf.getTop10().tolist()
movie_list = [embedding_movie_list[each] for each in top10_list]
print(movie_list)
