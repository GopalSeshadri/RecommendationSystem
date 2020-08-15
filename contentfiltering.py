import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utilities import Util

class ContentFilter:
    '''
    This class implements the content filtering usig cosine_similarity
    '''
    def __init__(self, embeddings_matrix, current):
        '''
        This constructor takes in an embeddings matrix and the vector of the current movie or
        weighted movie vector of an user.

        Parameters:
        embeddings_matrix (Matrix) : The embedding matrix.
        current (list)             : The current movie vector or the weighted movie vector of an user.
        '''
        self.embeddings_matrix = embeddings_matrix
        self.current = current

    def calculateCosineSimilarity(self):
        '''
        This function calculate the similarity score between the current vector and the embedding matrix.

        Returns:
        similarity_scores[0] (array) : An array of similarity scores between current and the other movies.
        '''
        similarity_scores = cosine_similarity(X = self.current, Y = self.embeddings_matrix)
        return similarity_scores[0]

    def getTop10(self):
        '''
        This function returns a list of top 10 similarity scores index locations.

        Returns:
        top10 (list) : The indices of top 10 movies with high similarity scores.
        '''
        similarity_scores = self.calculateCosineSimilarity()
        top10 = np.argpartition(similarity_scores, -10)[-10:]
        return top10

# embeddings = Util.loadObj('final_vector_df')
# embeddings_matrix =  embeddings.loc[:, embeddings.columns != 'movieId']
# embedding_movie_list = embeddings['movieId'].tolist()
#
# current = [embeddings_matrix.iloc[1]]
# cf = ContentFilter(embeddings_matrix, current)
# top10_list = cf.getTop10().tolist()
# movie_list = [embedding_movie_list[each] for each in top10_list]
