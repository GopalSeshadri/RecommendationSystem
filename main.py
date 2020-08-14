import numpy as np
import pandas as pd

from preprocess import Preprocess
from utilities import Util
from contentfiltering import ContentFilter

## Loading the data files
links_df = Preprocess.loadFile("links")
movies_df = Preprocess.loadFile("movies")
ratings_df = Preprocess.loadFile("ratings")
tags_df = Preprocess.loadFile("tags")

imdb_df = Preprocess.preprocessImdbFile()

## grouping the tags for each movie
grouped_df = Preprocess.groupMovieTags(tags_df)

## getting the genre tags for each movie
movies2_df = movies_df.iloc[:][['movieId', 'genres']]
movies2_df['genres'] = movies2_df['genres'].apply(lambda x: x.replace('|', ' ').lower())

## combining the genre tags and the user tags
movies2_df = movies2_df.merge(grouped_df, on = 'movieId', how = 'left')
movies2_df['tag'] = movies2_df['tag'].apply(lambda x: str(x)) + ' ' + movies2_df['genres']
movies2_df['tag'] = movies2_df['tag'].apply(lambda x : x.replace('nan', '').strip())
tags_grouped_df = movies2_df.iloc[:][['movieId', 'tag']]
print(tags_grouped_df.head())
print(tags_grouped_df.shape)

## calculating the TFIDF matrix
tfidf_df = Preprocess.createTFIDFMatrix(tags_grouped_df)
print(tfidf_df.shape)

## dumping the tfidf matrix
Util.saveObj(tfidf_df, 'tfidf_df')

# ## loading the TFIDF matrix
# tfidf_df =  Util.loadObj('tfidf_df')
# print(tfidf_df.shape)

## loading the reduced TFIDF matrix
tfidf_reduced_df =  Util.loadObj('tfidf_reduced_df')
print(tfidf_reduced_df.shape)

## creating vector df with spacy sentence vector
vector_df = createSentenceVector(imdb_df)
print(vector_df.shape)

## dumping the vector df
Util.saveObj(vector_df, 'vector_df')

# ## loading vector df
# vector_df =  Util.loadObj('vector_df')
# print(vector_df.shape)

## merging tfidf reduced df and vector df
vector_df['movieId'] = vector_df['movieId'].apply(lambda x : int(x))
tfidf_reduced_df['movieId'] = tfidf_reduced_df['movieId'].apply(lambda x : int(x))
final_vector_df = pd.merge(vector_df, tfidf_reduced_df, on = 'movieId', how = 'left')
print(final_vector_df.shape)
print(final_vector_df.isna().sum())

## dumping the final vector df
Util.saveObj(final_vector_df, 'final_vector_df')


print(movies_df[movies_df['movieId'].isin([5, 151, 163, 168, 186, 225, 315, 342, 466, 543, 552, 555, 596, 1234, 1376, 1396, 1639, 2302, 2395, 2699])])


## getting user rating history
movies_watched_list = ratings_df[ratings_df['userId'] == 147]['movieId'].tolist()
ratings_given_list = ratings_df[ratings_df['userId'] == 147]['rating'].tolist()


if len(movies_watched_list) > 10:
    movies_watched_list = movies_watched_list[-10:]
    ratings_given_list = ratings_given_list[-10:]

sum_ratings_given = sum(ratings_given_list)
rating_adjusted_list = [round(r/sum_ratings_given, 3) for r in ratings_given_list]

## creating the 600 dimensional movie compound representation for the user
movie_600_list = []
for i, m in enumerate(movies_watched_list):
    movie_600 = final_vector_df[final_vector_df['movieId'] == m].iloc[:, 1:].to_numpy() * rating_adjusted_list[i]
    # print(len(movie_600))
    movie_600_list.append(movie_600[0])
movie_600_final = np.sum(movie_600_list, axis = 0)

## finding the movie list for that user based on his compound embedding
embeddings = Util.loadObj('final_vector_df')
embeddings_matrix =  embeddings.loc[:, embeddings.columns != 'movieId']
embedding_movie_list = embeddings['movieId'].tolist()

current = [movie_600_final]
cf = ContentFilter(embeddings_matrix, current)
top10_list = cf.getTop10().tolist()
movie_list = [embedding_movie_list[each] for each in top10_list]
print(movie_list)

print(movies_df[movies_df['movieId'].isin(movie_list)])

movie_list2 = [29, 260, 750, 223, 3671, 1198, 2000, 4993, 1967, 55820]

print(movies_df[movies_df['movieId'].isin(movie_list2)])
