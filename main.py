import numpy as np
import pandas as pd

from preprocess import Preprocess
from utilities import Util

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

# ## calculating the TFIDF matrix
# tfidf_df = Preprocess.createTFIDFMatrix(tags_grouped_df)
# print(tfidf_df.shape)
#
# ## dumping the tfidf matrix
# Util.saveObj(tfidf_df, 'tfidf_df')

## loading the TFIDF matrix
tfidf_df =  Util.loadObj('tfidf_df')
print(tfidf_df.shape)

## loading the reduced TFIDF matrix
tfidf_reduced_df =  Util.loadObj('tfidf_reduced_df')
print(tfidf_reduced_df.shape)

# ## creating vector df with spacy sentence vector
# vector_df = createSentenceVector(imdb_df)
# print(vector_df.shape)
#
# ## dumping the vector df
# Util.saveObj(vector_df, 'vector_df')

## loading vector df
vector_df =  Util.loadObj('vector_df')
print(vector_df.shape)

## merging tfidf reduced df and vector df
vector_df['movieId'] = vector_df['movieId'].apply(lambda x : int(x))
tfidf_reduced_df['movieId'] = tfidf_reduced_df['movieId'].apply(lambda x : int(x))
final_vector_df = pd.merge(vector_df, tfidf_reduced_df, on = 'movieId', how = 'left')
print(final_vector_df.shape)
print(final_vector_df.isna().sum())

## dumping the final vector df
Util.saveObj(final_vector_df, 'final_vector_df')

# print(True if 46855 in vector_df['movieId'].tolist() else False)
# print(True if 46855 in tfidf_reduced_df['movieId'].tolist() else False)
# print(True if 46855 in tfidf_df.index.tolist() else False)
# print(True if 46855 in tags_grouped_df['movieId'].tolist() else False)

print(movies_df[movies_df['movieId'].isin([5, 151, 163, 168, 186, 225, 315, 342, 466, 543, 552, 555, 596, 1234, 1376, 1396, 1639, 2302, 2395, 2699])])

## 720 1107 2377 86668 176601

## 5, 151, 163, 168, 186, 225, 315, 342, 466, 543, 552, 555, 596, 1234, 1376, 1396, 1639, 2302, 2395, 2699
## 183611, 6583, 40851, 9010, 6979, 1625, 102590, 6790, 179401, 2
