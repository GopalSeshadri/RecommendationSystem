import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


def loadFile(filename):
    return pd.read_csv('Data/{}.csv'.format(filename))

def groupMovieTags(data):
    data = data[['movieId', 'tag']]
    data_grouped = data.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    return data_grouped

def createTFIDFMatrix(data):
    tfidf = TfidfVectorizer(ngram_range=(1, 1), min_df=0.001, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['tag'])
    return tfidf_matrix


## Loading the data files
links_df = loadFile("links")
movies_df = loadFile("movies")
ratings_df = loadFile("ratings")
tags_df = loadFile("tags")

## Making movieId and userId to start from zero
links_df['movieId'] = links_df['movieId'] - 1
movies_df['movieId'] = movie_df['movieId'] - 1
ratings_df['movieId'] = ratings_df['movieId'] - 1
ratings_df['userId'] = ratings_df['userId'] - 1
tags_df['movieId'] = tags_df['movieId'] - 1
tags_df['userId'] = tags_df['userId'] - 1

## grouping the tags for each movie
tags_grouped_df = groupMovieTags(tags_df)

## calculating the TFIDF matrix
tfidf_matrix = createTFIDFMatrix(tags_grouped_df)
