import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition.pca import PCA
from sklearn.decomposition.truncated_svd import TruncatedSVD
import tensorflow as tf
import tensorflow.keras as keras

from utilities import Util


def loadFile(filename: str):
    '''
    This function takes in a file name and loads it's content to a DataFrame.

    Parameters:
    filename (str) : The name of the input file.

    Returns:
    df (DataFrame) : The DataFrame with data loaded from the input file.
    '''

    df = pd.read_csv('Data/{}.csv'.format(filename))
    return df

def groupMovieTags(data: pd.DataFrame):
    '''
    This function takes in the tags DataFrame and returns the data grouped by movieId.

    Parameters:
    data (DataFrame) : Tags DataFrame

    Returns:
    data_grouped (DataFrame) : The DataFrame with tags grouped by each movieId.
    '''

    data = data[['movieId', 'tag']]
    data_grouped = data.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    return data_grouped

def createTFIDFMatrix(data: pd.DataFrame):
    '''
    This function takes in the tags DataFrame grouped by movieId and returns a
    TFIDF matrix fitted on grouped tags.

    Parameters:
    data (DataFrame) : The DataFrame with tags grouped by each movieId.

    Returns:
    tfidf_matrix (DataFrame) : A matrix of dimension (1572, 723).
    '''

    tfidf = TfidfVectorizer(ngram_range=(1, 1), min_df=0.0001, stop_words='english')
    tfidf_matrix = pd.DataFrame(tfidf.fit_transform(data['tag']).toarray())
    tfidf_matrix.set_index(data['movieId'], inplace = True)
    return tfidf_matrix

def preprocessImdbFile():
    filename = '\Data\imdb.csv'
    df = pd.read_csv('Data/imdb.csv'.format(filename))
    df.columns = ['index', 'movieId', 'title', 'oneLiner', 'director', 'cast1', 'cast2', 'cast3']
    df.set_index('index', inplace = True)
    df.to_csv('Data/imdb.csv')
    return df



## Loading the data files
links_df = loadFile("links")
movies_df = loadFile("movies")
ratings_df = loadFile("ratings")
tags_df = loadFile("tags")

imdb_df = preprocessImdbFile()

## Making movieId and userId to start from zero
links_df['movieId'] = links_df['movieId'] - 1
movies_df['movieId'] = movies_df['movieId'] - 1
ratings_df['movieId'] = ratings_df['movieId'] - 1
ratings_df['userId'] = ratings_df['userId'] - 1
tags_df['movieId'] = tags_df['movieId'] - 1
tags_df['userId'] = tags_df['userId'] - 1

## grouping the tags for each movie
tags_grouped_df = groupMovieTags(tags_df)

## calculating the TFIDF matrix
tfidf_matrix = createTFIDFMatrix(tags_grouped_df)

## dumping the tfidf matrix
Util.saveObj(tfidf_matrix, 'tfidf_matrix')

## loading the reduced TFIDF matrix
tfidf_matrix_reduced =  Util.loadObj('tfidf_matrix')

print(movies_df[movies_df['movieId'].isin([25885, 6272, 3113, 27019, 3159, 1725, 198, 122917, 2354, 0])])
