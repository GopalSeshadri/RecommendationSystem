import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition.pca import PCA
from sklearn.decomposition.truncated_svd import TruncatedSVD
import tensorflow as tf
import tensorflow.keras as keras
import spacy

from utilities import Util

nlp = spacy.load("en_core_web_md")


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
    data_grouped = data.groupby('movieId')['tag'].apply(lambda x: ' '.join(x).lower()).reset_index()
    return data_grouped

def createTFIDFMatrix(data: pd.DataFrame):
    '''
    This function takes in the tags DataFrame grouped by movieId and returns a
    TFIDF matrix fitted on grouped tags.

    Parameters:
    data (DataFrame) : The DataFrame with tags grouped by each movieId.

    Returns:
    tfidf_df (DataFrame) : A matrix of dimension (1572, 723).
    '''

    tfidf = TfidfVectorizer(ngram_range=(1, 1), min_df=0.0001, stop_words='english')
    tfidf_df = pd.DataFrame(tfidf.fit_transform(data['tag']).toarray())
    tfidf_df.set_index(data['movieId'], inplace = True)
    return tfidf_df

def preprocessImdbFile():
    filename = '\Data\imdb.csv'
    df = pd.read_csv('Data/imdb.csv'.format(filename))
    df.columns = ['index', 'movieId', 'title', 'oneLiner', 'director', 'cast1', 'cast2', 'cast3']
    df.set_index('index', inplace = True)
    df.to_csv('Data/imdb.csv')
    return df

def createSentenceVector(data: pd.DataFrame):
    sentences = data['oneLiner'].tolist()
    stopwords = nlp.Defaults.stop_words
    sentence_dict = {}

    for i, sentence in enumerate(sentences):
        if sentence != "":
            word_vector_list = []
            for token in nlp(sentence.lower()):
                if token.text not in stopwords and token.pos_  not in ('PUNCT', 'NUM', 'SYM'):
                    word_vector_list.append(token.vector)
            sentence_dict[data.iloc[i]['movieId']] = np.mean(word_vector_list, axis = 0)
            # print(type(sentence_dict))


    vector_df =  pd.DataFrame.from_dict(sentence_dict).T
    vector_df = pd.concat([data['movieId'].reset_index(drop = True), vector_df.reset_index(drop = True)], axis = 1)
    return vector_df





## Loading the data files
links_df = loadFile("links")
movies_df = loadFile("movies")
ratings_df = loadFile("ratings")
tags_df = loadFile("tags")

imdb_df = preprocessImdbFile()

## Making movieId and userId to start from zero
# links_df['movieId'] = links_df['movieId'] - 1
# movies_df['movieId'] = movies_df['movieId'] - 1
# ratings_df['movieId'] = ratings_df['movieId'] - 1
# ratings_df['userId'] = ratings_df['userId'] - 1
# tags_df['movieId'] = tags_df['movieId'] - 1
# tags_df['userId'] = tags_df['userId'] - 1

# ## grouping the tags for each movie
# grouped_df = groupMovieTags(tags_df)
#
# ## getting the genre tags for each movie
# movies2_df = movies_df.iloc[:][['movieId', 'genres']]
# movies2_df['genres'] = movies2_df['genres'].apply(lambda x: x.replace('|', ' ').lower())
#
# ## combining the genre tags and the user tags
# movies2_df = movies2_df.merge(grouped_df, on = 'movieId', how = 'left')
# movies2_df['tag'] = movies2_df['tag'].apply(lambda x: str(x)) + ' ' + movies2_df['genres']
# movies2_df['tag'] = movies2_df['tag'].apply(lambda x : x.replace('nan', '').strip())
# tags_grouped_df = movies2_df.iloc[:][['movieId', 'tag']]
# print(tags_grouped_df.head())
# print(tags_grouped_df.shape)
#
# ## calculating the TFIDF matrix
# tfidf_df = createTFIDFMatrix(grouped_df)
# print(tfidf_df.shape)
#
# ## dumping the tfidf matrix
# Util.saveObj(tfidf_df, 'tfidf_df')
#
# ## loading the reduced TFIDF matrix
# tfidf_reduced_df =  Util.loadObj('tfidf_reduced_df')
# print(tfidf_reduced_df.shape)

# ## loading imdb df with spacy sentence vector
vector_df = createSentenceVector(imdb_df)
print(vector_df.shape)

## dumping the vector df
Util.saveObj(vector_df, 'vector_df')

# vector_df =  Util.loadObj('vector_df')
# print(vector_df.shape)

# vector_df['movieId'] = vector_df['movieId'].apply(lambda x : int(x))
# tfidf_reduced_df['movieId'] = tfidf_reduced_df['movieId'].apply(lambda x : int(x))
# ## merging tfidf reduced df and vector df
# final_vector_df = pd.merge(vector_df, tfidf_reduced_df, on = 'movieId', how = 'left')
# print(final_vector_df.shape)
# print(final_vector_df.isna().sum())
#
# ## dumping the final vector df
# Util.saveObj(final_vector_df, 'final_vector_df')


print(movies_df[movies_df['movieId'].isin([43869, 93272, 476, 8934, 885, 55280, 1324, 175199, 2798, 1])])
