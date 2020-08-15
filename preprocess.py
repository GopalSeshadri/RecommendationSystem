import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition.pca import PCA
from sklearn.decomposition.truncated_svd import TruncatedSVD
import tensorflow as tf
import tensorflow.keras as keras
import spacy

nlp = spacy.load("en_core_web_md")

class Preprocess:

    def loadFile(filename):
        '''
        This function takes in a file name and loads it's content to a DataFrame.

        Parameters:
        filename (str) : The name of the input file.

        Returns:
        df (DataFrame) : The DataFrame with data loaded from the input file.
        '''

        df = pd.read_csv('Data/{}.csv'.format(filename))
        return df

    def groupMovieTags(data):
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

    def createTFIDFMatrix(data):
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
        tfidf_df = pd.concat([data['movieId'].reset_index(drop = True), tfidf_df.reset_index(drop = True)], axis = 1)
        tfidf_df.set_index('movieId', inplace = True)
        return tfidf_df

    def preprocessImdbFile():
        '''
        This function reads in the imdb.csv file and returns a dataframe with added
        column headers and index.

        Returns:
        df (DataFrame) : The Imdb Data Frame.
        '''
        filename = 'Data/imdb.csv'
        df = pd.read_csv('Data/imdb.csv'.format(filename))
        df.columns = ['index', 'movieId', 'title', 'oneLiner', 'director', 'cast1', 'cast2', 'cast3']
        df.set_index('index', inplace = True)
        df.to_csv('Data/imdb.csv')
        return df

    def createSentenceVector(data):
        '''
        This function takes in the imdb.csv file and returns a dataframe with spacy vectors for
        each movies.

        Parameters:
        data (DataFrame) : The Imdb Data Frame.

        Returns:
        vector_df (DataFrame) : The data frame with spacy vectors.
        '''
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
