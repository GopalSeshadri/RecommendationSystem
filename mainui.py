import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st

from preprocess import Preprocess
from helpers import Helpers
from utilities import Util
from contentfiltering import ContentFilter
from collaborativefiltering import NeuMF, LMF, MLP

st.title('Recommendation System')

## Loading the data files
@st.cache
def loadData():
    '''
    This function loads the data from various data files and does the basic preprocessing.
    Created to leverage the power of streamlit cache.
    '''
    movies_df = Preprocess.loadFile("movies")
    ratings_df = Preprocess.loadFile("ratings")
    final_vector_df = Util.loadObj('final_vector_df')
    embeddings_matrix =  final_vector_df.loc[:, final_vector_df.columns != 'movieId']
    embedding_movie_list = final_vector_df['movieId'].tolist()

    ratings_df2 = Preprocess.loadFile("ratings")
    # ratings_input =  [ratings_df['userId'].to_numpy(), ratings_df['movieId'].to_numpy(), ratings_df['rating'].to_numpy()]
    users =  list(set(ratings_df['userId'].tolist()))
    movies = list(set(ratings_df['movieId'].tolist()))

    users_dict = {u : i for i, u in enumerate(users)}
    movies_dict = {m : i for i, m in enumerate(movies)} # Movie Id to Idx
    movies_idx_dict = {i : m for i, m in enumerate(movies)} #Idx to movie Id

    ratings_df2['userId'] = ratings_df2['userId'].apply(lambda x: users_dict[x])
    ratings_df2['movieId'] = ratings_df2['movieId'].apply(lambda x: movies_dict[x])

    return movies_df, ratings_df, final_vector_df, embeddings_matrix, embedding_movie_list, ratings_df2, users, movies, users_dict, movies_dict, movies_idx_dict


def createImageGrid(movies_list, movies_df):
    '''
    This function creates a 2x5 Image Grid to display the movie pictures and it's title in the webpage.

    Parameters:
    movies_list (list)    : A list of movies to be displayed
    movies_df (DataFrame) : The movies dataframe with all movies in the data.

    Returns:
    fig (figure) : Image grid with 10 movie pictures and title.
    '''
    fig, ax = plt.subplots(2, 5, figsize=(30, 30))

    for i, each in enumerate(movies_list):
        ax[i//5, i%5].imshow(mpimg.imread('Images/{}.jpg'.format(each)))
        ax[i//5, i%5].set_xticks([])
        ax[i//5, i%5].set_yticks([])
        ax[i//5, i%5].title.set_text(movies_df[movies_df['movieId'] == each]['title'].to_numpy()[0])
        ax[i//5, i%5].title.set_size(18)

    plt.tight_layout()
    # plt.subplots_adjust(wspace =.05, hspace =.05)
    return fig

data_load_state = st.text('Creating a synthetic user and his movie history...')
movies_df, ratings_df, final_vector_df, embeddings_matrix, embedding_movie_list, ratings_df2, users, movies, users_dict, movies_dict, movies_idx_dict = loadData()

num_users = len(users)
num_movies = len(movies)

RANDOM_USER_ID = 1#np.random.randint(0, len(ratings_df['userId'].unique()))

st.sidebar.markdown('To create a random user profile, press the button')

if st.sidebar.button('Create Random User'):
    RANDOM_USER_ID = np.random.randint(0, len(ratings_df['userId'].unique()))

st.sidebar.markdown('**Content Checkboxes**')
st.sidebar.markdown('Check/Uncheck the checkbox to see/hide the data frame and the image grid')

df_checkbox = st.sidebar.checkbox('Show DataFrame', 1)
ig_checkbox = st.sidebar.checkbox('Show ImageGrid')

LATENT_FEATURES = 128

## ########################### Content Filtering #######################

## getting user rating history
movies_watched_list = ratings_df[ratings_df['userId'] == RANDOM_USER_ID]['movieId'].tolist()
ratings_given_list = ratings_df[ratings_df['userId'] == RANDOM_USER_ID]['rating'].tolist()


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
current = [movie_600_final]
cf = ContentFilter(embeddings_matrix, current)
content_top10_list = cf.getTop10().tolist()
content_movie_list = [embedding_movie_list[each] for each in content_top10_list]

## ################################ collaborative Filtering ##################

cp_directory = 'Temp/collaborative/NeuMF/'
latest = tf.train.latest_checkpoint(cp_directory)
model = NeuMF(num_users, num_movies, LATENT_FEATURES, 0.5, 5)

# Load the previously saved weights
model.load_weights(latest)
print('Model loaded successfully')

user_idx = users_dict[RANDOM_USER_ID]

## Creating the test input for the random user and all movies in the data
test_input = [np.array([user_idx] * num_movies), np.array(list(range(num_movies)))]
user_ratings = model.predict(test_input).reshape(num_movies,)

movies_watched_idx_list = [movies_dict[each] for each in movies_watched_list]
user_ratings[movies_watched_idx_list] = 0.5
collaborative_top10_list = np.argpartition(user_ratings, -10)[-10:]
collaborative_movie_list = [movies_idx_dict[each] for each in collaborative_top10_list]
data_load_state.text('Created synthetic user and his movie history...done!')


st.subheader('Movies already watched by the user')
movies_watched_df = movies_df[movies_df['movieId'].isin(movies_watched_list)][['title', 'genres']]

if df_checkbox:
    st.write(movies_watched_df)

if ig_checkbox:
    movies_watched_plot = createImageGrid(movies_watched_list, movies_df)
    st.pyplot(movies_watched_plot)
print(movies_df[movies_df['movieId'].isin(movies_watched_list)])

st.subheader('Movies recommended by content filtering')
movies_rec_content_df = movies_df[movies_df['movieId'].isin(content_movie_list)][['title', 'genres']]

if df_checkbox:
    st.write(movies_rec_content_df)

if ig_checkbox:
    movies_rec_content_plot = createImageGrid(content_movie_list, movies_df)
    st.pyplot(movies_rec_content_plot)
print(movies_df[movies_df['movieId'].isin(content_movie_list)])

st.subheader('Movies recommended by collaborative filtering')
movies_rec_collaborative_df = movies_df[movies_df['movieId'].isin(collaborative_movie_list)][['title', 'genres']]

if df_checkbox:
    st.write(movies_rec_collaborative_df)

if ig_checkbox:
    movies_rec_collaborative_plot = createImageGrid(collaborative_movie_list, movies_df)
    st.pyplot(movies_rec_collaborative_plot)
print(movies_df[movies_df['movieId'].isin(collaborative_movie_list)])
