import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from preprocess import Preprocess
from utilities import Util

class EmbeddingLayer:
    def __init__(self, n_items, n_factors, name):
        self.n_items = n_items
        self.n_factors = n_factors
        self.name = name

    def __call__(self, x):
        x = keras.layers.Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer = keras.regularizers.l2(1e-6), name = self.name)(x)
        x = keras.layers.Reshape((self.n_factors,))(x)
        return x

class LambdaLayer(keras.layers.Layer):
    def __init__(self, min_rating, max_rating, **kwargs):
        super(LambdaLayer, self).__init__(**kwargs)
        self.min_rating = min_rating
        self.max_rating = max_rating

    def call(self, input):
        return tf.math.add(tf.math.multiply(input, (self.max_rating - self.min_rating)), self.min_rating)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'min_rating': self.min_rating, \
                    'max_rating' : self.max_rating})
        return config


def RecommenderV2(n_users, n_movies, n_factors, min_rating, max_rating):
    user = keras.layers.Input(shape = (1,))
    u = EmbeddingLayer(n_users, n_factors, 'user_weights')(user)
    ub = EmbeddingLayer(n_users, 1, 'user_bias')(user)

    movie = keras.layers.Input(shape = (1,))
    m = EmbeddingLayer(n_movies, n_factors, 'movie_weights')(movie)
    mb = EmbeddingLayer(n_movies, 1, 'movie_bias')(movie)
    x = keras.layers.Dot(axes = 1)([u, m])
    x = keras.layers.Add()([x, ub, mb])
    x = keras.layers.Activation('sigmoid')(x)
    # x = keras.layers.Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)
    x = LambdaLayer(min_rating, max_rating)(x)
    model = keras.Model(inputs=[user, movie], outputs = x)
    opt = keras.optimizers.Adam(lr = 0.001)
    model.compile(loss = 'mean_squared_error', optimizer = opt)
    return model

LATENT_FEATURES = 128
NUM_EPOCHS = 2
BATCH_SIZE = 32

ratings_df = Preprocess.loadFile("ratings")

# ratings_input =  [ratings_df['userId'].to_numpy(), ratings_df['movieId'].to_numpy(), ratings_df['rating'].to_numpy()]
users =  list(set(ratings_df['userId'].tolist()))
movies = list(set(ratings_df['movieId'].tolist()))

users_dict = {u : i for i, u in enumerate(users)}
movies_dict = {m : i for i, m in enumerate(movies)} # Movie Id to Idx
movies_idx_dict = {i : m for i, m in enumerate(movies)} #Idx to movie Id

num_users = len(users)
num_movies = len(movies)

ratings_df['userId'] = ratings_df['userId'].apply(lambda x: users_dict[x])
ratings_df['movieId'] = ratings_df['movieId'].apply(lambda x: movies_dict[x])
ratings_input = [ratings_df['userId'].to_numpy(), ratings_df['movieId'].to_numpy()]
ratings_output = ratings_df[['rating']].to_numpy()

min_rating, max_rating = np.amin(ratings_df['rating'].to_numpy()), np.amax(ratings_df['rating'].to_numpy())

# lmf_model = LMF(LATENT_FEATURES, users_dict, movies_dict, min_rating, max_rating)

model = RecommenderV2(num_users, num_movies, LATENT_FEATURES, min_rating, max_rating)
model.summary()

model.fit(x = ratings_input, y = ratings_output, batch_size = BATCH_SIZE, epochs = 5, verbose = 1)

model.save('Temp/collaborative.h5')

model = tf.keras.models.load_model('Temp/collaborative.h5', custom_objects={'LambdaLayer': LambdaLayer})

embeddings =  model.predict(ratings_input)

print([(i, layer.name) for i, layer in enumerate(model.layers)])

user_weights = model.get_layer("user_weights").get_weights()[0]
movie_weights = model.get_layer("movie_weights").get_weights()[0]

user_bias = model.get_layer("user_bias").get_weights()[0]
movie_bias = model.get_layer("movie_bias").get_weights()[0]

dp_matrix = np.dot(user_weights, np.transpose(movie_weights))
dp_matrix = np.add(np.add(dp_matrix, user_bias), np.transpose(movie_bias))

print(dp_matrix.shape)

sigmoid =  np.vectorize(lambda x : 1/(1 + np.exp(-(x))))

dp_sigmoid_matrix = sigmoid(dp_matrix)
dp_scaled_matrix = dp_sigmoid_matrix * (max_rating - min_rating) + min_rating

print(dp_scaled_matrix)

user_idx = 146 ## reduced by 1
movies_rated = list(ratings_df[ratings_df['userId'] == user_idx]['movieId'])
count = 0

user_ratings = dp_scaled_matrix[user_idx, :]
print(len(user_ratings))
user_ratings[movies_rated] = 0.5
top10_list = np.argpartition(user_ratings, -10)[-10:]
movie_list = [movies_idx_dict[each] for each in top10_list]

print(dp_scaled_matrix[user_idx, :])
print(movie_list)
print(movies_rated)
print(dp_scaled_matrix[user_idx, :][movies_rated])
