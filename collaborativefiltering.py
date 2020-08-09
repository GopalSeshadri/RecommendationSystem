import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from preprocess import Preprocess
from utilities import Util

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

class LMF(keras.Model):
    def __init__(self, num_users, num_movies, num_latent, min_rating, max_rating):
        super(LMF, self).__init__(name = 'LMF')
        self.num_users = num_users
        self.num_movies = num_movies
        self.num_latent = num_latent
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.user_input = keras.layers.Input(shape = (1,), name = 'user_input')
        self.user_weights = keras.layers.Embedding(self.num_users, self.num_latent, embeddings_initializer = 'he_normal',
                      embeddings_regularizer = keras.regularizers.l2(1e-6), name = 'user_weights')
        self.user_bias = keras.layers.Embedding(self.num_users, 1, embeddings_initializer = 'he_normal',
                      embeddings_regularizer = keras.regularizers.l2(1e-6), name = 'user_bias')

        self.movie_input = keras.layers.Input(shape = (1,), name = 'movie_input')
        self.movie_weights = keras.layers.Embedding(self.num_movies, self.num_latent, embeddings_initializer = 'he_normal',
                      embeddings_regularizer = keras.regularizers.l2(1e-6), name = 'movie_weights')
        self.movie_bias = keras.layers.Embedding(self.num_movies, 1, embeddings_initializer = 'he_normal',
                      embeddings_regularizer = keras.regularizers.l2(1e-6), name = 'movie_bias')

        self.reshape_weight_layer = keras.layers.Reshape((self.num_latent,))
        self.reshape_bias_layer = keras.layers.Reshape((1,))

        self.dot_layer = keras.layers.Dot(axes = 1)
        self.add_layer = keras.layers.Add()
        self.sigmoid_layer = keras.layers.Activation('sigmoid')
        self.lambda_layer = LambdaLayer(min_rating, max_rating)

    def call(self, inputs):
        users, movies = inputs[0], inputs[1]

        # ui = self.user_input(users)
        uw = self.user_weights(users)
        uw = self.reshape_weight_layer(uw)
        ub = self.user_bias(users)
        ub = self.reshape_bias_layer(ub)

        # mi = self.movie_input(movies)
        mw = self.movie_weights(movies)
        mw = self.reshape_weight_layer(mw)
        mb = self.movie_bias(movies)
        mb = self.reshape_bias_layer(mb)

        x = self.dot_layer([uw, mw])
        x = self.add_layer([x, ub, mb])
        x = self.sigmoid_layer(x)
        # x = keras.layers.Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)
        out = self.lambda_layer(x)

        return out

class MLP(keras.Model):
    def __init__(self, num_users, num_movies, num_latent, min_rating, max_rating):
        super(MLP, self).__init__(name = 'MLP')

        self.num_users = num_users
        self.num_movies = num_movies
        self.num_latent = num_latent
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.user_input = keras.layers.Input(shape = (1,), name = 'user_input')
        self.user_weights = keras.layers.Embedding(self.num_users, self.num_latent, embeddings_initializer = 'he_normal',
                      embeddings_regularizer = keras.regularizers.l2(1e-6), name = 'user_weights')
        # self.user_bias = keras.layers.Embedding(self.num_users, 1, embeddings_initializer = 'he_normal',
        #               embeddings_regularizer = keras.regularizers.l2(1e-6), name = 'user_bias')

        self.movie_input = keras.layers.Input(shape = (1,), name = 'movie_input')
        self.movie_weights = keras.layers.Embedding(self.num_movies, self.num_latent, embeddings_initializer = 'he_normal',
                      embeddings_regularizer = keras.regularizers.l2(1e-6), name = 'movie_weights')
        # self.movie_bias = keras.layers.Embedding(self.num_movies, 1, embeddings_initializer = 'he_normal',
        #               embeddings_regularizer = keras.regularizers.l2(1e-6), name = 'movie_bias')

        self.reshape_weight_layer = keras.layers.Reshape((self.num_latent,))
        # self.reshape_bias_layer = keras.layers.Reshape((1,))

        self.concatenate_layer = keras.layers.Concatenate(name = 'concatenate')
        self.dropout_layer = keras.layers.Dropout(0.2)

        self.first_dense_layer = keras.layers.Dense(64, kernel_initializer = 'he_normal', name = 'first_dense')
        self.second_dense_layer = keras.layers.Dense(16, kernel_initializer = 'he_normal', name = 'second_dense')
        self.final_dense_layer = keras.layers.Dense(1, kernel_initializer = 'he_normal', name = 'third_dense')

        self.relu_layer = keras.layers.Activation('relu')
        self.sigmoid_layer = keras.layers.Activation('sigmoid')
        self.lambda_layer = LambdaLayer(min_rating, max_rating)

    def call(self, inputs, training = False):
        print(training)
        users, movies = inputs[0], inputs[1]

        # ui = self.user_input(users)
        uw = self.user_weights(users)
        uw = self.reshape_weight_layer(uw)

        # mi = self.movie_input(movies)
        mw = self.movie_weights(movies)
        mw = self.reshape_weight_layer(mw)

        ## concatenating user and movie weights
        x = self.concatenate_layer([uw, mw])
        if training:
            x = self.dropout_layer(x, training = training)

        ## passing through the dense layers
        x = self.first_dense_layer(x)
        x = self.relu_layer(x)
        if training:
            x = self.dropout_layer(x, training = training)

        x = self.second_dense_layer(x)
        x = self.relu_layer(x)
        if training:
            x = self.dropout_layer(x, training = training)

        x = self.final_dense_layer(x)
        x = self.sigmoid_layer(x)
        out = self.lambda_layer(x)

        return out

class NeuMF(keras.Model):
    def __init__(self):
        super(NeuMF, self).__init__(name = 'NeuMF')

        self.num_users = num_users
        self.num_movies = num_movies
        self.num_latent = num_latent
        self.min_rating = min_rating
        self.max_rating = max_rating

        ## LMF layers
        self.lmf_user_weights = keras.layers.Embedding(self.num_users, self.num_latent, embeddings_initializer = 'he_normal',
                      embeddings_regularizer = keras.regularizers.l2(1e-6), name = 'lmf_user_weights')
        self.lmf_user_bias = keras.layers.Embedding(self.num_users, 1, embeddings_initializer = 'he_normal',
                      embeddings_regularizer = keras.regularizers.l2(1e-6), name = 'lmf_user_bias')

        self.lmf_movie_weights = keras.layers.Embedding(self.num_movies, self.num_latent, embeddings_initializer = 'he_normal',
                      embeddings_regularizer = keras.regularizers.l2(1e-6), name = 'lmf_movie_weights')
        self.lmf_movie_bias = keras.layers.Embedding(self.num_movies, 1, embeddings_initializer = 'he_normal',
                      embeddings_regularizer = keras.regularizers.l2(1e-6), name = 'lmf_movie_bias')

        self.lmf_dot_layer = keras.layers.Dot(axes = 1)
        self.lmf_add_layer = keras.layers.Add()

        ## MLP layers
        self.mlp_user_weights = keras.layers.Embedding(self.num_users, self.num_latent, embeddings_initializer = 'he_normal',
                      embeddings_regularizer = keras.regularizers.l2(1e-6), name = 'mlp_user_weights')

        self.mlp_movie_weights = keras.layers.Embedding(self.num_movies, self.num_latent, embeddings_initializer = 'he_normal',
                      embeddings_regularizer = keras.regularizers.l2(1e-6), name = 'mlp_movie_weights')

        self.mlp_concatenate_layer = keras.layers.Concatenate(name = 'mlp_concatenate')
        self.mlp_dropout_layer = keras.layers.Dropout(0.2)

        self.mlp_first_dense_layer = keras.layers.Dense(64, kernel_initializer = 'he_normal', name = 'mlp_first_dense')
        self.mlp_second_dense_layer = keras.layers.Dense(16, kernel_initializer = 'he_normal', name = 'mlp_second_dense')
        self.mlp_third_dense_layer = keras.layers.Dense(1, kernel_initializer = 'he_normal', name = 'mlp_third_dense')
        self.mlp_relu_layer = keras.layers.Activation('relu')

        ## Common layers
        self.reshape_weight_layer = keras.layers.Reshape((self.num_latent,))
        self.reshape_bias_layer = keras.layers.Reshape((1,))

        self.sigmoid_layer = keras.layers.Activation('sigmoid')
        self.concatenate_layer = keras.layers.Concatenate(name = 'concatenate')
        self.final_dense_layer = keras.layers.Dense(1, kernel_initializer = 'he_normal', name = 'final_dense')
        self.lambda_layer = LambdaLayer(min_rating, max_rating)

    def call(self, inputs, training = False):
        users, movies = inputs[0], inputs[1]

        ## LMF layers
        lmf_uw = self.lmf_user_weights(users)
        lmf_uw = self.reshape_weight_layer(lmf_uw)
        lmf_ub = self.lmf_user_bias(users)
        lmf_ub = self.reshape_bias_layer(lmf_ub)

        lmf_mw = self.lmf_movie_weights(movies)
        lmf_mw = self.reshape_weight_layer(lmf_mw)
        lmf_mb = self.lmf_movie_bias(movies)
        lmf_mb = self.reshape_bias_layer(lmf_mb)

        lmf_x = self.lmf_dot_layer([lmf_uw, lmf_mw])
        lmf_x = self.lmf_add_layer([lmf_x, ub, mb])
        lmf_x = self.sigmoid_layer(lmf_x)

        ## MLP layers
        mlp_uw = self.mlp_user_weights(users)
        mlp_uw = self.reshape_weight_layer(mlp_uw)

        mlp_mw = self.mlp_movie_weights(movies)
        mlp_mw = self.reshape_weight_layer(mlp_mw)

        mlp_x = self.mlp_concatenate_layer([mlp_uw, mlp_mw])
        if training:
            mlp_x = self.mlp_dropout_layer(mlp_x, training = training)

        mlp_x = self.mlp_first_dense_layer(mlp_x)
        mlp_x = self.mlp_relu_layer(mlp_x)
        if training:
            mlp_x = self.mlp_dropout_layer(mlp_x, training = training)

        mlp_x = self.mlp_second_dense_layer(mlp_x)
        mlp_x = self.mlp_relu_layer(mlp_x)
        if training:
            mlp_x = self.mlp_dropout_layer(mlp_x, training = training)

        mlp_x = self.mlp_third_dense_layer(mlp_x)
        mlp_x = self.sigmoid_layer(mlp_x)

        print(lmf_x.shape)
        print(mlp_x.shape)
        ## concatenating both channels
        x = self.concatenate_layer([lmf_x, mlp_x])
        x = self.final_dense_layer(x)
        x = self.sigmoid_layer(x)
        out = self.lambda_layer(x)

LATENT_FEATURES = 128
NUM_EPOCHS = 2
BATCH_SIZE = 32
model_name = 'LMF'

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

if model_name == 'LMF':
    model = LMF(num_users, num_movies, LATENT_FEATURES, min_rating, max_rating)
    opt = keras.optimizers.Adam(lr = 0.0003)
    model.compile(loss = 'mean_squared_error', optimizer = opt)

    ## Creating check point callback
    cp_directory = 'Temp/collaborative/LMF/'
    cp_filepath = cp_directory + 'cp-{epoch:04d}.ckpt'
    cp_callback = keras.callbacks.ModelCheckpoint(filepath = cp_filepath, verbose = 1, \
                                        save_weights_only = True, period = 5)

    model.save_weights(cp_filepath.format(epoch = 0))

    model.fit(x = ratings_input, y = ratings_output, batch_size = BATCH_SIZE, epochs = 20, \
                 callbacks = [cp_callback], verbose = 1)
    model.summary()
    # model.save('Temp/collaborative.h5', save_format='tf')
    print('Model saved successfully')

    latest = tf.train.latest_checkpoint(cp_directory)
    model = LMF(num_users, num_movies, LATENT_FEATURES, min_rating, max_rating)
    # Load the previously saved weights
    model.load_weights(latest)
    print('Model loaded successfully')

    # print([(i, layer.name) for i, layer in enumerate(model.layers)])

    # user_weights = model.get_layer("user_weights").get_weights()[0]
    # movie_weights = model.get_layer("movie_weights").get_weights()[0]
    #
    # user_bias = model.get_layer("user_bias").get_weights()[0]
    # movie_bias = model.get_layer("movie_bias").get_weights()[0]
    #
    # dp_matrix = np.dot(user_weights, np.transpose(movie_weights))
    # dp_matrix = np.add(np.add(dp_matrix, user_bias), np.transpose(movie_bias))
    #
    # print(dp_matrix.shape)
    #
    # sigmoid =  np.vectorize(lambda x : 1/(1 + np.exp(-(x))))
    #
    # dp_sigmoid_matrix = sigmoid(dp_matrix)
    # dp_scaled_matrix = dp_sigmoid_matrix * (max_rating - min_rating) + min_rating
    #
    # print(dp_scaled_matrix)

    user_idx = 146 ## reduced by 1
    movies_rated = list(ratings_df[ratings_df['userId'] == user_idx]['movieId'])
    test_input = [np.array([user_idx] * num_movies), np.array(list(range(num_movies)))]

    user_ratings = model.predict(test_input).reshape(num_movies,)

    # user_ratings = dp_scaled_matrix[user_idx, :]
    print(len(user_ratings))
    user_ratings[movies_rated] = 0.5
    top10_list = np.argpartition(user_ratings, -10)[-10:]
    movie_list = [movies_idx_dict[each] for each in top10_list]

    # print(dp_scaled_matrix[user_idx, :])
    print(movie_list)
    print(movies_rated)
    # print(dp_scaled_matrix[user_idx, :][movies_rated])

elif model_name == 'MLP':
    model = MLP(num_users, num_movies, LATENT_FEATURES, min_rating, max_rating)
    opt = keras.optimizers.Adam(lr = 0.0003)
    model.compile(loss = 'mean_squared_error', optimizer = opt)

    ## Creating check point callback
    cp_directory = 'Temp/collaborative/MLP/'
    cp_filepath = cp_directory + 'cp-{epoch:04d}.ckpt'
    cp_callback = keras.callbacks.ModelCheckpoint(filepath = cp_filepath, verbose = 1, \
                                        save_weights_only = True, period = 5)

    model.save_weights(cp_filepath.format(epoch = 0))

    model.fit(x = ratings_input, y = ratings_output, batch_size = BATCH_SIZE, epochs = 50, \
                 callbacks = [cp_callback], verbose = 1)
    model.summary()
    # model.save('Temp/collaborative.h5', save_format='tf')
    print('Model saved successfully')

    latest = tf.train.latest_checkpoint(cp_directory)
    model = MLP(num_users, num_movies, LATENT_FEATURES, min_rating, max_rating)
    # Load the previously saved weights
    model.load_weights(latest)
    print('Model loaded successfully')

    user_idx = 146 ## reduced by 1
    movies_rated = list(ratings_df[ratings_df['userId'] == user_idx]['movieId'])
    test_input = [np.array([user_idx] * num_movies), np.array(list(range(num_movies)))]

    user_ratings = model.predict(test_input).reshape(num_movies,)

    print(user_ratings.shape)
    user_ratings[movies_rated] = 0.5
    top10_list = np.argpartition(user_ratings, -10)[-10:]
    movie_list = [movies_idx_dict[each] for each in top10_list]

    print(movie_list)

elif model_name == 'NeuMF':
    model = NeuMF(num_users, num_movies, LATENT_FEATURES, min_rating, max_rating)
    opt = keras.optimizers.Adam(lr = 0.0003)
    model.compile(loss = 'mean_squared_error', optimizer = opt)

    ## Creating check point callback
    cp_directory = 'Temp/collaborative/NeuMF/'
    cp_filepath = cp_directory + 'cp-{epoch:04d}.ckpt'
    cp_callback = keras.callbacks.ModelCheckpoint(filepath = cp_filepath, verbose = 1, \
                                        save_weights_only = True, period = 5)

    model.save_weights(cp_filepath.format(epoch = 0))

    model.fit(x = ratings_input, y = ratings_output, batch_size = BATCH_SIZE, epochs = 50, \
                 callbacks = [cp_callback], verbose = 1)
    model.summary()
    # model.save('Temp/collaborative.h5', save_format='tf')
    print('Model saved successfully')

    latest = tf.train.latest_checkpoint(cp_directory)
    model = NeuMF(num_users, num_movies, LATENT_FEATURES, min_rating, max_rating)
    # Load the previously saved weights
    model.load_weights(latest)
    print('Model loaded successfully')

    user_idx = 146 ## reduced by 1
    movies_rated = list(ratings_df[ratings_df['userId'] == user_idx]['movieId'])
    test_input = [np.array([user_idx] * num_movies), np.array(list(range(num_movies)))]

    user_ratings = model.predict(test_input).reshape(num_movies,)

    print(user_ratings.shape)
    user_ratings[movies_rated] = 0.5
    top10_list = np.argpartition(user_ratings, -10)[-10:]
    movie_list = [movies_idx_dict[each] for each in top10_list]

    print(movie_list)
