import numpy as np
import pandas as pd
import tensorflow as tf

from preprocess import Preprocess
from collaborativefiltering import NeuMF, LMF, MLP

## This file has the helper function for collaborative filtering
## This include the train test splitter and the metric
class Helpers:
    def createTrainTestData(data, num_movies):
        ## taking the last record for each user
        last_data = data.groupby('userId').last().reset_index()

        ## removing the last entries from training set
        # train_data =  data[np.array(~data.isin(last_data).all(axis = 1), dtype = bool)]
        train_data = pd.concat([data, last_data]).drop_duplicates(['userId', 'movieId', 'rating', 'timestamp'], \
                                                        keep = False)

        ## creating the test data
        last_data['isTrue'] = 1
        test_data = last_data[['userId', 'movieId', 'isTrue']]
        for user in last_data['userId']:
            users_list = [user] * 99
            movies_list = np.random.randint(num_movies, size = 99).tolist()
            istrue_list = [0] * 99
            each_data = pd.DataFrame(list(zip(users_list, movies_list, istrue_list)), \
                                columns = ['userId', 'movieId', 'isTrue'])
            test_data = test_data.append(each_data)

        return train_data, test_data

    def topAtKMetric(test_data, K, num_users, model):
        count = 0
        for user_id in range(0, num_users):
            temp_data = test_data[test_data['userId'] == user_id]
            already_watched = temp_data[temp_data['isTrue'] == 1]['movieId'].to_numpy()[0]
            test_input = [temp_data['userId'].to_numpy(), temp_data['movieId'].to_numpy()]

            user_ratings = model.predict(test_input).reshape(100,)

            topK_list = np.argpartition(user_ratings, -K)[-K:]
            topK_movie_list = temp_data.iloc[topK_list]['movieId'].to_list()

            if already_watched in topK_movie_list:
                count += 1

        return count/num_users



LATENT_FEATURES = 128

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

ratings_train_data, ratings_test_data = Helpers.createTrainTestData(ratings_df, num_movies)

ratings_input = [ratings_train_data['userId'].to_numpy(), ratings_train_data['movieId'].to_numpy()]
ratings_output = ratings_train_data[['rating']].to_numpy()

cp_directory = 'Temp/collaborative/NeuMF/'
latest = tf.train.latest_checkpoint(cp_directory)
model = NeuMF(num_users, num_movies, LATENT_FEATURES, 0.5, 5)

# Load the previously saved weights
model.load_weights(latest)
print('Model loaded successfully')

topKscore = Helpers.topAtKMetric(ratings_test_data, 10, num_users, model)
print(topKscore)
