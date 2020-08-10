import numpy as np
import pandas as pd

from preprocess import Preprocess

## This file has the helper function for collaborative filtering
## This include the train test splitter and the metric
class Helpers:
    def createTrainTestData(data, num_movies):
        ## taking the last record for each user
        last_data = data.groupby('userId').last().reset_index()

        ## removing the last entries from training set
        # train_data =  data[np.array(~data.isin(last_data).all(axis = 1), dtype = bool)]
        train_data = pd.concat([data, last_data]).drop_duplicates(['userId', 'movieId', 'rating', 'timestamp'], keep = False)

        ## creating the test data
        last_data['isTrue'] = 1
        test_data = last_data[['userId', 'movieId', 'isTrue']]
        for user in last_data['userId']:
            users_list = [user] * 99
            movies_list = np.random.randint(num_movies, size = 99).tolist()
            istrue_list = [0] * 99
            each_data = pd.DataFrame(list(zip(users_list, movies_list, istrue_list)), columns = ['userId', 'movieId', 'isTrue'])
            test_data = test_data.append(each_data)

        return train_data, test_data

    def topAtKMetric(test_data, K):
        pass





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


ratings_train_data, ratings_test_data = Helpers.createTrainTestData(ratings_df, num_movies)
