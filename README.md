# RecommendationSystem

An attempt to build a recommendation system from MovieLens20M dataset using TensorFlow.

Here, I implemented both Content Filtering and Collaborative Filtering approaches.

## Content Filtering:

For content filtering, I have created a TFIDF vector based on the tags extracted from **tags.csv** file and from the genre column in **movies.csv**. This TFIDF vector is compressed using an autoencoder to 300 dimensions. Also, the vector for each movie is augmented with 300 dimensional spacy sentence vector on the one-line movie description scraped from the imdb page of that movie.


The weighted average of movie vectors of each movie watched by an user is taken based on the ratings given by that user. This
weighted movie vector is used in the content filtering approach, where 10 movies with highest cosine similarity to the weighted
movie vector is recommended.

## Collaborative Filtering:

For collaborative filtering, I have created three models using tensorflow. They are,

1. Linear Matrix Factorization (LMF) Model
2. Multi Layer Perceptron (MLP) Model
3. Neural Matrix Factorization (NeuMF) Model

The performance of each model analyzed using a metric called Top@K, which is the proportion of times the held out last movie actually watched by the user is recommended by the model within top K Recommendations. NeuMF model has the best performance compared to its counterparts with a Top@K score of 0.49.


The project used MovieLens dataset and the data is available in this [location](https://grouplens.org/datasets/movielens/20m/).
