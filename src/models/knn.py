from collections import defaultdict

import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from tqdm import tqdm

from utils import data_processing, dataset
import torch
import torch.nn as nn
import torch.optim as optim
import math
from scipy.sparse import csr_matrix

from utils.dataset import DatasetWrapper
from .algobase import AlgoBase

eps = 1e-6


class KNNUserMovie(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """

    def __init__(self, user_metric='cosine', user_algorithm='auto', movie_metric='cosine', movie_algorithm='auto', n_user_neighbors=5,  n_movie_neighbors=5, track_to_comet=False, method_name=None,
                 api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)

        self.n_user_neighbors = n_user_neighbors
        self.n_movie_neighbors = n_movie_neighbors
        self.movie_knn = NearestNeighbors(metric=movie_metric, algorithm=movie_algorithm, n_neighbors=n_movie_neighbors, n_jobs=-1)
        self.user_knn = NearestNeighbors(metric=user_metric, algorithm=user_algorithm, n_neighbors=n_user_neighbors, n_jobs=-1)
        self.knn_euclidean = KNN(metric='euclidean', algorithm='auto', n_neighbors=500)


        self.data_wrapper = None
        self.movie_nearest_neighbors = None
        self.user_nearest_neighbors = None

        self.user_weight = 0

    def compute_neighbors(self):
        # Compute nearest neighbors of users
        user_vectors = self.data_wrapper.movie_per_user_encodings  # shape (10000, 1000)
        _, indices = self.user_knn.kneighbors(user_vectors, n_neighbors=self.n_user_neighbors)  # shape (10000, k)
        movies = self.data_wrapper.movie_per_user_encodings[indices]  # shape (10000, k, 1000)
        movies[movies == 0] = np.nan
        user_mean_values = np.nanmean(movies, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)

        self.user_nearest_neighbors = user_mean_values

        # Compute nearest neighbors of movies
        movie_vectors = self.data_wrapper.user_per_movie_encodings  # shape (10000, 1000)
        _, indices = self.movie_knn.kneighbors(movie_vectors, n_neighbors=self.n_movie_neighbors)  # shape (10000, k)
        users = self.data_wrapper.user_per_movie_encodings[indices]  # shape (10000, k, 1000)
        users[users == 0] = np.nan
        movie_mean_values = np.nanmean(users, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)

        self.movie_nearest_neighbors = movie_mean_values

    def fit(self, data_wrapper):
        self.data_wrapper = data_wrapper
        user_matrix = data_wrapper.movies_per_user_representation()
        movie_matrix = data_wrapper.users_per_movie_representation()

        self.user_knn.fit(user_matrix)
        self.knn_euclidean.fit(data_wrapper)
        self.movie_knn.fit(movie_matrix)
        self.compute_neighbors()


    def predict(self, users, movies):

        user_predictions = self.user_nearest_neighbors[tuple([users, movies])]
        movie_predictions = self.movie_nearest_neighbors[tuple([movies, users])]
        euclidean_user_predictions = self.knn_euclidean.nearest_neighbors[tuple([users, movies])]

        mask_either_is_nan = np.logical_or(np.isnan(user_predictions), np.isnan(movie_predictions))
        predictions = movie_predictions
        # predictions[np.isnan(user_predictions)] = movie_predictions[np.isnan(user_predictions)]


        # predictions[~mask_either_is_nan] = self.user_weight*user_predictions[~mask_either_is_nan] + (1-self.user_weight)*movie_predictions[~mask_either_is_nan]
        # predictions[predictions < 3.5] = euclidean_user_predictions[predictions < 3.5]

        counter_nan = 0

        for i, pred in enumerate(predictions):
            r = self.data_wrapper.rating_available(users[i], movies[i])
            if (r > 0):
                print("we have this rating")
                predictions[i] = r
            if np.isnan(pred):
                counter_nan += 1

                predictions[i] = self.data_wrapper.user_means[users[i]]

        print("had to impute ", counter_nan, " movies")
        return predictions

class KNNBag(AlgoBase):
    def __init__(self, knns=None, track_to_comet=False, method_name=None, api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)
        # self.knn_cosine = None
        # self.knn_manhattan = None
        # self.knn_euclidean = None

        # if knns is None:
        #     self.knns = self.get_standard_knns()
        # else:
        #     self.knns = knns  # list of multiple KNNs to bag together

        self.knn_cosine = None
        self.knn_manhattan = None
        self.knn_euclidean = None
        self.knn_manhattan_large = None

        self.get_standard_knns()

        self.data_wrapper = None

    def fit(self, data_wrapper):
        self.data_wrapper = data_wrapper
        for knn in self.knns:
            knn.fit(data_wrapper)  # fit all the knns

    def predict(self, users, movies):

        predictions_euclidean = self.knn_euclidean.nearest_neighbors[tuple([users, movies])]
        predictions_cosine = self.knn_cosine.nearest_neighbors[tuple([users, movies])]
        # predictions_manhattan = self.knn_manhattan.nearest_neighbors[tuple([users, movies])]
        # predictions_manhattan_large = self.knn_manhattan_large.nearest_neighbors[tuple([users, movies])]

        predictions = predictions_euclidean

        predictions[predictions >= 3.5] = predictions_cosine[predictions >= 3.5]




        # fill predictions that are STILL nan with the mean
        counter_nan = 0
        for i, pred in enumerate(predictions):
            if np.isnan(pred):
                predictions[i] = self.knns[0].data_wrapper.movie_means[movies[i]]
                counter_nan += 1
        print("In the end, we had to impute ", counter_nan, " of the predictions with the movie-mean")

        predictions = predictions + self.data_wrapper.movie_bias[movies] + self.data_wrapper.user_bias[users]   # include biases

        return predictions

    def get_standard_knns(self):
        self.knn_cosine = KNN(metric='cosine', algorithm='auto', n_neighbors=500)

        # self.knn_manhattan_large = KNN(metric='manhattan', algorithm='auto', n_neighbors=100)
        # self.knn_manhattan = KNN(metric='manhattan', algorithm='auto', n_neighbors=2)
        self.knn_euclidean = KNN(metric='euclidean', algorithm='auto', n_neighbors=500)
        self.knns = [self.knn_cosine, self.knn_euclidean]



class KNN(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """

    def __init__(self, metric='cosine', algorithm='auto', n_neighbors=5, track_to_comet=False, method_name=None,
                 api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)

        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(metric=metric, algorithm=algorithm, n_neighbors=n_neighbors, n_jobs=-1)
        self.data_wrapper = None
        self.nearest_neighbors = None


        self.user_based = True
        # self.item_based = False

    def compute_neighbors(self):

        if self.user_based:
            user_vectors = self.data_wrapper.movie_per_user_encodings  # shape (10000, 1000)
            _, indices = self.knn.kneighbors(user_vectors, n_neighbors=self.n_neighbors)  # shape (10000, k)
            movies = self.data_wrapper.movie_per_user_encodings[indices]  # shape (10000, k, 1000)
            movies[movies == 0] = np.nan
            user_mean_values = np.nanmean(movies, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)

            self.nearest_neighbors = user_mean_values

        else:
            movie_vectors = self.data_wrapper.user_per_movie_encodings  # shape (10000, 1000)
            _, indices = self.knn.kneighbors(movie_vectors,
                                                   n_neighbors=self.n_neighbors)  # shape (10000, k)
            users = self.data_wrapper.user_per_movie_encodings[indices]  # shape (10000, k, 1000)
            users[users == 0] = np.nan
            movie_mean_values = np.nanmean(users, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)

            self.nearest_neighbors = movie_mean_values

    def fit(self, data_wrapper):
        self.data_wrapper = data_wrapper
        matrix = data_wrapper.movies_per_user_representation()
        self.knn.fit(matrix)

        self.compute_neighbors()

    def predict(self, users, movies):
        return self.predict_movie_mean(users, movies)



    def predict_movie_mean(self, users, movies):
        self.compute_neighbors()

        if(self.user_based):
            predictions = self.nearest_neighbors[tuple([users, movies])]
        else:
            predictions = self.nearest_neighbors[tuple([movies, users])]

        counter_nan = 0
        for i, pred in enumerate(predictions):
            r = self.data_wrapper.rating_available(users[i], movies[i])
            if (r > 0):
                print("we have this rating")
                predictions[i] = r
            if np.isnan(pred):
                counter_nan += 1

                predictions[i] = self.data_wrapper.movie_means[movies[i]]
        print("In the end, we had to impute ", counter_nan, " of the predictions with the movie-mean")

        return predictions


