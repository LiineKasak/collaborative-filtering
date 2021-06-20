from collections import defaultdict

import numpy as np
from sklearn.neighbors import NearestNeighbors,KNeighborsRegressor
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


class KNN(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """

    def __init__(self, algorithm='auto', n_neighbors=5, track_to_comet=False, method_name=None, api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self,  track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)

        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(algorithm=algorithm, n_neighbors=n_neighbors, n_jobs=-1)
        self.data_wrapper = None
        self.nearest_neighbors = None

    def compute_neighbors(self):
        user_vectors = self.data_wrapper.user_per_movie_encodings   # shape (10000, 1000)
        _, indices = self.knn.kneighbors(user_vectors, n_neighbors=self.n_neighbors)    # shape (10000, k)
        movies = self.data_wrapper.user_per_movie_encodings[indices]  # shape (10000, k, 1000)

        movies[movies == 0] = np.nan


        mean_value = np.nanmean(movies, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)

        # TODO: this could (in theory) contain nans

        self.nearest_neighbors = mean_value


    def fit(self, data_wrapper):
        self.data_wrapper = data_wrapper
        matrix = data_wrapper.movies_per_user_representation()
        self.knn.fit(matrix)


    def predict(self, users, movies):
        return self.predict_movie_mean(users, movies)

    def predict_combined_mean(self, users, movies):
        self.compute_neighbors()

        predictions = self.nearest_neighbors[tuple([users, movies])]
        for i, pred in enumerate(predictions):
            r = self.data_wrapper.rating_available(users[i], movies[i])
            if (r > 0):
                print("we have this rating")
                predictions[i] = r
            if np.isnan(pred):
                predictions[i] = (self.data_wrapper.movie_means[movies[i]] + self.data_wrapper.user_means[users[i]])/2

        return predictions

    def predict_movie_mean(self, users, movies):
            self.compute_neighbors()

            predictions = self.nearest_neighbors[tuple([users, movies])]
            for i, pred in enumerate(predictions):
                r = self.data_wrapper.rating_available(users[i], movies[i])
                if (r > 0):
                    print("we have this rating")
                    predictions[i] = r
                if np.isnan(pred):
                    predictions[i] = self.data_wrapper.movie_means[movies[i]]

            return predictions

    def predict_user_mean(self, users, movies):
        self.compute_neighbors()

        predictions = self.nearest_neighbors[tuple([users, movies])]
        for i, pred in enumerate(predictions):
            r = self.data_wrapper.rating_available(users[i], movies[i])
            if (r > 0):
                print("we have this rating")
                predictions[i] = r
            if np.isnan(pred):
                predictions[i] = self.data_wrapper.user_means[users[i]]

        return predictions
