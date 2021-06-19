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

    def __init__(self, track_to_comet=False, method_name=None, api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)

        self.knn = NearestNeighbors(algorithm='ball_tree', n_neighbors=5, n_jobs=-1)
        self.data_wrapper = None
        self.nearest_neighbors = None

    def compute_neighbors(self):
        self.nearest_neighbors = np.zeros((self.data_wrapper.num_users, self.data_wrapper.num_movies))

        user_vectors = self.data_wrapper.user_per_movie_encodings
        _, indices = self.knn.kneighbors(user_vectors, n_neighbors=5)
        movies = self.data_wrapper.user_per_movie_encodings[indices]    # shape (num_users, 5, 1000)
        # self.nearest_neighbors = np.average(movies, axis=1)             # shape (num_users, 1000) (average over the 5 neighbors)
        #TODO: we only want nonzero elements..
        self.nearest_neighbors = movies


    def fit(self, data_wrapper):
        self.data_wrapper = data_wrapper
        matrix = data_wrapper.movies_per_user_representation()
        self.knn.fit(matrix)

        self.compute_neighbors()


    def predict(self, users, movies):
        print('predicting...')
        predictions = np.zeros(len(users))
        tuples = list(zip(users, movies))
        for idx, (user, movie) in tqdm(enumerate(tuples), desc='prediction..'):
            relevant_entries = self.nearest_neighbors[user, :, movie]
            if len(relevant_entries) == 0:
                relevant_entries = self.data_wrapper.movie_means[movie]
            predictions[idx] = np.mean(relevant_entries[relevant_entries != 0])

        return predictions
