import argparse

import numpy as np
from utils import data_processing
from utils.dataset import DatasetWrapper
from models.algobase import AlgoBase


class SVD(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """

    def __init__(self, params : argparse.Namespace):
        AlgoBase.__init__(self,)

        number_of_singular_values = min(self.number_of_users, self.number_of_movies)
        assert (params.k_singular_values <= number_of_singular_values), "svd received invalid number of singular values (too large)"

        self.k = params.k_singular_values  # number of singular values to use
        self.reconstructed_matrix = np.zeros((self.number_of_movies, self.number_of_movies))

    @staticmethod
    def default_params():
        return argparse.Namespace(k_singular_values=5)

    def fit(self, train_data: DatasetWrapper, test_data: DatasetWrapper = None):
        matrix, _ = data_processing.get_data_mask(train_data.users, train_data.movies, train_data.ratings)
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

        S = np.zeros((self.number_of_movies, self.number_of_movies))
        S[:self.k, :self.k] = np.diag(s[:self.k])

        self.reconstructed_matrix = U.dot(S).dot(Vt)

    def predict(self, users, movies):
        predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)

        return predictions

    @staticmethod
    def get_embeddings(k, matrix):
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

        nr_movies = data_processing.get_number_of_movies()
        S_sqrt = np.zeros((nr_movies, nr_movies))
        S_sqrt[:k, :k] = np.diag(np.sqrt(s[:k]))

        U_embedding = U.dot(S_sqrt)
        Vt_embedding = S_sqrt.dot(Vt).T
        return U_embedding[:, :k], Vt_embedding[:, :k]
