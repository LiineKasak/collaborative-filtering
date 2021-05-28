import pandas as pd
import numpy as np
from auxiliary import data_processing
from src.algobase import AlgoBase


class SVD(AlgoBase):
    def __init__(self, k):
        AlgoBase.__init__(self)

        number_of_singular_values = min(self.number_of_users, self.number_of_movies)
        assert (k <= number_of_singular_values), "svd received invalid number of singular values (too large)"

        self.k = k  # number of singular values to use
        self.reconstructed_matrix = np.zeros((self.number_of_movies, self.number_of_movies))

        # TODO: Initialize comet-experiment here to make sure things are tracked

    def fit(self, users, movies, predictions):
        matrix, _ = data_processing.get_data_mask(users, movies, predictions)
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

        S = np.zeros((self.number_of_movies, self.number_of_movies))
        S[:self.k, :self.k] = np.diag(s[:self.k])

        self.reconstructed_matrix = U.dot(S).dot(Vt)

    def predict(self, users, movies):
        predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)

        return predictions
