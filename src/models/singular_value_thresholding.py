import pandas as pd
import numpy as np
import math
from auxiliary import data_processing
from src.algobase import AlgoBase


def shrink(Y, tau):
    U, S, V = np.linalg.svd(Y, full_matrices=False)
    S = np.maximum(S - tau, 0)
    X = U * S @ V

    return X

    # map X to the space of observed entries (Omega)


def map_to_omega(X, mask):
    return mask * X


class SVT(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """

    def __init__(self, max_iterations=1000, delta=10.1958, eps=0.01, tau=27500, track_to_comet=False):
        AlgoBase.__init__(self, track_to_comet)

        number_of_singular_values = min(self.number_of_users, self.number_of_movies)

        self.max_iterations = max_iterations  # number of singular values to use
        self.eps = eps
        self.tau = tau
        self.delta = delta

        self.reconstructed_matrix = np.zeros((self.number_of_movies, self.number_of_movies))

    def fit(self, users, movies, predictions):
        matrix, mask = data_processing.get_data_mask(users, movies, predictions)

        Y = np.zeros_like(matrix)

        for k in range(self.max_iterations):
            X = shrink(Y, self.tau)
            Y += self.delta * map_to_omega(matrix - X, mask)

            recon_error = np.linalg.norm(map_to_omega(X - matrix, mask)) / np.linalg.norm(
                map_to_omega(matrix, mask))

            if recon_error < self.eps:
                print("svt terminated early, at k = ", k)
                break

        self.reconstructed_matrix = X

    def predict(self, users, movies):
        predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)

        return predictions


class ASVT(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """

    def __init__(self, max_iterations=50, delta=10.1958, a=0.001176952, b=60000, eps=0.10, track_to_comet=False):
        AlgoBase.__init__(self, track_to_comet)

        number_of_singular_values = min(self.number_of_users, self.number_of_movies)

        self.max_iterations = max_iterations  # number of singular values to use
        self.reconstructed_matrix = np.zeros((self.number_of_movies, self.number_of_movies))
        self.eps = eps

        self.deltas = np.full(shape=self.max_iterations, fill_value=delta)
        self.reconstructed_matrix = np.zeros((self.number_of_movies, self.number_of_movies))

        self.a = a
        self.b = b

    def fit(self, users, movies, predictions):
        matrix, mask = data_processing.get_data_mask(users, movies, predictions)

        Y = np.zeros_like(matrix)

        for k in range(self.max_iterations):
            tau_k = self.b * math.exp(-self.a * k)
            X = shrink(Y, tau_k)
            Y += self.deltas[k] * map_to_omega(matrix - X, mask)

            recon_error = np.linalg.norm(map_to_omega(X - matrix, mask)) / np.linalg.norm(
                map_to_omega(matrix, mask))

            if recon_error < self.eps:
                print("asvt terminated early, at k = ", k)
                break

        self.reconstructed_matrix = X

    def predict(self, users, movies):
        predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)

        return predictions
