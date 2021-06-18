import numpy as np
from utils import data_processing
import torch
import torch.nn as nn
import torch.optim as optim
import math
from .algobase import AlgoBase

eps = 1e-6


class SVD(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """

    def __init__(self, k_singular_values, track_to_comet=False, method_name=None, api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)

        number_of_singular_values = min(self.number_of_users, self.number_of_movies)
        assert (
                    k_singular_values <= number_of_singular_values), "svd received invalid number of singular values (too large)"

        self.k = k_singular_values  # number of singular values to use
        self.reconstructed_matrix = np.zeros((self.number_of_movies, self.number_of_movies))

    def fit(self, users, movies, predictions):
        matrix, _ = data_processing.get_imputed_data_mask(users, movies, predictions)
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

        S = np.zeros((self.number_of_movies, self.number_of_movies))
        S[:self.k, :self.k] = np.diag(s[:self.k])

        self.reconstructed_matrix = U.dot(S).dot(Vt)

    def predict(self, users, movies):
        predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)

        return predictions


class NMF(AlgoBase):
    """ Prediction based on dimensionality reduction through Non-Negative Matrix Factorization """
    def __init__(self, max_iterations=1000, rank_k=5, track_to_comet=False, method_name=None,
                 api_key="rISpuwcLQoWU6qan4jRCAPy5s", projectname="cil-experiments", workspace="veroniquek",
                 tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)

        self.k = rank_k  # number of singular values to use
        self.reconstructed_matrix = np.zeros((self.number_of_movies, self.number_of_movies))
        self.max_iterations = max_iterations

    def fit(self, users, movies, predictions):
        X, _ = data_processing.get_imputed_data_mask(users, movies, predictions)
        X = torch.from_numpy(X)
        X = X.float()

        K = self.k  # Number of features

        # Initialize W and Z from a uniform distribution U(0, 1) Additionally, the matrices are scaled by 1/sqrt(K)
        # to make the variance of the resulting product independent of K
        U = torch.rand(X.shape[0], K).mul_(1 / math.sqrt(K)).requires_grad_()
        V = torch.rand(X.shape[1], K).mul_(1 / math.sqrt(K)).requires_grad_()

        optimizer = optim.SGD([U, V], lr=0.9, momentum=0.8)  # need momentum to degrease the loss faster
        loss_fn = nn.MSELoss()

        for i in range(self.max_iterations):
            optimizer.zero_grad()
            loss = torch.sqrt(loss_fn(U @ V.t(), X) + eps)  # add eps in case of 0
            loss.backward()
            optimizer.step()

            # Project onto valid set of solutions
            U.data.clamp_(min=0)
            V.data.clamp_(min=0)

        self.reconstructed_matrix = U @ V.t()

    def predict(self, users, movies):
        predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)
        return predictions


class ALS(AlgoBase):
    """ Prediction based on dimensionality reduction through Alternating Least Squares """
    def __init__(self, max_iterations=1000, rank_k=5, track_to_comet=False, method_name=None,
                 api_key="rISpuwcLQoWU6qan4jRCAPy5s", projectname="cil-experiments", workspace="veroniquek",
                 tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)

        self.k = rank_k  # number of singular values to use
        self.reconstructed_matrix = np.zeros((self.number_of_movies, self.number_of_movies))
        self.max_iterations = max_iterations

    def fit(self, users, movies, predictions):
        X, _ = data_processing.get_imputed_data_mask(users, movies, predictions)
        X = torch.from_numpy(X)

        X = X.float()

        (n, p) = X.shape

        U = torch.rand(n, self.k).mul_(1 / math.sqrt(self.k)).requires_grad_()
        V = torch.rand(p, self.k).mul_(1 / math.sqrt(self.k)).requires_grad_()

        # optimizer = optim.SGD([U, V], lr=0.005)
        # loss_fn = nn.MSELoss()
        optimizer_U = optim.Adam([U], lr=0.005)
        optimizer_V = optim.Adam([V], lr=0.005)

        loss_fn_U = nn.MSELoss()
        loss_fn_V = nn.MSELoss()

        for i in range(self.max_iterations):
            # optimize U
            optimizer_U.zero_grad()
            loss_u = torch.sqrt(loss_fn_U(U @ V.t(), X) + eps)  # add eps in case of 0
            loss_u.backward()
            optimizer_U.step()

            # optimize V
            optimizer_V.zero_grad()
            loss_v = torch.sqrt(loss_fn_V(U @ V.t(), X) + eps)  # add eps in case of 0
            loss_v.backward()
            optimizer_V.step()

            # Project onto valid set of solutions
            U.data.clamp_(min=0)
            V.data.clamp_(min=0)

        self.reconstructed_matrix = U @ V.t()

    def predict(self, users, movies):
        predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)
        return predictions
