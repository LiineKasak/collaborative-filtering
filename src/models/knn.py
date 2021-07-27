import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils import data_processing
from models.algobase import AlgoBase
from models.svd_sgd import SVD_SGD

eps = 1e-6


class KNNImprovedSVDEmbeddings(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """

    def __init__(self, params: argparse.Namespace):
        AlgoBase.__init__(self)

        self.n_neighbors = params.n_neighbors
        self.knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=self.n_neighbors, n_jobs=-1)
        self.data_wrapper = None
        self.nearest_neighbors = None

        self.k = params.k_singular_values

        self.movie_embeddings, self.user_embeddings, self.user_biases_matrix, self.movie_biases_matrix = None, None, None, None
        self.svd = SVD_SGD(params)

    @staticmethod
    def default_params():
        return argparse.Namespace(k_singular_values=5, epochs=10, verbal=True, n_neighbors=5)

    def compute_neighbors(self):
        user_vectors = self.user_embeddings  # shape (10000, 1000)
        _, indices = self.knn.kneighbors(user_vectors, n_neighbors=self.n_neighbors)  # shape (10000, k)

        low_rank = self.user_embeddings.dot(self.movie_embeddings.T)
        np.copyto(low_rank, self.data_wrapper.data_matrix, where=self.data_wrapper.mask.astype(bool))

        ratings = low_rank[indices]

        nearest_neighbors = np.mean(ratings, axis=1)
        np.copyto(nearest_neighbors, self.data_wrapper.data_matrix,
                  where=self.data_wrapper.mask.astype(bool))

        self.nearest_neighbors = nearest_neighbors

    def fit(self, data_wrapper):
        self.data_wrapper = data_wrapper

        self.svd.fit(data_wrapper)
        self.user_embeddings = self.svd.pu
        self.movie_embeddings = self.svd.qi
        self.user_biases_matrix = np.reshape(self.svd.bu, (self.number_of_users, 1))
        self.movie_biases_matrix = np.reshape(self.svd.bi, (1, self.number_of_movies))

        self.user_biases = self.svd.bu
        self.movie_biases = self.svd.bi
        self.knn.fit(self.user_embeddings)

        self.knn.fit(self.movie_embeddings)

        print("computing nearest neighbours...")
        self.compute_neighbors()

    def predict(self, users, movies):
        pred_matrix = self.nearest_neighbors + self.user_biases_matrix + self.movie_biases_matrix
        predictions = data_processing.extract_prediction_from_full_matrix(pred_matrix, users, movies)

        nan_mask = np.isnan(predictions)
        np.copyto(predictions, self.data_wrapper.movie_means[movies], where=nan_mask)  # impute nan values

        predictions[predictions < 1] = 1
        predictions[predictions > 5] = 5

        return predictions
