import numpy as np
from utils import data_processing
from .algobase import AlgoBase


class SVD(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """
    def __init__(self, k_singular_values, track_to_comet=False, method_name=None, api_key="rISpuwcLQoWU6qan4jRCAPy5s", projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key, projectname=projectname, workspace=workspace, tag=tag)

        number_of_singular_values = min(self.number_of_users, self.number_of_movies)
        assert (k_singular_values <= number_of_singular_values), "svd received invalid number of singular values (too large)"

        self.k = k_singular_values  # number of singular values to use
        self.reconstructed_matrix = np.zeros((self.number_of_movies, self.number_of_movies))

    def fit(self, users, movies, predictions):
        matrix, _ = data_processing.get_data_mask(users, movies, predictions)
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

        S = np.zeros((self.number_of_movies, self.number_of_movies))
        S[:self.k, :self.k] = np.diag(s[:self.k])

        self.reconstructed_matrix = U.dot(S).dot(Vt)

    def predict(self, users, movies):
        predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)

        return predictions
