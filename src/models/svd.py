import numpy as np
from src.utils import data_processing
from src.models.algobase import AlgoBase


class SVD(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """
    def __init__(self, k_singular_values, track_to_comet=False):
        AlgoBase.__init__(self, track_to_comet)

        number_of_singular_values = min(self.number_of_users, self.number_of_movies)
        assert (k_singular_values <= number_of_singular_values), "svd received invalid number of singular values (too large)"

        self.k = k_singular_values  # number of singular values to use
        self.reconstructed_matrix = np.zeros((self.number_of_movies, self.number_of_movies))

    def fit(self, data_wrapper, val_users, val_movies):
        users, movies, predictions = data_wrapper.users, data_wrapper.movies, data_wrapper.ratings
        matrix, mask = data_processing.get_data_mask(data_wrapper=data_wrapper, impute='fancy', val_users=val_users, val_movies=val_movies)
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

        S = np.zeros((self.number_of_movies, self.number_of_movies))
        S[:self.k, :self.k] = np.diag(s[:self.k])

        self.reconstructed_matrix = U.dot(S).dot(Vt)
        np.copyto(dst=self.reconstructed_matrix, src=matrix, where=mask.astype(bool))  # this does not improve anything

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


if __name__ == '__main__':
    data_pd = data_processing.read_data()
    svd = SVD(12)
    rsmes = svd.cross_validate(data_pd)
    print(rsmes)
    print(np.mean(rsmes))