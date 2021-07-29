import numpy as np
from utils import data_processing
from models.algobase import AlgoBase
from models.svd import SVD
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from utils.dataset import DatasetWrapper
import argparse


class SVD_SGD(AlgoBase):
    """
    Running SGD on SVD initialized embeddings.
    By Surprise documentation:
    https://surprise.readthedocs.io/en/stable/matrix_factorization.html#matrix-factorization-based-algorithms
    """

    def __init__(self, params: argparse.Namespace):
        AlgoBase.__init__(self)

        self.k = params.k_singular_values  # number of singular values to use
        self.epochs = params.epochs
        self.learning_rate = params.learning_rate
        self.regularization = params.regularization
        self.verbal = params.verbal

        self.matrix = np.zeros((self.number_of_users, self.number_of_movies))
        self.reconstructed_matrix = np.zeros((self.number_of_users, self.number_of_movies))

        self.pu = np.empty((self.number_of_users, self.k))  # user embedding
        self.qi = np.empty((self.number_of_movies, self.k))  # item embedding
        self.bu = np.zeros(self.number_of_users)  # user bias
        self.bi = np.zeros(self.number_of_movies)  # item bias
        self.mu = 0

        self.enable_bias = params.enable_bias

    @staticmethod
    def default_params():
        return argparse.Namespace(k_singular_values=12, epochs=75, learning_rate=0.001, regularization=0.05,
                                  verbal=True, enable_bias=True)

    def _update_reconstructed_matrix(self):
        dot_product = self.pu.dot(self.qi.T)
        user_biases_matrix = np.reshape(self.bu, (self.number_of_users, 1))
        movie_biases_matrix = np.reshape(self.bi, (1, self.number_of_movies))
        self.reconstructed_matrix = dot_product + user_biases_matrix + movie_biases_matrix + self.mu

    def fit(self, train_data: DatasetWrapper, test_data: DatasetWrapper = None):
        users, movies, ground_truth = train_data.users, train_data.movies, train_data.ratings
        self.matrix, _ = data_processing.get_data_mask(users, movies, ground_truth)

        self.pu, self.qi = SVD.get_embeddings(self.k, self.matrix)

        indices = np.arange(len(users))

        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f'./logs/SGD_{time_string}'
        writer = SummaryWriter(log_dir)

        with tqdm(total=self.epochs * len(users), disable=not self.verbal) as pbar:
            for epoch in range(self.epochs):
                np.random.shuffle(indices)
                for user, movie in zip(users[indices], movies[indices]):
                    prediction = self.bu[user] + self.bi[movie] + np.dot(self.pu[user], self.qi[movie])
                    error = self.matrix[user, movie] - prediction

                    if self.enable_bias:
                        self.bu[user] += self.learning_rate * (error - self.regularization * self.bu[user])
                        self.bi[movie] += self.learning_rate * (error - self.regularization * self.bi[movie])

                    self.pu[user] += self.learning_rate * (
                            error * self.qi[movie] - self.regularization * self.pu[user])
                    self.qi[movie] += self.learning_rate * (
                            error * self.pu[user] - self.regularization * self.qi[movie])

                    pbar.update(1)
                self._update_reconstructed_matrix()

                predictions = self.predict(users, movies)
                rmse_loss = data_processing.get_score(predictions, ground_truth)
                writer.add_scalar('rmse', rmse_loss, epoch)

                if test_data:
                    valid_predictions = self.predict(test_data.users, test_data.movies)
                    reconstruction_rmse = data_processing.get_score(valid_predictions, test_data.ratings)
                    pbar.set_description(f'Epoch {epoch}:  rmse {rmse_loss:.4f}, val_rmse {reconstruction_rmse:.4f}')
                    writer.add_scalar('val_rmse', reconstruction_rmse, epoch)
                    rmse = reconstruction_rmse
                else:
                    pbar.set_description(f'Epoch {epoch}:  rmse {rmse_loss}')
                    rmse = rmse_loss
        return rmse, self.epochs

    def predict(self, users, movies):
        predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)
        predictions[predictions > 5] = 5
        predictions[predictions < 1] = 1
        return predictions

