import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import data_processing
from src.models.algobase import AlgoBase
from src.models.svd import SVD
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

EPSILON = 1e-5


class SGD(AlgoBase):
    """
    Running SGD on SVD initialized embeddings.
    By Surprise documentation:
    https://surprise.readthedocs.io/en/stable/matrix_factorization.html#matrix-factorization-based-algorithms
    """

    def __init__(self, k_singular_values=17, epochs=100, learning_rate=0.001, regularization=0.05, verbal=False,
                 track_to_comet=False):
        AlgoBase.__init__(self, track_to_comet)

        self.k = k_singular_values  # number of singular values to use
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.verbal = verbal

        self.matrix = np.zeros((self.number_of_users, self.number_of_movies))
        self.reconstructed_matrix = np.zeros((self.number_of_users, self.number_of_movies))

        self.pu = np.empty((self.number_of_users, self.k))  # user embedding
        self.qi = np.empty((self.number_of_movies, self.k))  # item embedding
        self.bu = np.zeros(self.number_of_users)  # user bias
        self.bi = np.zeros(self.number_of_movies)  # item bias
        self.mu = 0

    def _update_reconsturcted_matrix(self):
        dot_product = self.pu.dot(self.qi.T)
        user_biases_matrix = np.reshape(self.bu, (self.number_of_users, 1))
        movie_biases_matrix = np.reshape(self.bi, (1, self.number_of_movies))
        self.reconstructed_matrix = dot_product + user_biases_matrix + movie_biases_matrix + self.mu

    def _train(self, users, movies, ground_truth, valid_users=None, valid_movies=None, valid_ground_truth=None):
        self.matrix, _ = data_processing.get_data_mask(users, movies, ground_truth)
        self.pu, self.qi = SVD.get_embeddings(self.k, self.matrix)

        run_validation = valid_users is not None and valid_movies is not None and valid_ground_truth is not None
        indices = np.arange(len(users))
        mean = np.mean(self.matrix[np.nonzero(self.matrix)])

        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f'./logs/SGD_{time_string}'
        writer = SummaryWriter(log_dir)
        previous_loss = 100

        with tqdm(total=self.epochs * len(users), disable=not self.verbal) as pbar:
            for epoch in range(self.epochs):
                np.random.shuffle(indices)
                for user, movie in zip(users[indices], movies[indices]):
                    lagrangian = self.bu[user] + self.bi[movie] - mean
                    u_values = self.pu[user]
                    z_values = self.qi[movie]
                    residual = self.matrix[user, movie] - self.bu[user] - self.bi[movie] - np.dot(u_values, z_values)

                    self.pu[user] *= (1 - self.regularization * self.learning_rate)
                    self.pu[user] += self.learning_rate * residual * z_values
                    self.qi[movie] *= (1 - self.regularization * self.learning_rate)
                    self.qi[movie] += self.learning_rate * residual * z_values

                    self.bu[user] += self.learning_rate * (residual - self.regularization * lagrangian)
                    self.bi[movie] += self.learning_rate * (residual - self.regularization * lagrangian)
                    pbar.update(1)
                self._update_reconsturcted_matrix()

                predictions = self.predict(users, movies)
                rmse_loss = data_processing.get_score(predictions, ground_truth)
                writer.add_scalar('rmse', rmse_loss, epoch)

                if run_validation:
                    valid_predictions = self.predict(valid_users, valid_movies)
                    reconstruction_rmse = data_processing.get_score(valid_predictions, valid_ground_truth)
                    pbar.set_description(f'Epoch {epoch}:  rmse {rmse_loss:.4f}, val_rmse {reconstruction_rmse:.4f}')
                    writer.add_scalar('val_rmse', reconstruction_rmse, epoch)
                else:
                    pbar.set_description(f'Epoch {epoch}:  rmse {rmse_loss}')

    def fit(self, users, movies, predictions):
        self.matrix, _ = data_processing.get_data_mask(users, movies, predictions)
        # TODO: handle variance

        self._train(users, movies, predictions)

    def predict(self, users, movies):
        predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)
        predictions[predictions > 5] = 5
        predictions[predictions < 1] = 1
        return predictions


if __name__ == '__main__':
    data_pd = data_processing.read_data()
    train_pd, test_pd = train_test_split(data_pd, train_size=0.9, random_state=42)
    sgd = SGD(k_singular_values=10, epochs=4, verbal=True)

    users, movies, predictions = data_processing.extract_users_items_predictions(data_pd)
    sgd.fit(users, movies, predictions)
    sgd.predict_for_submission('basic_svd_sgd')

    # users, movies, predictions = data_processing.extract_users_items_predictions(train_pd)
    # val_users, val_movies, val_predictions = data_processing.extract_users_items_predictions(test_pd)
    # sgd._train(users, movies, predictions, val_users, val_movies, val_predictions)
