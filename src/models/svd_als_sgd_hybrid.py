import numpy as np
from sklearn.model_selection import train_test_split
from utils import data_processing
from models.algobase import AlgoBase
from models.svd import SVD
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from collections import defaultdict

EPSILON = 1e-5


class SVD_ALS_SGD(AlgoBase):
    """
    Running SGD on SVD on pu qi paramteres only. Baseline parameters bu and bi are preloaded
    and were calculated with ALS. Init matrix is result of Singular Value Thresholding algorithm.
    """

    def __init__(self, k_singular_values=12, epochs=75, learning_rate=0.001, regularization=0.05, verbal=False,
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

        self.directory_path = data_processing.get_project_directory()

        self.mu = 0

    def _update_reconstructed_matrix(self):
        dot_product = self.pu.dot(self.qi.T)
        user_biases_matrix = np.reshape(self.bu, (self.number_of_users, 1))
        movie_biases_matrix = np.reshape(self.bi, (1, self.number_of_movies))
        self.reconstructed_matrix = dot_product + user_biases_matrix + movie_biases_matrix + self.mu

    def create_adjacency_lists(self, train_user_ids, train_movie_ids, train_ratings):
        ur = defaultdict(list)
        ir = defaultdict(list)

        for i in range(len(train_user_ids)):
            ur[train_user_ids[i]].append((train_movie_ids[i], train_ratings[i]))
    
        for i in range(len(train_movie_ids)):
            ir[train_movie_ids[i]].append((train_user_ids[i], train_ratings[i]))

        return (ur, ir)

    def fit_model_baseline_als(self, ur, ir, num_epochs):
        reg_u = 15.0
        reg_i = 10.0

        for dummy in range(num_epochs):
            for u in ur:
                sum_u = 0
                for idx in range(len(ur[u])):
                    item_idx = ur[u][idx][0]
                    rating = ur[u][idx][1]
                    sum_u += rating - self.bi[item_idx]
                self.bu[u] = (sum_u/(reg_u+len(ur[u])))

            for i in ir:
                sum_i = 0
                for idx in range(len(ir[i])):
                    user_idx = ir[i][idx][0]
                    rating = ir[i][idx][1]
                    sum_i += rating - self.bu[user_idx]
                self.bi[i] = (sum_i/(reg_i+len(ir[i])))

    def fit(self, users, movies, ground_truth, valid_users=None, valid_movies=None, valid_ground_truth=None):
        
        self.matrix, _ = data_processing.get_data_mask(users, movies, ground_truth)

        (ur, ir) = self.create_adjacency_lists(users, movies, ground_truth)
        self.fit_model_baseline_als(ur, ir, 50)

        self.pu, self.qi = SVD.get_embeddings(self.k, self.matrix)

        run_validation = valid_users is not None and valid_movies is not None and valid_ground_truth is not None
        indices = np.arange(len(users))
        # global_mean = np.mean(self.matrix[np.nonzero(self.matrix)])

        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f'./logs/SVD_ALS_SGD_{time_string}'
        writer = SummaryWriter(log_dir)

        with tqdm(total=self.epochs * len(users), disable=not self.verbal) as pbar:
            for epoch in range(self.epochs):
                np.random.shuffle(indices)

                for user, movie in zip(users[indices], movies[indices]):
                    prediction = self.bu[user] + self.bi[movie] + np.dot(self.pu[user], self.qi[movie])
                    error = self.matrix[user, movie] - prediction

                    self.pu[user] += self.learning_rate * (
                            error * self.qi[movie] - self.regularization * self.pu[user])
                    self.qi[movie] += self.learning_rate * (
                            error * self.pu[user] - self.regularization * self.qi[movie])

                    pbar.update(1)
                self._update_reconstructed_matrix()

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

    def predict(self, users, movies):
        predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)
        predictions[predictions > 5] = 5
        predictions[predictions < 1] = 1
        return predictions


if __name__ == '__main__':
    data_pd = data_processing.read_data()
    k = 12
    epochs = 75

    submit = False

    sgd = SVD_ALS_SGD(k_singular_values=k, epochs=epochs, verbal=True)

    if submit:
        users, movies, predictions = data_processing.extract_users_items_predictions(data_pd)
        sgd.fit(users, movies, predictions)
        sgd.predict_for_submission(f'svd_als_sgd_k{k}_{epochs}')
    else:
        train_pd, test_pd = train_test_split(data_pd, train_size=0.9, random_state=42)
        users, movies, predictions = data_processing.extract_users_items_predictions(train_pd)
        val_users, val_movies, val_predictions = data_processing.extract_users_items_predictions(test_pd)
        sgd.fit(users, movies, predictions, val_users, val_movies, val_predictions)
