import numpy as np
from sklearn.model_selection import train_test_split
from utils import data_processing
from utils.dataset import DatasetWrapper
from models.algobase import AlgoBase
from models.svd import SVD
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

EPSILON = 1e-5


class SVT_SVD(AlgoBase):
    """
    Running SGD on SVD on pu qi paramteres only. Baseline parameters bu and bi are preloaded
    and were calculated with ALS. Init matrix is result of Singular Value Thresholding algorithm.
    """

    def __init__(self, k_singular_values=12, epochs=43, learning_rate=0.001, regularization=0.05, submit=False , verbal=False,
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

        if(submit):
            with open(self.directory_path + '/data/bu_bi_no_mean.npy', 'rb') as f:
                self.bu = np.load(f, allow_pickle=True)
                self.bi = np.load(f, allow_pickle=True)
        else: 
            with open(self.directory_path + '/data/bu_bi_no_mean_trainonly.npy', 'rb') as f:
                self.bu = np.load(f, allow_pickle=True)
                self.bi = np.load(f, allow_pickle=True)

        self.mu = 0

    def _update_reconstructed_matrix(self):
        dot_product = self.pu.dot(self.qi.T)
        user_biases_matrix = np.reshape(self.bu, (self.number_of_users, 1))
        movie_biases_matrix = np.reshape(self.bi, (1, self.number_of_movies))
        self.reconstructed_matrix = dot_product + user_biases_matrix + movie_biases_matrix + self.mu

    def fit(self, train_data: DatasetWrapper, test_data: DatasetWrapper = None):
        users = train_data.users
        movies = train_data.movies
        ground_truth = train_data.ratings

        with open(self.directory_path + '/data/svt_Xopt_Yk_sh100k_5000_to_5500.npy', 'rb') as f:
            self.matrix = np.load(f, allow_pickle=True)
            Yk = np.load(f, allow_pickle=True)
        
        self.pu, self.qi = SVD.get_embeddings(self.k, self.matrix)

        run_validation = test_data is not None
        indices = np.arange(len(users))
        # global_mean = np.mean(self.matrix[np.nonzero(self.matrix)])

        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f'./logs/SGD_SVT_100sh_k12_5500init_NOMEANBL_ep100_{time_string}'
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
                    valid_predictions = self.predict(test_data.users, test_data.movies)
                    reconstruction_rmse = data_processing.get_score(valid_predictions, test_data.ratings)
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
    epochs = 43

    submit = False

    sgd = SVT_SVD(k_singular_values=k, epochs=epochs, verbal=True, submit=submit)

    if submit:
        data = DatasetWrapper(data_pd)
        sgd.fit(data)
        sgd.predict_for_submission(f'svd_sgd_norm_k{k}_{epochs}')
    else:
        train_pd, test_pd = train_test_split(data_pd, train_size=0.9, random_state=42)
        train_data, test_data = DatasetWrapper(train_pd), DatasetWrapper(test_pd)
        users, movies, predictions = data_processing.extract_users_items_predictions(train_pd)
        sgd.fit(train_data, test_data)
