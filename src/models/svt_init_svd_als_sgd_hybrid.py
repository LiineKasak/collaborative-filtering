import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import data_processing
from models.algobase import AlgoBase
from models.svd import SVD
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import DatasetWrapper

import time
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import KFold

EPSILON = 1e-5


class SVT_INIT_SVD_ALS_SGD(AlgoBase):
    """
    Running SGD on SVD on pu qi paramteres only. Baseline parameters bu and bi are preloaded
    and were calculated with ALS. Init matrix is result of Singular Value Thresholding algorithm.
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

        self.directory_path = data_processing.get_project_directory()
        self.cv_svt_matrix_filenames = [
            'svt_Xopt_Yk_sh100k_0_to_2000_CV_20210723-005714.npy',
            'svt_Xopt_Yk_sh100k_0_to_2000_CV_20210723-080627.npy',
            'svt_Xopt_Yk_sh100k_0_to_2000_CV_20210723-151440.npy',
            'svt_Xopt_Yk_sh100k_0_to_2000_CV_20210723-222641.npy',
            'svt_Xopt_Yk_sh100k_0_to_2000_CV_20210724-054423.npy'
        ]
        self.svt_init_matrix_path = ''
        self.svt_matrix = np.zeros((self.number_of_users, self.number_of_movies))

        self.mu = 0

    @staticmethod
    def default_params():
        return argparse.Namespace(k_singular_values=12, epochs=43, learning_rate=0.001, regularization=0.05,
                                  verbal=False)

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
                self.bu[u] = (sum_u / (reg_u + len(ur[u])))

            for i in ir:
                sum_i = 0
                for idx in range(len(ir[i])):
                    user_idx = ir[i][idx][0]
                    rating = ir[i][idx][1]
                    sum_i += rating - self.bu[user_idx]
                self.bi[i] = (sum_i / (reg_i + len(ir[i])))

    def fit(self, train_data: DatasetWrapper, test_data: DatasetWrapper = None):
        users, movies, ground_truth = train_data.users, train_data.movies, train_data.ratings

        with open(self.directory_path + self.svt_init_matrix_path, 'rb') as f:
            self.svt_matrix = np.load(f, allow_pickle=True)
            # Yk = np.load(f, allow_pickle=True) #this is not used but in the file since it was needed for svt

        self.matrix, _ = data_processing.get_data_mask(users, movies, ground_truth)

        (ur, ir) = self.create_adjacency_lists(users, movies, ground_truth)
        self.fit_model_baseline_als(ur, ir, 50)

        self.pu, self.qi = SVD.get_embeddings(self.k, self.svt_matrix)

        indices = np.arange(len(users))

        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f'./logs/SVT_INIT_SVD_ALS_SGD_{time_string}'
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

                if test_data:
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

    def cross_validate(self, data_pd, folds=5, random_state=42):
        """ Run Crossvalidation using kfold, taking a pandas-dataframe of the raw data as input
            (as it is read in from the .csv file) """
        kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)

        rmses = []

        bar = tqdm(total=folds, desc='cross_validation')
        counter = 0
        for train_index, test_index in kfold.split(data_pd):
            train_users, train_movies, train_predictions = data_processing.extract_users_items_predictions(
                data_pd.iloc[train_index])
            val_users, val_movies, val_predictions = data_processing.extract_users_items_predictions(
                data_pd.iloc[test_index])
            train_data = DatasetWrapper(train_users, train_movies, train_predictions)
            val_data = DatasetWrapper(val_users, val_movies, val_predictions)

            self.svt_init_matrix_path = '/data/phase1_precomputed_matrix/' + self.cv_svt_matrix_filenames[counter]

            self.fit(train_data=train_data, test_data=val_data)
            counter += 1

            predictions = self.predict(val_users, val_movies)
            rmses.append(data_processing.get_score(predictions, val_predictions))

            bar.update()

        bar.close()

        mean_rmse = np.mean(rmses)
        # track mean rmses to comet if we are tracking
        if self.track_on_comet:
            self.comet_experiment.log_metrics(
                {
                    "root_mean_squared_error": mean_rmse
                }
            )
        print(rmses)
        return rmses


if __name__ == '__main__':
    data_pd = data_processing.read_data()
    k = 12
    epochs = 43

    submit = False

    svt_init_svd_hybrid = SVT_INIT_SVD_ALS_SGD(k_singular_values=k, epochs=epochs, verbal=True)

    if submit:
        data = DatasetWrapper(data_pd)
        svt_init_svd_hybrid.fit(data)
        # svt_init_svd_hybrid.predict_for_submission(f'svt_init_svd_als_sgd_k{k}_{epochs}')
        svt_init_svd_hybrid.save(f'models/submit/svt_advanced_{k}.pickle')  # export model
        # instead of fitting for all data, fit only for cross-validation fold
        # split on panda dataframe and give it a fold?
    else:
        rmses = svt_init_svd_hybrid.cross_validate(data_pd)
        print("RMSES of ", svt_init_svd_hybrid.method_name, "\n", rmses, "\n")
