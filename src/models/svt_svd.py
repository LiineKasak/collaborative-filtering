import os

import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from src.utils import data_processing, dataset
from src.models.algobase import AlgoBase
from src.models.svd import SVD
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import pandas as pd
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

        if submit:
            with open(self.directory_path + '/data/bu_bi_no_mean.npy', 'rb') as f:
                self.bu = np.load(f, allow_pickle=True)
                self.bi = np.load(f, allow_pickle=True)
        else: 
            with open(self.directory_path + '/data/bu_bi_no_mean_trainonly.npy', 'rb') as f:
                self.bu = np.load(f, allow_pickle=True)
                self.bi = np.load(f, allow_pickle=True)

        self.mu = 0


    def get_pearson_similarities(self, datawrapper):
        directory_path = data_processing.get_project_directory()
        file = f"{directory_path}/data/precompute/pearson_matrix.csv"
        if not os.path.exists(file):
            self.sim = np.zeros((datawrapper.num_users, datawrapper.num_users))
            with tqdm(total=datawrapper.num_users * datawrapper.num_users, disable=False, desc="computing pearson matrix") as pbar:

                for u in range(datawrapper.num_users):
                    for v in range(datawrapper.num_users):
                        self.sim[u, v], _ = pearsonr(self.pu[u], self.pu[v])
                        pbar.update()

            df = pd.DataFrame(self.sim)
            df.to_csv(file)

            # df = pd.DataFrame(self.pearson_matrix)
            # directory_path = data_processing.get_project_directory()
            # storing_directory = f"{directory_path}/data/precompute"
            # if not os.path.exists(storing_directory):
            #     os.mkdir(storing_directory)
            # df.to_csv(f"{storing_directory}/pearson_matrix.csv",
            #           index=True,
            #           )
            return self.sim

        else:
            df = pd.read_csv(file)
            self.sim = np.array(df)[:, 1:]


    def get_similarity_terms(self, datawrapper):
        directory_path = data_processing.get_project_directory()
        file = f"{directory_path}/data/precompute/similarity_measure_complete.csv"
        self.get_pearson_similarities(datawrapper)
        similarity_terms_matrix = np.zeros((data_processing.number_of_users, data_processing.number_of_movies))
        # return similarity_terms_matrix
        if not os.path.exists(file):
            similarity_terms_matrix = np.zeros((data_processing.number_of_users, data_processing.number_of_movies))

            with tqdm(total=data_processing.number_of_movies, disable=False, desc="computing similarity terms") as pbar:

                # for user in range(data_processing.number_of_users):
                #     for movie in range(data_processing.number_of_movies):
                #         neighbours = datawrapper.movie_dict[movie]
                #         similarity_term = 0
                #         normalization_term = 0
                #
                #         for v, rvi in neighbours:
                #             bvi = (self.bu[v] + self.bi[movie])
                #             sim_uv = self.sim[user, v]
                #             similarity_term += sim_uv * (rvi - bvi)
                #             normalization_term += sim_uv
                #
                #         similarity_term = similarity_term / normalization_term
                #         similarity_terms_matrix[user, movie] = similarity_term
                #         pbar.update()
                for movie in range(data_processing.number_of_movies):
                    neighbours = datawrapper.movie_dict[movie]
                    users, ratings = np.array(list(zip(*neighbours))[0]), np.array(list(zip(*neighbours))[1])
                    bvi = (self.bu[users] + self.bi[movie])
                    rvi = ratings
                    sim_uv = self.sim[users, users]

                    similarity_term = np.nansum(sim_uv * (rvi - bvi))
                    normalization_term = np.nansum(sim_uv)

                    similarity_terms_matrix[users, movie] = similarity_term / normalization_term
                    pbar.update()

            df = pd.DataFrame(similarity_terms_matrix)
            df.to_csv(file)
            return similarity_terms_matrix

        else:
            df = pd.read_csv(file)

            return np.array(df)[:, 1:]


    def _update_reconstructed_matrix(self):
        dot_product = self.pu.dot(self.qi.T)
        user_biases_matrix = np.reshape(self.bu, (self.number_of_users, 1))
        movie_biases_matrix = np.reshape(self.bi, (1, self.number_of_movies))
        self.reconstructed_matrix = dot_product + user_biases_matrix + movie_biases_matrix + self.mu


    def fancy_error(self, actual, output):
        # weights = {
        #     1: 2,
        #     2: 1.8,
        #     3: 1,
        #     4: 1,
        #     5: 1
        # }
        weights = {
            1: 1.2,
            2: 1,
            3: 1,
            4: 1,
            5: 1.2
        }
        rounded = np.rint(actual)
        if (rounded < 1):
            rounded = 1
        if rounded > 5:
            rounded = 5
        w = weights[rounded]
        error = w * (actual - output)

        return error




    def fit(self,datawrapper, valid_users=None, valid_movies=None, valid_ground_truth=None):
        users, movies, ground_truth = datawrapper.users, datawrapper.movies, datawrapper.ratings
        similarity_terms = self.get_similarity_terms(datawrapper)



        with open(self.directory_path + '/data/svt_Xopt_Yk_sh100k_5000_to_5500.npy', 'rb') as f:
            self.matrix = np.load(f, allow_pickle=True)
            Yk = np.load(f, allow_pickle=True)
        
        self.pu, self.qi = SVD.get_embeddings(self.k, self.matrix)

        run_validation = valid_users is not None and valid_movies is not None and valid_ground_truth is not None
        indices = np.arange(len(users))
        # global_mean = np.mean(self.matrix[np.nonzero(self.matrix)])

        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f'./logs/SGD_SVT_100sh_k12_5500init_NOMEANBL_ep100_{time_string}'
        writer = SummaryWriter(log_dir)

        with tqdm(total=self.epochs * len(users), disable=not self.verbal) as pbar:
            for epoch in range(self.epochs):
                np.random.shuffle(indices)

                for user, movie, _ in zip(users[indices], movies[indices], ground_truth[indices]):


                    similarity_term = similarity_terms[user, movie]

                    bui = self.bu[user] + self.bi[movie]
                    prediction = bui + similarity_term + np.dot(self.pu[user], self.qi[movie])
                    error = self.fancy_error(self.matrix[user, movie], prediction)

                    self.pu[user] += self.learning_rate * (error * self.qi[movie] - self.regularization * self.pu[user])
                    self.qi[movie] += self.learning_rate * (error * self.pu[user] - self.regularization * self.qi[movie])

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

            # df = pd.DataFrame(similarity_terms_matrix)
            # directory_path = data_processing.get_project_directory()
            # storing_directory = f"{directory_path}/data/precompute"
            # if not os.path.exists(storing_directory):
            #     os.mkdir(storing_directory)
            # df.to_csv(f"{storing_directory}/similarity_terms_matrix.csv",
            #           index=True,
            #           )

    def predict(self, users, movies):
        predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)
        predictions[predictions > 5] = 5
        predictions[predictions < 1] = 1
        return predictions


if __name__ == '__main__':
    data_pd = data_processing.read_data()
    k = 12
    epochs = 43

    submit = True

    sgd = SVT_SVD(k_singular_values=k, epochs=epochs, verbal=True, submit=submit)

    if submit:
        # users, movies, predictions = data_processing.extract_users_items_predictions(data_pd)
        datawrapper = dataset.DatasetWrapper(data_pd)
        sgd.fit(datawrapper)
        sgd.predict_for_submission(f'svt_svd_fancy_error_similarity_k{k}_{epochs}')
    else:
        train_pd, test_pd = train_test_split(data_pd, train_size=0.9, random_state=42)
        train_wrapper = dataset.DatasetWrapper(train_pd, impute='fancy')
        users, movies, predictions = data_processing.extract_users_items_predictions(train_pd)
        val_users, val_movies, val_predictions = data_processing.extract_users_items_predictions(test_pd)
        sgd.fit(train_wrapper, val_users, val_movies, val_predictions)
        pred = sgd.predict(val_users, val_movies)
        print(f"rsme: {np.mean(data_processing.get_score(pred, val_predictions))}")
