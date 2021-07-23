import os

import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from src.utils import data_processing
from src.models.algobase import AlgoBase
from src.models.svd import SVD
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import pandas as pd
from utils import dataset, data_analysis

EPSILON = 1e-5

STORE_DATA = True


class ADAPTED_SVD_SGD(AlgoBase):
    """
    Running SGD on SVD initialized embeddings.
    By Surprise documentation:
    https://surprise.readthedocs.io/en/stable/matrix_factorization.html#matrix-factorization-based-algorithms
    """

    def __init__(self, error='standard', k_singular_values=17, epochs=100, learning_rate=0.001, regularization=0.05,
                 use_prestored=False, store=True, verbal=False,
                 track_to_comet=False):
        AlgoBase.__init__(self, track_to_comet)

        self.k = k_singular_values  # number of singular values to use
        self.epochs = epochs
        # self.learning_rate = learning_rate
        self.learning_rate = {
            1: 0.001,
            2: 0.001,
            3: 0.001,
            4: 0.001,
            5: 0.001
        }
        self.regularization = regularization
        self.verbal = verbal
        self.use_prestored = use_prestored
        self.store = store

        self.matrix = np.zeros((self.number_of_users, self.number_of_movies))
        self.reconstructed_matrix = np.zeros((self.number_of_users, self.number_of_movies))

        self.pu = np.empty((self.number_of_users, self.k))  # user embedding
        self.qi = np.empty((self.number_of_movies, self.k))  # item embedding

        self.bu = np.zeros(self.number_of_users)  # user bias
        self.bi = np.zeros(self.number_of_movies)  # item bias
        self.mu = 0

        if error == 'fancy':
            self.error = self.fancy_error
        elif error == 'weighted':
            self.error = self.weighted_error
        else:
            self.error = self.standard_error

    def _update_reconstructed_matrix(self):
        dot_product = self.pu.dot(self.qi.T)
        user_biases_matrix = np.reshape(self.bu, (self.number_of_users, 1))
        movie_biases_matrix = np.reshape(self.bi, (1, self.number_of_movies))
        self.reconstructed_matrix = dot_product + user_biases_matrix + movie_biases_matrix + self.mu

    # Probabilities:
    # 1 0.036966673237311294
    # 2 0.08426851732271155
    # 3 0.23308257261128745
    # 4 0.2758821090409804
    # 5 0.36980012778770927
    def weighted_error(self, actual, output):
        weights = {
            1: 0.0370037,
            2: 0.08430843,
            3: 0.2330233,
            4: 0.27582758,
            5: 0.36983698
        }
        w = (1 - weights[actual]) * 1.38  # make sure the sum of all the weights in the dataset stays the same
        error = w * (actual - output)

        return error, w

    def standard_error(self, actual, output):
        return (actual - output), 1

    def get_pearson_similarities(self, datawrapper):
        directory_path = data_processing.get_project_directory()
        file = f"{directory_path}/data/precompute/pearson_matrix.csv"
        if not os.path.exists(file):
            self.sim = np.zeros((datawrapper.num_users, datawrapper.num_users))
            with tqdm(total=datawrapper.num_users * datawrapper.num_users, disable=False,
                      desc="computing pearson matrix") as pbar:

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
        w = weights[actual]
        error = w * (actual - output)

        return error, w

    def read_prestored_matrices(self):
        directory_path = data_processing.get_project_directory()
        self.pu = pd.read_csv(f"{directory_path}/data/precomputed_svd_e{self.epochs}_k{self.k}/pu.csv")
        self.qi = pd.read_csv(f"{directory_path}/data/precomputed_svd_e{self.epochs}_k{self.k}/qi.csv")
        self.bu = pd.read_csv(f"{directory_path}/data/precomputed_svd_e{self.epochs}_k{self.k}/bu.csv")
        self.bi = pd.read_csv(f"{directory_path}/data/precomputed_svd_e{self.epochs}_k{self.k}/bi.csv")

    def fit(self, datawrapper, valid_users=None, valid_movies=None, valid_ground_truth=None):
        if self.use_prestored:
            self.read_prestored_matrices()
            return

        users, movies, ground_truth = data_wrapper.users, data_wrapper.movies, data_wrapper.ratings
        self.matrix, _ = data_processing.get_data_mask(data_wrapper, impute='fancy', val_users=valid_users,
                                                       val_movies=valid_movies)
        similarity_terms = self.get_similarity_terms(datawrapper)

        # normalized_matrix = data_processing.normalize_by_variance(self.matrix)
        # self.pu, self.qi = SVD.get_embeddings(self.k, normalized_matrix)
        self.pu, self.qi = SVD.get_embeddings(self.k, self.matrix)

        run_validation = valid_users is not None and valid_movies is not None and valid_ground_truth is not None
        indices = np.arange(len(users))
        # global_mean = np.mean(self.matrix[np.nonzero(self.matrix)])

        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f'./logs/SGD_multiLr_{self.epochs}e_{self.k}k_{self.error.__func__.__name__}_{time_string}'
        writer = SummaryWriter(log_dir)

        with tqdm(total=self.epochs * len(users), disable=not self.verbal) as pbar:
            for epoch in range(self.epochs):
                np.random.shuffle(indices)

                weights = 0
                for user, movie in zip(users[indices], movies[indices]):
                    similarity_term = similarity_terms[user, movie]

                    prediction = self.bu[user] + self.bi[movie] + similarity_term + np.dot(self.pu[user], self.qi[movie])
                    error, w = self.error(self.matrix[user, movie], prediction)
                    # bias_change = self.learning_rate * (error - self.regularization * (self.bu[user] + self.bi[movie] - global_mean))

                    lr = self.learning_rate[self.matrix[user, movie]]
                    self.bu[user] += lr * (error - self.regularization * self.bu[user])
                    self.bi[movie] += lr * (error - self.regularization * self.bi[movie])

                    self.pu[user] += lr * (
                            error * self.qi[movie] - self.regularization * self.pu[user])
                    self.qi[movie] += lr * (
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

        if self.store:
            directory_path = data_processing.get_project_directory()

            bu = pd.DataFrame({'bu': np.array(self.bu)})
            bi = pd.DataFrame({'bi': np.array(self.bi)})
            qi = pd.DataFrame(np.array(self.qi))
            pu = pd.DataFrame(np.array(self.pu))

            # export to file:
            storing_directory = f"{directory_path}/data/precomputed_svd_e{self.epochs}_k{self.k}"
            if not os.path.exists(storing_directory):
                os.mkdir(storing_directory)

            # archive_name is required to create working zip on my computer
            bu.to_csv(f"{storing_directory}/bu.csv",
                      index=True,
                      )
            # export to file:
            # archive_name is required to create working zip on my computer
            bi.to_csv(f"{storing_directory}/bi.csv",
                      index=True,
                      )
            # export to file:
            # archive_name is required to create working zip on my computer
            pu.to_csv(f"{storing_directory}/pu.csv",
                      index=True,
                      )
            # export to file:
            # archive_name is required to create working zip on my computer
            qi.to_csv(f"{storing_directory}/qi.csv",
                      index=True,
                      )

    def predict(self, users, movies):
        predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)
        predictions[predictions > 5] = 5
        predictions[predictions < 1] = 1
        return predictions


if __name__ == '__main__':
    data_pd = data_processing.read_data()
    k = 12
    epochs = 75
    error = 'fancy'

    submit = False
    train_val = True
    store_matrices = False
    use_precomputed = False

    sgd = ADAPTED_SVD_SGD(k_singular_values=k, error=error, epochs=epochs, verbal=True, use_prestored=use_precomputed, store=store_matrices)

    if submit:
        data_wrapper = dataset.DatasetWrapper(data_pd)
        sgd.fit(data_wrapper, valid_movies=None, valid_users=None)
        sgd.predict_for_submission(f'svd_sgd_norm_k{k}_{epochs}')

    elif store_matrices:
        data_wrapper = dataset.DatasetWrapper(data_pd)
        sgd.fit(data_wrapper, valid_movies=None, valid_users=None)

        print("done precomputing stuff")

    else:

        train_pd, test_pd = train_test_split(data_pd, train_size=0.9, random_state=42)
        train_pd, val_pd = train_test_split(train_pd, train_size=0.9, random_state=42)
        data_wrapper = dataset.DatasetWrapper(train_pd)
        # users, movies, predictions = data_processing.extract_users_items_predictions(train_pd)

        # rsmes = sgd.cross_validate(data_pd)
        # print("Cross validations core: ", np.mean(rsmes))
        val_users, val_movies, val_predictions = data_processing.extract_users_items_predictions(val_pd)
        test_users, test_movies, test_predictions = data_processing.extract_users_items_predictions(test_pd)
        sgd.fit(data_wrapper, valid_movies=val_movies, valid_users=val_users, valid_ground_truth=val_predictions)

        pred = sgd.predict(test_users, test_movies)

        pa = data_analysis.PredictionAnalyser(test_movies=test_movies, test_predictions=test_predictions,
                                              test_users=test_users, output_predictions=pred)

        pa.analyze_prediction()

        data_processing.create_validation_file(test_users, test_movies, pred, test_predictions)
        da = data_analysis.DataAnalyzer(data_pd)
        da.create_validation_histograms(pred, val_predictions)
        da.create_validation_scatterplot(pred, val_predictions)
