from collections import defaultdict

import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler, PowerTransformer, MinMaxScaler
from src.models.svd import SVD
from utils import data_processing, dataset
import torch
import torch.nn as nn
import torch.optim as optim
import math
from scipy.sparse import csr_matrix
from src.models.adapted_svd_sgd import ADAPTED_SVD_SGD
from utils.dataset import DatasetWrapper
from .algobase import AlgoBase
from models.svd_sgd import SVD_SGD

eps = 1e-6


class KNNFancyFeatures(AlgoBase):
    def __init__(self, epochs=75, k=12, user_based=True, metric='cosine', algorithm='auto', n_neighbors=2, track_to_comet=False,
                 method_name=None,
                 api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)

        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(metric=metric, algorithm=algorithm, n_neighbors=n_neighbors, n_jobs=-1)
        self.datawrapper = None
        self.nearest_neighbors = None

        self.k = k
        self.user_based = user_based
        self.transformer = MinMaxScaler(feature_range=(1,5))

        self.svd = ADAPTED_SVD_SGD(k_singular_values=k, epochs=epochs, use_prestored=True, store=False)

        print("using power transformer")

    def compute_neighbors(self):
            embeddings = self.embeddings  # shape (10000, 1000)
            _, indices = self.knn.kneighbors(embeddings, n_neighbors=self.n_neighbors)  # shape (10000, k)




            ratings = self.datawrapper.ratings[indices]

            nearest_neighbors_normalized = np.mean(ratings, axis=1)
            np.copyto(nearest_neighbors_normalized, self.datawrapper.data_matrix,
                      where=self.datawrapper.mask.astype(bool))

            self.nearest_neighbors = self.transformer.inverse_transform(nearest_neighbors_normalized)

            # user_biases_matrix = np.reshape(self.svd.bu, (self.number_of_users, 1))
            # movie_biases_matrix = np.reshape(self.svd.bi, (1, self.number_of_movies))
            # self.nearest_neighbors = self.nearest_neighbors + user_biases_matrix + movie_biases_matrix + self.svd.mu



    def fit(self, datawrapper):
        self.datawrapper = datawrapper
        users, movies = self.datawrapper.users, self.datawrapper.movies
        self.svd.fit(datawrapper)
        self.embeddings = self.get_features(users, movies)


        self.knn.fit(self.embeddings)

        self.compute_neighbors()

    def predict(self, users, movies):
        self.compute_neighbors()

        if self.user_based:
            predictions = self.nearest_neighbors[tuple([users, movies])]
        else:
            predictions = self.nearest_neighbors[tuple([movies, users])]

        nan_mask = np.isnan(predictions)
        counter_nan = np.count_nonzero(nan_mask)

        np.copyto(predictions, self.data_wrapper.movie_means[movies], where=nan_mask)  # impute nan values

        if (counter_nan > 0):
            print("** had to impute ", counter_nan, " movies **")

        return predictions

    def get_features(self, users, movies):
        user_features = self.svd.pu[users]
        movie_features = self.svd.qi[movies]
        user_means = np.reshape(self.datawrapper.user_means[users], (-1, 1))
        movie_means = np.reshape(self.datawrapper.movie_means[movies], (-1, 1))
        user_bias = np.reshape(self.svd.bu[users], (-1, 1))
        movie_bias = np.reshape(self.svd.bi[movies], (-1, 1))

        user_var = np.reshape(self.datawrapper.user_variance[users], (-1, 1))
        movie_var = np.reshape(self.datawrapper.movie_variance[movies], (-1, 1))

        num_movies_watched = np.reshape(self.datawrapper.num_movies_watched[users], (-1, 1))
        times_watched = np.reshape(self.datawrapper.times_watched[movies], (-1, 1))

        X = np.concatenate((user_features, movie_features, user_means, movie_means, user_bias, movie_bias, user_var,
                            movie_var, num_movies_watched, times_watched), axis=1)
        return X


class KNNImprovedSVDEmbeddings(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """

    def __init__(self, epochs=20, k=12, user_based=True, metric='cosine', algorithm='auto', n_neighbors=5, track_to_comet=False,
                 method_name=None,
                 api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)

        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(metric=metric, algorithm=algorithm, n_neighbors=n_neighbors, n_jobs=-1)
        self.data_wrapper = None
        self.nearest_neighbors = None

        self.k = k
        self.user_based = user_based
        self.transformer = MinMaxScaler(feature_range=(1,5))

        self.svd = ADAPTED_SVD_SGD(k_singular_values=k, epochs=epochs)

        print("using power transformer")

    def compute_neighbors(self):

        if self.user_based:
            user_vectors = self.user_embeddings  # shape (10000, 1000)
            _, indices = self.knn.kneighbors(user_vectors, n_neighbors=self.n_neighbors)  # shape (10000, k)
            # movies = self.movie_embeddings  # shape (10000, k, 1000)
            # movies[movies == 0] = np.nan
            # user_mean_values = np.nanmean(movies, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)
            low_rank = self.user_embeddings.dot(self.movie_embeddings.T)
            np.copyto(low_rank, self.data_wrapper.data_matrix, where=self.data_wrapper.mask.astype(bool))

            ratings = low_rank[indices]

            nearest_neighbors_normalized = np.mean(ratings, axis=1)
            np.copyto(nearest_neighbors_normalized, self.data_wrapper.data_matrix,
                      where=self.data_wrapper.mask.astype(bool))

            self.nearest_neighbors = self.transformer.inverse_transform(nearest_neighbors_normalized)

            user_biases_matrix = np.reshape(self.svd.bu, (self.number_of_users, 1))
            movie_biases_matrix = np.reshape(self.svd.bi, (1, self.number_of_movies))
            self.nearest_neighbors = self.nearest_neighbors + user_biases_matrix + movie_biases_matrix + self.svd.mu


        else:
            movie_vectors = self.movie_embeddings  # shape (10000, 1000)
            _, indices = self.knn.kneighbors(movie_vectors, n_neighbors=self.n_neighbors)  # shape (10000, k)
            # movies = self.movie_embeddings  # shape (10000, k, 1000)
            # movies[movies == 0] = np.nan
            # user_mean_values = np.nanmean(movies, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)
            ratings = np.tensordot(self.movie_embeddings[indices], np.transpose(self.user_embeddings), axes=1)
            self.nearest_neighbors = np.mean(ratings, axis=1)

    def fit(self, data_wrapper):
        self.data_wrapper = data_wrapper
        data = data_wrapper.data_matrix
        data = self.transformer.fit_transform(data)

        self.svd.fit(data_wrapper)
        #self.user_embeddings, self.movie_embeddings = SVD.get_embeddings(self.k, data)
        self.user_embeddings = self.svd.pu
        self.movie_embeddings = self.svd.qi
        if self.user_based:
            self.knn.fit(self.user_embeddings)

        else:
            self.knn.fit(self.movie_embeddings)

        self.compute_neighbors()

    def predict(self, users, movies):
        self.compute_neighbors()

        if self.user_based:
            predictions = self.nearest_neighbors[tuple([users, movies])]
        else:
            predictions = self.nearest_neighbors[tuple([movies, users])]

        nan_mask = np.isnan(predictions)
        counter_nan = np.count_nonzero(nan_mask)

        np.copyto(predictions, self.data_wrapper.movie_means[movies], where=nan_mask)  # impute nan values

        if (counter_nan > 0):
            print("** had to impute ", counter_nan, " movies **")

        return predictions



# [1.0008103659078311, 0.9991618105243478, 1.000329012060183, 0.9976463697503822, 0.9961922447259809]
class KNNSVD_Biases(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """

    def __init__(self, k=2, svd_epochs=10, user_based=True, metric='cosine', algorithm='auto', n_neighbors=5, track_to_comet=False,
                 method_name=None,
                 api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)

        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(metric=metric, algorithm=algorithm, n_neighbors=n_neighbors, n_jobs=-1)
        self.data_wrapper = None
        self.nearest_neighbors = None

        self.k = k
        self.user_based = user_based

        self.svd = SVD_SGD(k_singular_values=k, epochs=10)

        # self.transformer = RobustScaler(with_centering=False, with_scaling=False, unit_variance=False) # 1.0277 mean rsme
        self.transformer = PowerTransformer()
        print("using power transformer, k=", k, " epochs=", svd_epochs, "n = ", n_neighbors, " neighbors")

    def compute_neighbors(self):

        if self.user_based:
            user_vectors = self.reconstructed_matrix  # shape (10000, 1000)
            _, indices = self.knn.kneighbors(user_vectors, n_neighbors=self.n_neighbors)  # shape (10000, k)
            # movies = self.movie_embeddings  # shape (10000, k, 1000)
            # movies[movies == 0] = np.nan
            # user_mean_values = np.nanmean(movies, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)
            low_rank = self.reconstructed_matrix
           # np.copyto(low_rank, self.data_wrapper.data_matrix, where=self.data_wrapper.mask.astype(bool))

            ratings = low_rank[indices]

            nearest_neighbors_normalized = np.mean(ratings, axis=1)
            np.copyto(nearest_neighbors_normalized, self.data_wrapper.data_matrix,
                      where=self.data_wrapper.mask.astype(bool))

            #self.nearest_neighbors = self.transformer.inverse_transform(nearest_neighbors_normalized)
            self.nearest_neighbors = nearest_neighbors_normalized

        else:
            movie_vectors = self.movie_embeddings  # shape (10000, 1000)
            _, indices = self.knn.kneighbors(movie_vectors, n_neighbors=self.n_neighbors)  # shape (10000, k)
            # movies = self.movie_embeddings  # shape (10000, k, 1000)
            # movies[movies == 0] = np.nan
            # user_mean_values = np.nanmean(movies, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)
            ratings = np.tensordot(self.movie_embeddings[indices], np.transpose(self.user_embeddings), axes=1)
            self.nearest_neighbors = np.mean(ratings, axis=1)

    def fit(self, data_wrapper):
        self.data_wrapper = data_wrapper
        # data = data_wrapper.data_matrix
        # data = self.transformer.fit_transform(data)

        self.svd.fit(data_wrapper)

        self.reconstructed_matrix = self.svd.reconstructed_matrix

        # self.reconstructed_matrix = self.transformer.fit_transform(self.reconstructed_matrix)

        if self.user_based:
            self.knn.fit(self.reconstructed_matrix)

        else:
            self.knn.fit(self.reconstructed_matrix.T)

        self.compute_neighbors()

    def predict(self, users, movies):
        self.compute_neighbors()

        if self.user_based:
            predictions = self.nearest_neighbors[tuple([users, movies])]
        else:
            predictions = self.nearest_neighbors[tuple([movies, users])]

        nan_mask = np.isnan(predictions)
        counter_nan = np.count_nonzero(nan_mask)

        np.copyto(predictions, self.data_wrapper.movie_means[movies], where=nan_mask)  # impute nan values

        if (counter_nan > 0):
            print("** had to impute ", counter_nan, " movies **")

        return predictions


class KNNSVD_Embeddings(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """

    def __init__(self, k=5, user_based=True, metric='cosine', algorithm='auto', n_neighbors=5, track_to_comet=False,
                 method_name=None,
                 api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)

        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(metric=metric, algorithm=algorithm, n_neighbors=n_neighbors, n_jobs=-1)
        self.data_wrapper = None
        self.nearest_neighbors = None

        self.k = k
        self.user_based = user_based
        self.transformer = MinMaxScaler(feature_range=(1,5))  # 1.0277 mean rsme
        # self.transformer = RobustScaler(with_centering=False, with_scaling=False, unit_variance=False) # 1.0277 mean rsme
        # self.transformer = PowerTransformer()
        print("using power transformer")

    def compute_neighbors(self):

        if self.user_based:
            user_vectors = self.user_embeddings  # shape (10000, 1000)
            _, indices = self.knn.kneighbors(user_vectors, n_neighbors=self.n_neighbors)  # shape (10000, k)
            # movies = self.movie_embeddings  # shape (10000, k, 1000)
            # movies[movies == 0] = np.nan
            # user_mean_values = np.nanmean(movies, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)
            low_rank = self.user_embeddings.dot(self.movie_embeddings.T)
            np.copyto(low_rank, self.data_wrapper.data_matrix, where=self.data_wrapper.mask.astype(bool))

            ratings = low_rank[indices]

            nearest_neighbors_normalized = np.mean(ratings, axis=1)
            np.copyto(nearest_neighbors_normalized, self.data_wrapper.data_matrix,
                      where=self.data_wrapper.mask.astype(bool))

            self.nearest_neighbors = self.transformer.inverse_transform(nearest_neighbors_normalized)


        else:
            movie_vectors = self.movie_embeddings  # shape (10000, 1000)
            _, indices = self.knn.kneighbors(movie_vectors, n_neighbors=self.n_neighbors)  # shape (10000, k)
            # movies = self.movie_embeddings  # shape (10000, k, 1000)
            # movies[movies == 0] = np.nan
            # user_mean_values = np.nanmean(movies, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)
            ratings = np.tensordot(self.movie_embeddings[indices], np.transpose(self.user_embeddings), axes=1)
            self.nearest_neighbors = np.mean(ratings, axis=1)

    def fit(self, data_wrapper):
        self.data_wrapper = data_wrapper
        data = data_wrapper.data_matrix
        data = self.transformer.fit_transform(data)

        self.user_embeddings, self.movie_embeddings = SVD.get_embeddings(self.k, data)

        if self.user_based:
            self.knn.fit(self.user_embeddings)

        else:
            self.knn.fit(self.movie_embeddings)

        self.compute_neighbors()

    def predict(self, users, movies):
        self.compute_neighbors()

        if self.user_based:
            predictions = self.nearest_neighbors[tuple([users, movies])]
        else:
            predictions = self.nearest_neighbors[tuple([movies, users])]

        nan_mask = np.isnan(predictions)
        counter_nan = np.count_nonzero(nan_mask)

        np.copyto(predictions, self.data_wrapper.movie_means[movies], where=nan_mask)  # impute nan values

        if (counter_nan > 0):
            print("** had to impute ", counter_nan, " movies **")

        return predictions


class KNNUserMovie(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """

    def __init__(self, user_metric='cosine', user_algorithm='auto', movie_metric='cosine', movie_algorithm='auto',
                 n_user_neighbors=5, n_movie_neighbors=5, track_to_comet=False, method_name=None,
                 api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)

        self.n_user_neighbors = n_user_neighbors
        self.n_movie_neighbors = n_movie_neighbors
        self.movie_knn = NearestNeighbors(metric=movie_metric, algorithm=movie_algorithm, n_neighbors=n_movie_neighbors,
                                          n_jobs=-1)
        self.user_knn = NearestNeighbors(metric=user_metric, algorithm=user_algorithm, n_neighbors=n_user_neighbors,
                                         n_jobs=-1)
        self.knn_euclidean = KNN(metric='euclidean', algorithm='auto', n_neighbors=500)

        self.data_wrapper = None
        self.movie_nearest_neighbors = None
        self.user_nearest_neighbors = None

        self.user_weight = 0

    def compute_neighbors(self):
        # Compute nearest neighbors of users
        user_vectors = self.data_wrapper.movie_per_user_encodings  # shape (10000, 1000)
        _, indices = self.user_knn.kneighbors(user_vectors, n_neighbors=self.n_user_neighbors)  # shape (10000, k)
        movies = self.data_wrapper.movie_per_user_encodings[indices]  # shape (10000, k, 1000)
        movies[movies == 0] = np.nan
        user_mean_values = np.nanmean(movies, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)

        self.user_nearest_neighbors = user_mean_values

        # Compute nearest neighbors of movies
        movie_vectors = self.data_wrapper.user_per_movie_encodings  # shape (10000, 1000)
        _, indices = self.movie_knn.kneighbors(movie_vectors, n_neighbors=self.n_movie_neighbors)  # shape (10000, k)
        users = self.data_wrapper.user_per_movie_encodings[indices]  # shape (10000, k, 1000)
        users[users == 0] = np.nan
        movie_mean_values = np.nanmean(users, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)

        self.movie_nearest_neighbors = movie_mean_values

    def fit(self, data_wrapper):
        self.data_wrapper = data_wrapper
        user_matrix = data_wrapper.movies_per_user_representation()
        movie_matrix = data_wrapper.users_per_movie_representation()

        self.user_knn.fit(user_matrix)
        self.knn_euclidean.fit(data_wrapper)
        self.movie_knn.fit(movie_matrix)
        self.compute_neighbors()

    def predict(self, users, movies):

        user_predictions = self.user_nearest_neighbors[tuple([users, movies])]
        movie_predictions = self.movie_nearest_neighbors[tuple([movies, users])]
        euclidean_user_predictions = self.knn_euclidean.nearest_neighbors[tuple([users, movies])]

        mask_either_is_nan = np.logical_or(np.isnan(user_predictions), np.isnan(movie_predictions))
        predictions = movie_predictions
        # predictions[np.isnan(user_predictions)] = movie_predictions[np.isnan(user_predictions)]

        # predictions[~mask_either_is_nan] = self.user_weight*user_predictions[~mask_either_is_nan] + (1-self.user_weight)*movie_predictions[~mask_either_is_nan]
        # predictions[predictions < 3.5] = euclidean_user_predictions[predictions < 3.5]

        counter_nan = 0

        for i, pred in enumerate(predictions):
            r = self.data_wrapper.rating_available(users[i], movies[i])
            if (r > 0):
                print("we have this rating")
                predictions[i] = r
            if np.isnan(pred):
                counter_nan += 1

                predictions[i] = self.data_wrapper.user_means[users[i]]

        print("had to impute ", counter_nan, " movies")
        return predictions


class KNNBag(AlgoBase):
    def __init__(self, knns=None, track_to_comet=False, method_name=None, api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)
        # self.knn_cosine = None
        # self.knn_manhattan = None
        # self.knn_euclidean = None

        # if knns is None:
        #     self.knns = self.get_standard_knns()
        # else:
        #     self.knns = knns  # list of multiple KNNs to bag together

        self.knn_cosine = None
        self.knn_manhattan = None
        self.knn_euclidean = None
        self.knn_manhattan_large = None

        self.get_standard_knns()

        self.data_wrapper = None

    def fit(self, data_wrapper):
        self.data_wrapper = data_wrapper
        for knn in self.knns:
            knn.fit(data_wrapper)  # fit all the knns

    def predict(self, users, movies):

        predictions_euclidean = self.knn_euclidean.nearest_neighbors[tuple([users, movies])]
        predictions_cosine = self.knn_cosine.nearest_neighbors[tuple([users, movies])]
        # predictions_manhattan = self.knn_manhattan.nearest_neighbors[tuple([users, movies])]
        # predictions_manhattan_large = self.knn_manhattan_large.nearest_neighbors[tuple([users, movies])]

        predictions = predictions_euclidean

        predictions[predictions >= 3.5] = predictions_cosine[predictions >= 3.5]

        # fill predictions that are STILL nan with the mean
        counter_nan = 0
        for i, pred in enumerate(predictions):
            if np.isnan(pred):
                predictions[i] = self.knns[0].data_wrapper.movie_means[movies[i]]
                counter_nan += 1
        print("In the end, we had to impute ", counter_nan, " of the predictions with the movie-mean")

        predictions = predictions + self.data_wrapper.movie_bias[movies] + self.data_wrapper.user_bias[
            users]  # include biases

        return predictions

    def get_standard_knns(self):
        self.knn_cosine = KNN(metric='cosine', algorithm='auto', n_neighbors=500)

        # self.knn_manhattan_large = KNN(metric='manhattan', algorithm='auto', n_neighbors=100)
        # self.knn_manhattan = KNN(metric='manhattan', algorithm='auto', n_neighbors=2)
        self.knn_euclidean = KNN(metric='euclidean', algorithm='auto', n_neighbors=500)
        self.knns = [self.knn_cosine, self.knn_euclidean]


class KNN(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """

    def __init__(self, metric='cosine', algorithm='auto', n_neighbors=5, track_to_comet=False, method_name=None,
                 api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)

        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(metric=metric, algorithm=algorithm, n_neighbors=n_neighbors, n_jobs=-1)
        self.data_wrapper = None
        self.nearest_neighbors = None

        self.user_based = True
        # self.item_based = False

    def compute_neighbors(self):

        if self.user_based:
            user_vectors = self.data_wrapper.movie_per_user_encodings  # shape (10000, 1000)
            _, indices = self.knn.kneighbors(user_vectors, n_neighbors=self.n_neighbors)  # shape (10000, k)
            movies = self.data_wrapper.movie_per_user_encodings[indices]  # shape (10000, k, 1000)
            movies[movies == 0] = np.nan
            user_mean_values = np.nanmean(movies, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)

            self.nearest_neighbors = user_mean_values

        else:
            movie_vectors = self.data_wrapper.user_per_movie_encodings  # shape (10000, 1000)
            _, indices = self.knn.kneighbors(movie_vectors,
                                             n_neighbors=self.n_neighbors)  # shape (10000, k)
            users = self.data_wrapper.user_per_movie_encodings[indices]  # shape (10000, k, 1000)
            users[users == 0] = np.nan
            movie_mean_values = np.nanmean(users, axis=1)  # shape (10000, 1000) (average over the 5 neighbors)

            self.nearest_neighbors = movie_mean_values

    def fit(self, data_wrapper):
        self.data_wrapper = data_wrapper
        matrix = data_wrapper.movies_per_user_representation()
        self.knn.fit(matrix)

        self.compute_neighbors()

    def predict(self, users, movies):
        return self.predict_movie_mean(users, movies)

    def predict_movie_mean(self, users, movies):
        self.compute_neighbors()

        if (self.user_based):
            predictions = self.nearest_neighbors[tuple([users, movies])]
        else:
            predictions = self.nearest_neighbors[tuple([movies, users])]

        counter_nan = 0
        for i, pred in enumerate(predictions):
            r = self.data_wrapper.rating_available(users[i], movies[i])
            if (r > 0):
                print("we have this rating")
                predictions[i] = r
            if np.isnan(pred):
                counter_nan += 1

                predictions[i] = self.data_wrapper.movie_means[movies[i]]
        print("In the end, we had to impute ", counter_nan, " of the predictions with the movie-mean")

        return predictions
