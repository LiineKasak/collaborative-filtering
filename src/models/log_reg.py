import argparse

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from models.svd_sgd import SVD_SGD
from models.algobase import AlgoBase
import numpy as np


class LogisticRegression(AlgoBase):
    """
        Compute User- and Movie Embeddings, as well as biases through optimized SVD. Then use these as input to perform
        Gradient Boosting Classification
    """

    def __init__(self, params: argparse.Namespace):
        AlgoBase.__init__(self)
        self.datawrapper = None
        self.user_embeddings, self.movie_embeddings = None, None

        self.clf = HistGradientBoostingClassifier(scoring='neg_root_mean_squared_error',
                                                  l2_regularization=1e-3,
                                                  verbose=params.verbal,
                                                  max_iter=150)

        self.sgd = SVD_SGD(params)

    @staticmethod
    def default_params():
        return argparse.Namespace(k_singular_values=12, epochs=75, learning_rate=0.001, regularization=0.05,
                                  verbal=True, enable_bias=True)

    def predict(self, users, movies):
        X = self.get_features(users, movies)
        probs = self.clf.predict_proba(X)
        ratings = np.array([1, 2, 3, 4, 5])

        pred = probs * ratings

        return np.sum(pred, axis=1)

    def fit(self, datawrapper, val_users=None, val_movies=None):
        self.datawrapper = datawrapper
        users, movies, ratings = datawrapper.users, datawrapper.movies, datawrapper.ratings

        # compute svd
        self.sgd.fit(datawrapper)

        X = self.get_features(users, movies)
        y = ratings

        self.clf.fit(X, y)

    def get_features(self, users, movies):
        """
        Extract features given a set of users and movies. Only call after training the SvdInputClassifier
        (requires trained svd)
        """
        user_features = self.sgd.pu[users]
        movie_features = self.sgd.qi[movies]
        user_means = np.reshape(self.datawrapper.user_means[users], (-1, 1))
        movie_means = np.reshape(self.datawrapper.movie_means[movies], (-1, 1))
        user_bias = np.reshape(self.sgd.bu[users], (-1, 1))
        movie_bias = np.reshape(self.sgd.bi[movies], (-1, 1))

        user_var = np.reshape(self.datawrapper.user_variance[users], (-1, 1))
        movie_var = np.reshape(self.datawrapper.movie_variance[movies], (-1, 1))

        num_movies_watched = np.reshape(self.datawrapper.num_movies_watched[users], (-1, 1))
        times_watched = np.reshape(self.datawrapper.times_watched[movies], (-1, 1))

        X = np.concatenate((user_features, movie_features, user_means, movie_means, user_bias, movie_bias, user_var,
                            movie_var, num_movies_watched, times_watched), axis=1)
        return X
