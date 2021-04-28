import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
from constants import *
from collections import Counter
import os


class DatasetUtil:
    def __init__(self):
        self._data_pd = pd.read_csv('data_train.csv')

        train_size = 0.9
        self._train_pd, self._test_pd = train_test_split(self._data_pd, train_size=train_size, random_state=42)
        # also create full matrix of observed values
        self.data = np.full((NR_USERS, NR_MOVIES), np.mean(self._train_pd.Prediction.values))
        self.mask = np.zeros((NR_USERS, NR_MOVIES))  # 0 -> unobserved value, 1->observed value
        self._init_data_mask()

    def _init_data_mask(self):
        train_users, train_movies, train_predictions = self._extract_users_items_predictions(self._train_pd)

        for user, movie, pred in zip(train_users, train_movies, train_predictions):
            self.data[user - 1][movie - 1] = pred
            self.mask[user - 1][movie - 1] = 1

    def rmse_scores_from_matrix(self, predictions):
        # test our predictions with the true values
        train_users, train_movies, train_truth = self._extract_users_items_predictions(self._train_pd)
        test_users, test_movies, test_truth = self._extract_users_items_predictions(self._test_pd)

        predictions_train = self._extract_prediction_from_full_matrix(predictions, train_users, train_movies)
        predictions_test = self._extract_prediction_from_full_matrix(predictions, test_users, test_movies)
        return self._get_score(predictions_train, train_truth), self._get_score(predictions_test, test_truth)

    def rmse_scores(self, predictions_train, predictions_test):
        train_users, train_movies, train_truth = self._extract_users_items_predictions(self._train_pd)
        test_users, test_movies, test_truth = self._extract_users_items_predictions(self._test_pd)
        return self._get_score(predictions_train, train_truth), self._get_score(predictions_test, test_truth)

    @staticmethod
    def _get_score(predictions, ground_truth):
        return math.sqrt(mean_squared_error(predictions, ground_truth))

    @staticmethod
    def _extract_prediction_from_full_matrix(reconstructed_matrix, users, movies):
        # returns predictions for the users-movies combinations specified based on a full m \times n matrix
        assert (len(users) == len(movies)), "users-movies combinations specified should have equal length"
        predictions = np.zeros(len(users))

        for i, (user, movie) in enumerate(zip(users, movies)):
            predictions[i] = reconstructed_matrix[user][movie]
        return predictions

    @staticmethod
    def _extract_users_items_predictions(data_pd):
        users, movies = \
            [np.squeeze(arr) for arr in
             np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
        predictions = data_pd.Prediction.values
        return users, movies, predictions

    def save_predictions_from_matrix(self, matrix, model_name):
        users, movies, truth = self._extract_users_items_predictions(self._data_pd)
        predictions = self._extract_prediction_from_full_matrix(matrix, users, movies)
        self.save_predictions(predictions, model_name)

    def save_predictions(self, predictions, model_name):
        predictions = list(map(round, predictions))
        users, movies, truth = self._extract_users_items_predictions(self._data_pd)
        print(f'Occurences in ground truth: {Counter(truth)}')

        ids = [f'r{user}_c{movie}' for user, movie in zip(users, movies)]
        # TODO: do we have to round for submission?
        print(f'Occurences in predictions: {Counter(predictions)}')

        data = {'Id': ids, 'Prediction': predictions}
        dataframe = pd.DataFrame(data, columns=['Id', 'Prediction'])

        directory = 'submissions'
        if not os.path.exists(directory):
            os.makedirs(directory)
        dataframe.to_csv(f'{directory}{os.sep}{model_name}.csv', index=False)