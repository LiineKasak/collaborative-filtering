from utils import data_processing, dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class PredictionAnalyser:
    def __init__(self, test_users, test_movies, test_predictions, output_predictions, knn=None):
        self.test_users = test_users
        self.test_movies = test_movies
        self.test_predictions = test_predictions
        self.output_predictions = output_predictions
        self.overall_rsme = data_processing.get_score(output_predictions, test_predictions)

        self.knn = knn

        self.unavailable_output_predictions = None
        self.unavailable_test_predictions = None

        self.available_output_predictions = None
        self.available_test_predictions = None

    def nan_predictions(self):
        if self.knn is None:
            return

        mask = (~np.isnan(self.knn.nearest_neighbors))  # False if nan
        neighbor_rating_available = mask[tuple([self.test_users, self.test_movies])]  # 0 if nan
        neighbor_rating_unavailable = ~neighbor_rating_available

        self.unavailable_output_predictions = self.output_predictions[neighbor_rating_unavailable]
        self.unavailable_test_predictions = self.test_predictions[neighbor_rating_unavailable]

        self.available_output_predictions = self.output_predictions[neighbor_rating_available]
        self.available_test_predictions = self.test_predictions[neighbor_rating_available]

        for i in range(5):
            rating = i + 1
            indices_available = self.available_test_predictions == rating
            indices_unavailable = self.unavailable_test_predictions == rating

            rsme_unavailable = data_processing.get_score(self.unavailable_output_predictions[indices_unavailable],
                                                         self.unavailable_test_predictions[indices_unavailable])
            rsme_available = data_processing.get_score(self.available_output_predictions[indices_available],
                                                       self.available_test_predictions[indices_available])
            print("RSME for users for prediction with rating", rating, " with available neighbors was ", rsme_available)
            print("RSME for users prediction with rating", rating, " with no available neighbors was ",
                  rsme_unavailable)

    def correct_range(self):

        rounded_predictions = np.rint(self.output_predictions)
        correct_range = rounded_predictions == self.test_predictions
        predicted_too_high = rounded_predictions > self.test_predictions
        predicted_too_low = rounded_predictions < self.test_predictions
        for i in range(5):
            rating = i + 1
            indices = rounded_predictions == rating
            print("** RATING : ** \t", rating)

            print("Correct: \t", np.count_nonzero(correct_range[indices]))
            print("Too high: \t", np.count_nonzero(predicted_too_high[indices]))
            print("Too low: \t", np.count_nonzero(predicted_too_low[indices]))

            rsme = self.partial_rmse(self.test_predictions == rating)
            print("RSME: \t", rsme)

        print("stop here")

    def partial_rmse(self, indices):
        return data_processing.get_score(self.output_predictions[indices], self.test_predictions[indices])

    def analyze_prediction(self):
        """ analyze the output predictions and what samples they performed good / bad on """
        print("Overall rsme: ", data_processing.get_score(self.output_predictions, self.test_predictions))

        self.correct_range()
        # for i in range(5):
        #     rating = i + 1
        #     rsme = self.partial_rmse(self.test_predictions == rating)
        #     print("RSME for ratings with actual value ", rating, " was ", rsme)
