from utils import data_processing, dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



class DataAnalyzer():
    def __init__(self, data_pd):
        self.data_wrapper = dataset.DatasetWrapper(data_pd)
        self.df = pd.DataFrame()

        self.df['users'] = self.data_wrapper.users
        self.df['movies'] = self.data_wrapper.movies
        self.df['ratings'] = self.data_wrapper.ratings

        return

    def create_histograms(self):
        plt.hist(self.df.ratings, bins=5)
        plt.show()

        plt.hist(self.df.movies, bins=self.data_wrapper.num_movies)
        plt.show()

        plt.hist(self.df.users, bins=self.data_wrapper.num_users)

        # plt.hist(self.df['ratings'], bins=[1,2,3,4,5])

        plt.show()

    def create_test_histograms(self, filename='sampleSubmission.csv', with_ratings=False):

        if with_ratings:
            directory_path = data_processing.get_project_directory()
            data_pd = pd.read_csv(directory_path + '/data/'+filename)
            users, movies, predictions = data_processing.extract_users_items_predictions(data_pd)
            self.query_df = pd.DataFrame()

            self.query_df['users'] = users
            self.query_df['movies'] = movies
            self.query_df['ratings'] = predictions

            self.query_df.hist(bins=5)
            plt.show()

        else:
            sub_users, sub_movies = data_processing.get_users_movies_from_file(filename)

            self.query_df = pd.DataFrame()

            self.query_df['users'] = sub_users
            self.query_df['movies'] = sub_movies
            self.query_df.hist()
            plt.show()

    def create_validation_plots(self, filename='sampleSubmission.csv'):
        directory_path = data_processing.get_project_directory()
        data_pd = pd.read_csv(directory_path + '/data/validation_outputs/' + filename)

        # self.create_validation_scatterplot(data_pd.Prediction, data_pd.GroundTruth)
        self.create_validation_histograms(data_pd.Prediction, data_pd.GroundTruth)

    def create_validation_histograms(self, test_predictions, actual_predictions):


        self.query_df = pd.DataFrame()

        self.query_df['output'] = test_predictions
        self.query_df['ground truth'] = actual_predictions

        self.query_df.hist()
        plt.show()

    def create_validation_scatterplot(self, test_predictions, actual_predictions):

        plt.figure(figsize=(10,10))
  


        plt.scatter(y=test_predictions, x=actual_predictions, linewidths=2, marker=".")
        plt.show()


class PredictionAnalyser:
    def __init__(self, test_users, test_movies, test_predictions, output_predictions, knn=None):
        self.test_users = test_users
        self.test_movies = test_movies
        self.test_predictions = test_predictions
        self.output_predictions = output_predictions
        self.overall_rsme = data_processing.get_score(output_predictions, test_predictions)

        self.knn = knn

        # self.nan_predictions()


    def nan_predictions(self):
        if self.knn is None:
            return

        mask = (~np.isnan(self.knn.nearest_neighbors))   # False if nan
        neighbor_rating_available = mask[tuple([self.test_users, self.test_movies])] # 0 if nan
        neighbor_rating_unavailable = ~neighbor_rating_available

        self.unavailable_output_predictions = self.output_predictions[neighbor_rating_unavailable]
        self.unavailable_test_predictions = self.test_predictions[neighbor_rating_unavailable]

        self.available_output_predictions = self.output_predictions[neighbor_rating_available]
        self.available_test_predictions = self.test_predictions[neighbor_rating_available]



        for i in range(5):
            rating = i + 1
            indices_available = self.available_test_predictions == rating
            indices_unavailable = self.unavailable_test_predictions == rating

            rsme_unavailable = data_processing.get_score(self.unavailable_output_predictions[indices_unavailable], self.unavailable_test_predictions[indices_unavailable])
            rsme_available = data_processing.get_score(self.available_output_predictions[indices_available], self.available_test_predictions[indices_available])
            print("RSME for users for prediction with rating", rating, " with available neighbors was ", rsme_available)
            print("RSME for users prediction with rating", rating, " with no available neighbors was ", rsme_unavailable)


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
