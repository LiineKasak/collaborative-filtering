import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from comet_ml import Experiment

from auxiliary import data_processing


class AlgoBase():
    """ Base for all predictors, every predictor should inherit from this and implement
        (at least) a fit and predict method """

    def __init__(self, track_to_comet=False):
        self.number_of_users, self.number_of_movies = (10000, 1000)

        # name of the method. Will be set when the method calls AlgoBase.__init__(self)
        self.method_name = self.__class__.__name__
        self.track_on_comet = track_to_comet

        # initialize the comet experiment
        if self.track_on_comet:
            self.comet_experiment = Experiment(
                api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                project_name="cil-experiments",
                workspace="veroniquek",
            )
            self.comet_experiment.set_name(self.method_name)
            self.comet_experiment.add_tag("baseline")

    def predict(self, users, movies):
        """ Predict ratings for a given set of users and movies """
        raise NotImplementedError("predict-function has to be implemented! ")

    def fit(self, users, movies, predictions):
        """ Train / Fit the predictor """
        raise NotImplementedError("fit-function has to be implemented! ")

    def predict_for_submission(self, name="submission"):
        """ Predict and store the result to a kaggle-submisison file """
        # read the sample submission file to get the values we have to predict
        directory_path = data_processing.get_project_directory()
        submission_pd = pd.read_csv(directory_path + '/data/sampleSubmission.csv')
        sub_users, sub_movies, _ = data_processing.extract_users_items_predictions(submission_pd)

        predictions = self.predict(sub_users, sub_movies)

        data_processing.create_submission_file(sub_users, sub_movies, predictions, name=name)

    def cross_validate(self, data_pd, folds=5, random_state=42):
        """ Run Crossvalidation using kfold, taking a pandas-dataframe of the raw data as input
            (as it is read in from the .csv file) """
        kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)

        rmses = []

        for train_index, test_index in tqdm(kfold.split(data_pd), desc='cross_validation'):
            train_users, train_movies, train_predictions = data_processing.extract_users_items_predictions(
                data_pd.iloc[train_index])
            val_users, val_movies, val_predictions = data_processing.extract_users_items_predictions(
                data_pd.iloc[test_index])

            self.fit(train_users, train_movies, train_predictions)

            predictions = self.predict(val_users, val_movies)
            rmses.append(data_processing.get_score(predictions, val_predictions))

        mean_rmse = np.mean(rmses)
        # track mean rmses to comet if we are tracking
        if self.track_on_comet:
            self.comet_experiment.log_metrics(
                {
                    "root_mean_squared_error": mean_rmse
                }
            )

        return rmses
