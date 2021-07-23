import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from comet_ml import Experiment

from src.utils import data_processing, dataset


class AlgoBase():
    """ Base for all predictors, every predictor should inherit from this and implement
        (at least) a fit and predict method """

    def __init__(self, track_to_comet=False, method_name=None, api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        """ - initialize the method (number of users/movies, and the method name).
            - initialize the comet experiment if desired (default is no tracking)
            - if you want to track to a different comet workspace, you can pass arguments to it."""

        self.number_of_users, self.number_of_movies = (10000, 1000)

        # name of the method. Will be set when the method calls AlgoBase.__init__(self)
        if method_name:
            self.method_name = method_name
        else:
            self.method_name = self.__class__.__name__

        # print("method name:", self.method_name)
        self.track_on_comet = track_to_comet
        self.api_key = api_key
        self.projectname = projectname
        self.workspace = workspace
        self.tag = tag
        # initialize the comet experiment

    def start_comet(self):
        if self.track_on_comet:
            self.comet_experiment = Experiment(
                api_key=self.api_key,
                project_name=self.projectname,
                workspace=self.workspace,
            )
            self.comet_experiment.set_name(self.method_name)
            self.comet_experiment.add_tag(self.tag)

    def predict(self, users, movies):
        """ Predict ratings for a given set of users and movies """
        raise NotImplementedError("predict-function has to be implemented! ")

    def fit(self, users, movies, predictions):
        """ Train / Fit the predictor """
        raise NotImplementedError("fit-function has to be implemented! ")

    def fit(self, datawrapper):
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

        self.start_comet()

        kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)

        rmses = []

        bar = tqdm(total=folds, desc='cross_validation')

        for train_index, test_index in kfold.split(data_pd):
            train_users, train_movies, train_predictions = data_processing.extract_users_items_predictions(
                data_pd.iloc[train_index])
            trainset = dataset.DatasetWrapper(train_users, train_movies, train_predictions)

            val_users, val_movies, val_predictions = data_processing.extract_users_items_predictions(
                data_pd.iloc[test_index])

            self.fit(trainset)

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
        return rmses
