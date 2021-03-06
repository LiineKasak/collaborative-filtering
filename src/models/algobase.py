import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils.dataset import DatasetWrapper
import pickle

from utils import data_processing, dataset


class AlgoBase:
    """ Base for all predictors, every predictor should inherit from this and implement
        (at least) a fit and predict method """

    def __init__(self):
        """ - initialize the method (number of users/movies, and the method name).
            - initialize the comet experiment if desired (default is no tracking)
        """
        self.number_of_users, self.number_of_movies = (10000, 1000)

        self.method_name = self.__class__.__name__

    @staticmethod
    def default_params():
        """Default parameters for model."""
        raise NotImplementedError("default_params-function has to be implemented! ")

    def predict(self, users, movies):
        """ Predict ratings for a given set of users and movies """
        raise NotImplementedError("predict-function has to be implemented! ")

    def fit(self, data):
        """ Train / Fit the predictor """
        raise NotImplementedError("fit-function has to be implemented! ")

    def tune_params(self, train_data: DatasetWrapper, test_data: DatasetWrapper):
        """ Hyper-parameter tuning """
        raise NotImplementedError("tune_params-function has to be implemented! ")

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

        bar = tqdm(total=folds, desc='cross_validation')

        for train_index, test_index in kfold.split(data_pd):
            train_users, train_movies, train_predictions = data_processing.extract_users_items_predictions(
                data_pd.iloc[train_index])
            trainset = dataset.DatasetWrapper(train_users, train_movies, train_predictions)

            val_users, val_movies, val_predictions = data_processing.extract_users_items_predictions(
                data_pd.iloc[test_index])

            self.fit(trainset)

            predictions = self.predict(val_users, val_movies)
            rmse = data_processing.get_score(predictions, val_predictions)
            rmses.append(rmse)

            bar.update()

        bar.close()
        return rmses

    def save(self, filename: str):
        pickle.dump(self, open(filename, 'wb'))
