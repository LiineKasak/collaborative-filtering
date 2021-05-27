import pandas as pd
from auxiliary import processing


class AlgoBase():
    def __init__(self):
        # TODO: Initialize comet-experiment here to make sure things are tracked
        return

    def predict(self, sub_users, sub_movies):
        raise NotImplementedError("predict-function has to be implemented! ")

    def fit(self, sub_users, sub_movies):
        raise NotImplementedError("fit-function has to be implemented! ")

    def kaggle_predict(self):
        """ Predict and store the result to a kaggle-submisison file """
        # read the sample submission file to get the values we have to predict
        directory_path = processing.get_project_directory()
        submission_pd = pd.read_csv(directory_path + '/data/sampleSubmission.csv')
        sub_users, sub_movies, _ = processing.extract_users_items_predictions(submission_pd)

        predictions = self.predict(sub_users, sub_movies)

        processing.create_submission_file(sub_users, sub_movies, predictions)


