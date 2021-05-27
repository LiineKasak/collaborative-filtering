import pandas as pd
from auxiliary import processing
from src.algobase import AlgoBase

class SVD(AlgoBase):
    def __init__(self):
        # TODO: Initialize comet-experiment here to make sure things are tracked
        return

    def fit(self, input_data):
        # TODO
        return

    def predict(self, users, movies):
        directory_path = processing.get_project_directory()
        submission_pd = pd.read_csv(directory_path + '/data/sampleSubmission.csv')
        sub_users, sub_movies, sub_predictions = processing.extract_users_items_predictions(submission_pd)
        return sub_predictions
        # return


