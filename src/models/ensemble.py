import random
import numpy as np
from models.algobase import AlgoBase


"""
Input: 
    - approaches: array of AlgoBase approaches
"""
class Bagging(AlgoBase):
    def __init__(self, methods, track_to_comet=False, method_name=None, api_key="rISpuwcLQoWU6qan4jRCAPy5s", projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key, projectname=projectname, workspace=workspace, tag=tag)

        self.methods = methods
        self.num_methods = len(methods)

    def partition(self, users, movies, predictions, n):
        temp = list(zip(users, movies, predictions))
        random.Random(42).shuffle(temp)
        shuffled_users, shuffled_movies, shuffled_predictions = zip(*temp)

        return [list[i::n] for i in range(n)]

    def fit(self, users, movies, predictions):
        for i, method in enumerate(self.methods):
            method.fit(users, movies, predictions)

    def predict(self, users, movies):
        overall_predictions = []
        for method in self.methods:
            method_predictions = method.predict(users, movies)
            overall_predictions.append(method_predictions)

        # Return average predictions
        return np.divide(np.sum(overall_predictions, axis=0),self.num_methods).tolist()
