import random
import numpy as np
from models.algobase import AlgoBase


class Bagging(AlgoBase):
    """
    Input:
        - approachs: AlgoBase approach
        - NumLearnes: Number of weak learners to be combined
    """

    def __init__(self, predictor, num_learners=1, track_to_comet=False, method_name=None, api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)
        self.methods = []
        for i in range(num_learners):
            self.methods.append(predictor)

        self.num_learners = num_learners

        np.random.seed(42)

    def bootstrap(self, users, movies, predictions):
        sample = np.random.choice(len(users), size=len(users))
        return users[sample], movies[sample], predictions[sample]

    def fit(self, users, movies, predictions):
        for i, method in enumerate(self.methods):
            bootstrap_users, bootstrap_movies, bootstrap_predictions = self.bootstrap(users, movies, predictions)
            method.fit(bootstrap_users, bootstrap_movies, bootstrap_predictions)

    def predict(self, users, movies):
        overall_predictions = []
        for method in self.methods:
            method_predictions = method.predict(users, movies)
            overall_predictions.append(method_predictions)

        # Return average predictions
        return np.divide(np.sum(overall_predictions, axis=0), self.num_learners).tolist()


class MultiBagging(AlgoBase):
    """
    Combine multiple different weak learners to a stronger one using averaging
    Input:
        - approaches: array of AlgoBase approaches
    """

    def __init__(self, methods, track_to_comet=False, method_name=None, api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, track_to_comet=track_to_comet, method_name=method_name, api_key=api_key,
                          projectname=projectname, workspace=workspace, tag=tag)

        self.methods = methods
        self.num_methods = len(methods)

        np.random.seed(42)

    def bootstrap(self, users, movies, predictions):
        sample = np.random.choice(len(users), size=len(users))
        return users[sample], movies[sample], predictions[sample]

    def fit(self, users, movies, predictions):
        for i, method in enumerate(self.methods):
            bootstrap_users, bootstrap_movies, bootstrap_predictions = self.bootstrap(users, movies, predictions)
            method.fit(bootstrap_users, bootstrap_movies, bootstrap_predictions)

    def predict(self, users, movies):
        overall_predictions = []
        for method in self.methods:
            method_predictions = method.predict(users, movies)
            overall_predictions.append(method_predictions)

        # Return average predictions
        return np.divide(np.sum(overall_predictions, axis=0), self.num_methods).tolist()
