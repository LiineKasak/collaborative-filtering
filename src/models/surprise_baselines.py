import numpy as np
from .algobase import AlgoBase
from utils import data_processing
from surprise import NormalPredictor


class SurpriseBaselines(AlgoBase):
    """ Prediction based on dimensionality reduction through singular value decomposition """

    def __init__(self, predictor=NormalPredictor(), track_to_comet=False, method_name=None, api_key="rISpuwcLQoWU6qan4jRCAPy5s", projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        AlgoBase.__init__(self, method_name=predictor.__class__.__name__ + "_surprise", track_to_comet=False, api_key="rISpuwcLQoWU6qan4jRCAPy5s", projectname="cil-experiments", workspace="veroniquek", tag="baseline")

        self.predictor = predictor     # random predictor; will be overwritten by the init function

    def fit(self, users, movies, predictions):
        trainset = data_processing.load_surprise_dataframe_from_arrays(users, movies, predictions).build_full_trainset()
        self.predictor.fit(trainset)

    def predict(self, users, movies):
        num_entries = len(users)
        predictions = np.zeros(num_entries)
        for i in range(num_entries):
            uid = str(users[i])  # raw user id (as in the ratings file). They are **strings**!
            iid = str(movies[i])  # raw item id (as in the ratings file). They are **strings**!
            predictions[i] = self.predictor.predict(uid, iid, verbose=False).est

        # get a prediction for specific users and items.
        return predictions
