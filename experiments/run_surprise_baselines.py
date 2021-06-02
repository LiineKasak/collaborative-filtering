import pandas as pd
from auxiliary import data_processing
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF, SlopeOne, CoClustering, KNNWithMeans, \
    KNNWithZScore, KNNBaseline

from src.surprise_baselines import SurpriseBaselines

# # read the sample submission file to get the values we have to predict
data_pd = data_processing.read_data()
users, movies, predictions = data_processing.extract_users_items_predictions(data_pd)
sub_users, sub_movies = data_processing.get_users_movies_to_predict()

surprise_methods = [SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF, SlopeOne, CoClustering, KNNWithMeans,
                    KNNWithZScore, KNNBaseline]

for method in surprise_methods:
    predictor = SurpriseBaselines(predictor=method)
    rmses = predictor.cross_validate(data_pd)
    print("RMSES of ", predictor.method_name, "\n", rmses, "\n")
