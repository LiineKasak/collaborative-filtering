from utils import data_processing
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF, SlopeOne, CoClustering, KNNWithMeans, \
    KNNWithZScore, KNNBaseline
import numpy as np
from models.surprise_baselines import SurpriseBaselines

# # read the sample submission file to get the values we have to predict
data_pd = data_processing.read_data()
users, movies, predictions = data_processing.extract_users_items_predictions(data_pd)
sub_users, sub_movies = data_processing.get_users_movies_to_predict()

surprise_methods = [SlopeOne(), CoClustering(), KNNWithMeans()]


method5 = KNNWithMeans(k=5)
method10 = KNNWithMeans(k=10)
method40 = KNNWithMeans(k=40)
method100 = KNNWithMeans(k=100)



knn_s = [method5, method10, method40, method100]

for method in knn_s:
    predictor = SurpriseBaselines(predictor=method, track_to_comet=False)
    rmses = predictor.cross_validate(data_pd)
    print("RMSES of ", predictor.method_name, "\n \t", rmses, "\n")
