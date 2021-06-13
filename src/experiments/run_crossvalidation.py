import comet_ml
import numpy as np
from utils import data_processing
from src.models.surprise_baselines import SurpriseBaselines
from models.matrix_factorization import SVD, ALS, NMF
from models.ncf import NCF
from models.ensemble import Bagging, MultiBagging
from surprise import SlopeOne, CoClustering, KNNWithMeans, KNNWithZScore, KNNBaseline

""" 
    Run Cross-Validation for multiple approaches at the same time.
    
"""

TRACK = False
# -------------
#
# Variables
#
# -------------
number_of_users = data_processing.get_number_of_users()
number_of_movies = data_processing.get_number_of_movies()

# -------------
#
# APPROACHES
#
# -------------
svd = SVD(k_singular_values=2,track_to_comet=TRACK)
# ncf = NCF(
#     number_of_users=number_of_users,
#     number_of_movies=number_of_movies,
#     embedding_size=16,
#     track_to_comet=TRACK
# )
#
als = ALS(track_to_comet=TRACK)
# nmf = NMF(track_to_comet=TRACK)


slope = SurpriseBaselines(predictor=SlopeOne())


# ensemble20 = Bagging(predictor=slope, num_learners=20)
ensemble_slope = Bagging(predictor=slope, num_learners=10)
ensemble_svd = Bagging(predictor=als, num_learners=10)
ensemble_als = Bagging(predictor=svd, num_learners=10)

approaches = [(slope, ensemble_slope), (als, ensemble_als), (svd, ensemble_svd)]

# Read the data from the file:
data_pd = data_processing.read_data()

# cross validate all approaches
for (approach, ens) in approaches:
    print("================================")
    print(approach.method_name)
    print("================================\n")

    rmses_apprach = approach.cross_validate(data_pd, folds=5)
    rmses_ensemble = ens.cross_validate(data_pd, folds=5)

    print("RMSE single learner: ", np.mean(rmses_apprach))
    print("RMSE bagged learners: ", np.mean(rmses_ensemble), "\n")
