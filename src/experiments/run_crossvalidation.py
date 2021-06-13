import comet_ml
import numpy as np
from utils import data_processing
from src.models.surprise_baselines import SurpriseBaselines
from models.matrix_factorization import SVD, ALS, NMF
from models.ncf import NCF
from models.ensemble import Bagging
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
# als = ALS(track_to_comet=TRACK)
# nmf = NMF(track_to_comet=TRACK)




weak = SurpriseBaselines(predictor=KNNBaseline())
# ensemble20 = Bagging(predictor=slope, num_learners=20)
ensemble10 = Bagging(predictor=weak, num_learners=10)
ensemble5 = Bagging(predictor=weak, num_learners=5)

approaches = [weak, ensemble5, ensemble10]
# Read the data from the file:
data_pd = data_processing.read_data()


# cross validate all approaches
for approach in approaches:
    rmses = approach.cross_validate(data_pd, folds=5)

    print()
    print(approach.method_name, ":")
    print(rmses)
    print("Average: ", np.mean(rmses))
