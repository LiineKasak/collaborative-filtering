from constants import *
from utils import DatasetUtil
import numpy as np

data_util = DatasetUtil()

model_name = 'svd_v1'
k_singular_values = 1
number_of_singular_values = min(NR_USERS, NR_MOVIES)

assert (k_singular_values <= number_of_singular_values), "choose correct number of singular values"

U, s, Vt = np.linalg.svd(data_util.data, full_matrices=False)

S = np.zeros((NR_MOVIES, NR_MOVIES))
S[:k_singular_values, :k_singular_values] = np.diag(s[:k_singular_values])

reconstructed_matrix = U.dot(S).dot(Vt)

train_score, test_score = data_util.rmse_scores_from_matrix(reconstructed_matrix)

print("RMSE using SVD is on train: {:.4f},   test: {:.4f}".format(train_score, test_score))
data_util.save_predictions_from_matrix(reconstructed_matrix, model_name)