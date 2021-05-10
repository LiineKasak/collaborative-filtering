import sys
import os
from pathlib import Path
directory = Path(__file__).parent


directory_path = os.path.abspath(directory)
data_path = os.path.join(directory_path, "/data/data_train.csv")

print("directory: ", directory_path, "       Data path: ", data_path)
sys.path.append(directory_path)  # add local path to the project here

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from helpers import data_processing
from singular_value_thresholding.svt import svt, asvt, svd_completion

from tqdm import tqdm

# Results:
#     a = np.sum(mask) / (np.prod(data.shape)*100)
#     b = np.prod(data.shape) * 0.6 / 100
#     MAX_ITERATIONS = 50
#       ASVT error was:  1.6831289023803888
#       SVD error was:  4.016538365960446
#       SVT error was:  1.9620087568030429

directory = Path(__file__)
data_path = os.path.join(directory, "../data/data_train.csv")

print("directory: ", directory, "       Data path: ", data_path)
sys.path.append(directory)  # add local path to the project here
# path to the train file
DATA_PATH = "data/test.csv"
print("directory: ", directory)

TRAIN_SIZE = 0.9  # size of training set
MAX_ITERATIONS = 100  # max number of iteration for svt and asvt


# After some experimenting, I find that a should be small, while B should be relatively larger.
def find_asvt_parameters(data, mask, test_users, test_movies, test_predictions):
    a1 = np.sum(mask) / np.prod(data.shape)
    b1 = np.prod(data.shape) * 0.6

    best_a = 1
    best_b = 1
    best_score = 10000000000

    for a in tqdm([a1 / 100, a1 / 10, a1, a1 * 10], desc='cross-validation', leave=False):
        for b in tqdm([b1 / 100, b1, b1 * 10], desc='cross-validation', leave=False):
            Y = asvt(input_matrix=data, mask=mask, max_iterations=MAX_ITERATIONS, a=a, b=b, disable=True)
            predictions = data_processing.extract_prediction_from_full_matrix(Y, test_users, test_movies)
            score = data_processing.get_score(predictions, test_predictions)

            if score < best_score:
                best_score = score
                best_a = a
                best_b = b
                print("new best: ", best_score, " where a = ", best_a, ", b = ", best_b)

    print("best score:", best_score, " where a = ", best_a, ", b = ", best_b)


def score_svt(data, mask, test_users, test_movies, test_predictions):
    Y_svt = svt(input_matrix=data, mask=mask, max_iterations=MAX_ITERATIONS, disable=True)

    predictions = data_processing.extract_prediction_from_full_matrix(Y_svt, test_users, test_movies)

    rmse_svt = data_processing.get_score(predictions, test_predictions)
    return rmse_svt


def score_asvt(data, mask, test_users, test_movies, test_predictions, a, b):
    Y_asvt = asvt(input_matrix=data, mask=mask, max_iterations=MAX_ITERATIONS, a=a, b=b, disable=True)
    predictions = data_processing.extract_prediction_from_full_matrix(Y_asvt, test_users, test_movies)
    rmse_asvt = data_processing.get_score(predictions, test_predictions)
    return rmse_asvt


def score_svd(data, mask, test_users, test_movies, test_predictions):
    Y_svd = svd_completion(input_matrix=data, k_singular_values=2)
    predictions = data_processing.extract_prediction_from_full_matrix(Y_svd, test_users, test_movies)
    rmse_svd = data_processing.get_score(predictions, test_predictions)
    return rmse_svd

#
def main():
    data_pd = pd.read_csv(DATA_PATH)
    train_size = TRAIN_SIZE
    train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)
    train_users, train_movies, train_predictions = data_processing.extract_users_items_predictions(train_pd)
    test_users, test_movies, test_predictions = data_processing.extract_users_items_predictions(test_pd)

    data, mask = data_processing.get_data_mask(train_users, train_movies, train_predictions)

    # parameters for asvt

    # find_asvt_parameters(data, mask, test_users, test_movies, test_predictions)

    a1 = np.sum(mask) / (np.prod(data.shape) * 100)
    b1 = np.prod(data.shape) * 0.6 / 100

    print('SVD error was: ', score_svd(data, mask, test_users, test_movies, test_predictions))
    print('ASVT error was: ', score_asvt(data, mask, test_users, test_movies, test_predictions, a1, b1))
    print('SVT error was: ', score_svt(data, mask, test_users, test_movies, test_predictions))


if __name__ == "__main__":
    main()
