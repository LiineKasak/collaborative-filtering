import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from helpers import data_processing
from singular_value_thresholding.svt import svt, asvt, svd_completion

from tqdm import tqdm

# path to the train file
DATA_PATH = '/Users/veronique/Documents/ETHZ/Master/2. Semester/CIL/Project/collaborative-filtering/data/data_train.csv'
TRAIN_SIZE = 0.9  # size of training set
MAX_ITERATIONS = 10  # max number of iteration for svt and asvt

# After some experimenting, I find that a should be small, while B should be relatively larger.
# For a in [0.0001, 0.01, 1.0, 100], b in [0.0001, 0.01, 1.0, 100], the best result was a = 0.0001, b = 100
def find_asvt_parameters(data, mask, test_users, test_movies, test_predictions):
    best_a = 1
    best_b = 1
    best_score = 10000000000

    a = 1e-9
    for b in tqdm([1e7, 1e8, 1e9], desc='cross-validation', leave=False):
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
    Y_svt = svt(input_matrix=data, mask=mask, max_iterations=MAX_ITERATIONS)

    predictions = data_processing.extract_prediction_from_full_matrix(Y_svt, test_users, test_movies)

    rmse_svt = data_processing.get_score(predictions, test_predictions)
    print('SVT error was: ', rmse_svt)


def score_asvt(data, mask, test_users, test_movies, test_predictions, a, b):
    Y_asvt = asvt(input_matrix=data, mask=mask, max_iterations=MAX_ITERATIONS, a=a, b=b)
    predictions = data_processing.extract_prediction_from_full_matrix(Y_asvt, test_users, test_movies)
    rmse_asvt = data_processing.get_score(predictions, test_predictions)
    return rmse_asvt


def score_svd(data, mask, test_users, test_movies, test_predictions):
    Y_svd = svd_completion(input_matrix=data, k_singular_values=2)
    predictions = data_processing.extract_prediction_from_full_matrix(Y_svd, test_users, test_movies)
    rmse_svd = data_processing.get_score(predictions, test_predictions)
    print('SVD error was: ', rmse_svd)


def main():
    data_pd = pd.read_csv(DATA_PATH)
    train_size = TRAIN_SIZE
    train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)
    train_users, train_movies, train_predictions = data_processing.extract_users_items_predictions(train_pd)
    test_users, test_movies, test_predictions = data_processing.extract_users_items_predictions(test_pd)

    data, mask = data_processing.get_data_mask(train_users, train_movies, train_predictions)

    # parameters for asvt
    a = 1e-7
    b = 1e7

    # find_asvt_parameters(data, mask, test_users, test_movies, test_predictions)

    print('SVD error was: ', score_svd(data, mask, test_users, test_movies, test_predictions))
    print('SVT error was: ', score_svt(data, mask, test_users, test_movies, test_predictions))
    print('ASVT error was: ', score_asvt(data, mask, test_users, test_movies, test_predictions, a, b))


if __name__ == "__main__":
    main()
