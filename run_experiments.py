# set up comet experiment
import comet_ml
from comet_ml import Experiment
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# add to path such that we can import helpers etc.
import os
from pathlib import Path

# !! If "module helpers not found" error appears run the following code: !!
# import sys
# directory = Path(__file__).parent
# directory_path = os.path.abspath(directory)
# sys.path.append(directory_path)

from helpers import data_processing
from src.matrix_factorization import sgd_factorization, non_negative_matrix_factorization, als_factorization
from src.svt import svt, asvt, svd_completion

experiment = Experiment(
    api_key="rISpuwcLQoWU6qan4jRCAPy5s",
    project_name="cil-experiments",
    workspace="veroniquek",
)


num_eigenvalues = 5
max_iterations = 1000
train_size = 0.9
reconstruction_mode = 'asvt'


def run_completion(X, mask, mode):
    if mode == 'svt':
        U, V = non_negative_matrix_factorization(X, max_iterations, num_eigenvalues)
    elif mode == 'asvt':
        U, V = sgd_factorization(X, max_iterations, num_eigenvalues)
    else:  # svd
        U, S, V = torch.linalg.svd(X, full_matrices=False)
        S = S[:num_eigenvalues]
        V = torch.diag(S) @ V


def run_reconstruction(X, mask, mode):

    if mode == 'als':
        U, V = als_factorization(X, max_iterations, num_eigenvalues)
        reconstructed_matrix = U @ V.t()
    elif mode == 'sgd':
        U, V = sgd_factorization(X, max_iterations, num_eigenvalues)
        reconstructed_matrix = U @ V.t()
    elif mode == 'nmf':
        U, V = non_negative_matrix_factorization(X, max_iterations, num_eigenvalues)
        reconstructed_matrix = U @ V.t()

    elif mode == 'svt':
        reconstructed_matrix = svt(X.detach().numpy(), mask, max_iterations)
    elif mode == 'asvt':
        a = np.sum(mask) / (np.prod(X.shape) * 100)
        b = np.prod(X.shape) * 0.6 / 100
        reconstructed_matrix = asvt(X.detach().numpy(), mask, max_iterations, a, b)
    else:
        reconstructed_matrix = svd_completion(X.detach().numpy(), num_eigenvalues)

    return reconstructed_matrix


def run_sgd(data):
    X = torch.from_numpy(data)
    U, V = sgd_factorization(X, 1000, 5)
    reconstructed_matrix = U @ V.t()

    return reconstructed_matrix


def main():
    directory = Path(__file__).parent
    directory_path = os.path.abspath(directory)
    DATA_PATH = directory_path + '/data/data_train.csv'
    data_pd = pd.read_csv(DATA_PATH)
    train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)
    train_users, train_movies, train_predictions = data_processing.extract_users_items_predictions(train_pd)
    test_users, test_movies, test_predictions = data_processing.extract_users_items_predictions(test_pd)

    train_data, mask = data_processing.get_data_mask(train_users, train_movies, train_predictions)

    train_data = torch.from_numpy(train_data)
    train_data.float()

    reconstructed_matrix = run_reconstruction(train_data, mask, reconstruction_mode)

    predictions = data_processing.extract_prediction_from_full_matrix(reconstructed_matrix, test_users, test_movies)

    experiment.log_metrics(
        {
            "root_mean_squared_error": data_processing.get_score(predictions, test_predictions)
        }
    )

    rmse = data_processing.get_score(predictions,  test_predictions)
    print("RMSE using " + reconstruction_mode + " is: {:.4f}".format(rmse))


if __name__ == "__main__":
    main()
