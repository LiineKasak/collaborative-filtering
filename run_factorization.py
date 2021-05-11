# add to path such that we can import helpers etc.
import sys
import os
import numpy as np
from pathlib import Path
directory = Path(__file__).parent
directory_path = os.path.abspath(directory)
DATA_PATH = directory_path+'/data/data_train.csv'


MAX_ITERATIONS = 100
print("directory: ", directory_path, "       Data path: ", DATA_PATH)
sys.path.append(directory_path)

import pandas as pd
import torch
import torch.nn as nn

from src.matrix_factorization import sgd_factorization, non_negative_matrix_factorization
from sklearn.model_selection import train_test_split


from helpers import data_processing

def main():
    data_pd = pd.read_csv(DATA_PATH)
    users, movies, predictions = data_processing.extract_users_items_predictions(data_pd)
    data, mask = data_processing.get_data_mask(users, movies, predictions)

    # test only on a subset..
    num_users, num_movies = 500, 200

    # X = torch.from_numpy(data[: num_users, :num_movies])
    X = torch.from_numpy(data)

    U, V = sgd_factorization(X, 1000, 5)

    print("shape U: ", U.shape)
    print("shape V: ", V.t().shape)


    recovered = U @ V.t()
    print(recovered[:5, :5])

    eps = 1e-6
    loss_fn = nn.MSELoss()
    rmse = torch.sqrt(loss_fn(U @ V.t(), X))  # add eps in case of 0

    print("Error: ", rmse.item())





if __name__ == "__main__":
    main()