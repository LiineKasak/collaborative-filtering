from comet_ml import Experiment
from surprise import Dataset, Reader
from surprise import SVD
from surprise import Dataset
from surprise import accuracy

from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from helpers import data_processing
import os
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split

directory = Path(__file__).parent.parent
directory_path = os.path.abspath(directory)
DATA_PATH = directory_path + '/data/data_train.csv'

my_seed = 42
random.seed(my_seed)
np.random.seed(my_seed)


def load_surprise_dataframe(data_pd):
    items, users, predictions = data_processing.extract_users_items_predictions(data_pd)

    # Creation of the dataframe. Column names are irrelevant.
    ratings_dict = {'users': users,
                    'items': items,
                    'predictions': predictions}
    df = pd.DataFrame(ratings_dict)

    # The columns must correspond to user id, item id and ratings (in that order).
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['users', 'items', 'predictions']], reader=reader)
    return data


def main():
    data_pd = pd.read_csv(DATA_PATH)
    train_pd, test_pd = train_test_split(data_pd, train_size=0.9)

    # load the train data in surprise format for training
    train_data_surprise = load_surprise_dataframe(train_pd)
    test_data_surprise = load_surprise_dataframe(test_pd)

    # retrieve the trainset.
    trainset = train_data_surprise.build_full_trainset()

    # retrieve testset
    raw_testset = test_data_surprise.raw_ratings
    testset = train_data_surprise.construct_testset(raw_testset)

    # create SVD algorithm and train it
    svd = SVD(n_epochs=1)
    svd.fit(trainset)

    predictions = svd.test(testset)

    # validate it
    # out = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    rmse = accuracy.rmse(predictions)
    print("RMSE is: {:.4f}".format(rmse))




if __name__ == "__main__":
    main()
