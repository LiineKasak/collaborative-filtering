from comet_ml import Experiment
from surprise import Dataset, Reader
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from helpers import data_processing
import os
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

directory = Path(__file__).parent.parent
directory_path = os.path.abspath(directory)
DATA_PATH = directory_path + '/data/data_train.csv'


def parse(line):
    """ parses line and returns parsed row, column and value """
    m = re.search('r(.+?)_c(.+?),(.+?)', line.decode('utf-8'))
    row = int(m.group(1))
    column = int(m.group(2))
    value = int(m.group(3))
    return row, column, value

def parsef(line):
    """ parses line and returns parsed row, column and value """
    l1 = line.decode('utf-8').split(',')
    l2 = l1[0].split('_')
    row = int(l2[0][1:])
    column = int(l2[1][1:])
    value = float(l1[1])
    return row, column, value


def load_surprise_dataframe():
    data_pd = pd.read_csv(DATA_PATH)
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
    # load the train data in surprise format
    data = load_surprise_dataframe()

    # retrieve the trainset.
    trainset = data.build_full_trainset()

    # create SVD algorithm and train it
    algo = SVD(n_epochs=1)
    algo.fit(trainset)


    # validate it
    out = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


    # print errors
    print("RMSE: {0}, MAE: {1}".format(np.mean(out['test_rmse']), np.mean(out['test_mae'])))


if __name__ == "__main__":
    main()
