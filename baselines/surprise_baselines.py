from comet_ml import Experiment
from surprise import Dataset, Reader
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, SVDpp, NMF, SlopeOne, CoClustering, KNNWithMeans, \
    KNNWithZScore, KNNBaseline
from surprise import Dataset
from surprise import accuracy

from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from auxiliary import data_processing
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

load_to_comet = True


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


def surprise_baselineOnly(trainset, testset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_baselineOnly")
        experiment.add_tag("surprise")

        # create SVD algorithm and train it
    pred = BaselineOnly()
    pred.fit(trainset)

    predictions = pred.test(testset)

    rmse = accuracy.rmse(predictions)

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse
            }
        )


def surprise_normalPredictor(trainset, testset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_normalPredictor")
        experiment.add_tag("surprise")

        # create SVD algorithm and train it
    pred = NormalPredictor()
    pred.fit(trainset)

    predictions = pred.test(testset)

    rmse = accuracy.rmse(predictions)

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse
            }
        )


def surprise_knnBasic(trainset, testset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_svd")
        experiment.add_tag("surprise")

    # create SVD algorithm and train it
    pred = KNNBasic()
    pred.fit(trainset)

    predictions = pred.test(testset)

    rmse = accuracy.rmse(predictions)

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse
            }
        )


def surprise_svd(trainset, testset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_svd_20epochs")
        experiment.add_tag("surprise")

    # create SVD algorithm and train it
    pred = SVD(n_epochs=20)
    pred.fit(trainset)

    predictions = pred.test(testset)

    rmse = accuracy.rmse(predictions)

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse
            }
        )


def surprise_svdpp(trainset, testset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_svdpp_20epochs")
        experiment.add_tag("surprise")

    # create SVD algorithm and train it
    pred = SVDpp(n_epochs=20)
    pred.fit(trainset)

    predictions = pred.test(testset)

    rmse = accuracy.rmse(predictions)

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse
            }
        )


def surprise_nmf(trainset, testset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_nmf_20epochs")
        experiment.add_tag("surprise")

    # create SVD algorithm and train it
    pred = NMF(n_epochs=20)
    pred.fit(trainset)

    predictions = pred.test(testset)

    rmse = accuracy.rmse(predictions)

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse
            }
        )


def surprise_slope_one(trainset, testset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_svdpp_20epochs")
        experiment.add_tag("surprise")

    # create SVD algorithm and train it
    pred = SlopeOne()
    pred.fit(trainset)

    predictions = pred.test(testset)

    rmse = accuracy.rmse(predictions)

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse
            }
        )


def surprise_co_clustering(trainset, testset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_svdpp_20epochs")
        experiment.add_tag("surprise")

    # create SVD algorithm and train it
    pred = CoClustering()
    pred.fit(trainset)

    predictions = pred.test(testset)

    rmse = accuracy.rmse(predictions)

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse
            }
        )


# crossvalidate using surprise.cross_validate

def surprise_baselineOnly_crossval(dataset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_baselineOnly")
        experiment.add_tag("surprise")

        # create SVD algorithm and train it
    pred = BaselineOnly()
    metrics = cross_validate(pred, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    rmse = np.mean(metrics['test_rmse'])
    mae = np.mean(metrics['test_mae'])

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse,
                "mean absolute error": mae
            }
        )


def surprise_normalPredictor_crossval(dataset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_normalPredictor")
        experiment.add_tag("surprise")

        # create SVD algorithm and train it
    pred = NormalPredictor()
    metrics = cross_validate(pred, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    rmse = np.mean(metrics['test_rmse'])
    mae = np.mean(metrics['test_mae'])

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse,
                "mean absolute error": mae
            }
        )


def surprise_knnBasic_crossval(dataset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_knnBasic")
        experiment.add_tag("surprise")

    # create SVD algorithm and train it
    pred = KNNBasic()
    metrics = cross_validate(pred, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    rmse = np.mean(metrics['test_rmse'])
    mae = np.mean(metrics['test_mae'])

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse,
                "mean absolute error": mae
            }
        )


def surprise_svd_crossval(dataset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_svd_20epochs")
        experiment.add_tag("surprise")

    # create SVD algorithm and train it
    pred = SVD(n_epochs=20)
    metrics = cross_validate(pred, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    rmse = np.mean(metrics['test_rmse'])
    mae = np.mean(metrics['test_mae'])

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse,
                "mean absolute error": mae
            }
        )


def surprise_svdpp_crossval(dataset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_svdpp_20epochs")
        experiment.add_tag("surprise")

    # create SVD algorithm and train it
    pred = SVDpp(n_epochs=20)
    metrics = cross_validate(pred, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    rmse = np.mean(metrics['test_rmse'])
    mae = np.mean(metrics['test_mae'])

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse,
                "mean absolute error": mae
            }
        )


def surprise_nmf_crossval(dataset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_nmf_20epochs")
        experiment.add_tag("surprise")

    # create SVD algorithm and train it
    pred = NMF(n_epochs=20)
    metrics = cross_validate(pred, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    rmse = np.mean(metrics['test_rmse'])
    mae = np.mean(metrics['test_mae'])

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse,
                "mean absolute error": mae
            }
        )


def surprise_slope_one_crossval(dataset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_slopeOne")
        experiment.add_tag("surprise")

    # create SVD algorithm and train it
    pred = SlopeOne()
    metrics = cross_validate(pred, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    rmse = np.mean(metrics['test_rmse'])
    mae = np.mean(metrics['test_mae'])

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse,
                "mean absolute error": mae
            }
        )


def surprise_co_clustering_crossval(dataset):
    if load_to_comet:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
        experiment.set_name("surprise_co_clustering")
        experiment.add_tag("surprise")

    # create SVD algorithm and train it
    pred = CoClustering()
    metrics = cross_validate(pred, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    rmse = np.mean(metrics['test_rmse'])
    mae = np.mean(metrics['test_mae'])

    if load_to_comet:
        experiment.log_metrics(
            {
                "root_mean_squared_error": rmse,
                "mean absolute error": mae
            }
        )


def main():
    data_pd = pd.read_csv(DATA_PATH)
    train_pd, test_pd = train_test_split(data_pd, train_size=0.9)

    data_surprise = load_surprise_dataframe(data_pd)
    # load the train data in surprise format for training
    # train_data_surprise = load_surprise_dataframe(train_pd)
    # test_data_surprise = load_surprise_dataframe(test_pd)
    #
    # # retrieve the trainset.
    # trainset = train_data_surprise.build_full_trainset()
    #
    # # retrieve testset
    # raw_testset = test_data_surprise.raw_ratings
    # testset = train_data_surprise.construct_testset(raw_testset)

    # run cross validation of all approaches
    surprise_baselineOnly_crossval(data_surprise)
    # surprise_normalPredictor_crossval(data_surprise)
    # surprise_svd_crossval(data_surprise)
    # surprise_nmf_crossval(data_surprise)
    # surprise_knnBasic_crossval(data_surprise)
    # surprise_co_clustering_crossval(data_surprise)
    # surprise_slope_one_crossval(data_surprise)
    # surprise_svdpp_crossval(data_surprise)


if __name__ == "__main__":
    main()
