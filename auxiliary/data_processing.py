# test our predictions with the true values

from sklearn.metrics import mean_squared_error
import math
import numpy as np
import pandas as pd

number_of_users, number_of_movies = (10000, 1000)  # todo: want this as argument?


def get_data_statistics(predictions):
    mean = np.nanmean(predictions)
    var = np.nanvar(predictions)

    return mean, var

def rmse(x, y):
    return math.sqrt(mean_squared_error(x, y))


# return RMSE of predictions
def get_score(predictions, target_values):
    return rmse(predictions, target_values)


# given some users and movies, predictions[i] contains rating of users[i] for movies[i]
def extract_prediction_from_full_matrix(reconstructed_matrix, users, movies):
    # returns predictions for the users-movies combinations specified based on a full m \times n matrix
    assert (len(users) == len(movies)), "users-movies combinations specified should have equal length"
    predictions = np.zeros(len(users))

    for i, (user, movie) in enumerate(zip(users, movies)):
        predictions[i] = reconstructed_matrix[user][movie]

    return predictions


# splits the r44_c1 into "user 44, movie 1" and returns 3 lists: user[i] gave movie[i] rating predictions[i]
def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in
         np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


# return a mask containing 1 if the prediction is available,
# and a data matrix containing that prediction (mean imputation)
def get_data_mask(users, movies, predictions):
    data = np.full((number_of_users, number_of_movies), np.mean(predictions))
    mask = np.zeros((number_of_users, number_of_movies))  # 0 -> unobserved value, 1->observed value

    for user, movie, pred in zip(users, movies, predictions):
        data[user - 1][movie - 1] = pred
        mask[user - 1][movie - 1] = 1

    return data, mask


def get_number_of_users():
    return number_of_users


def get_number_of_movies():
    return number_of_movies
