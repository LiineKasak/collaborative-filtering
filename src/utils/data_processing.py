""" Mostly functions adapted from the ones in the jupyter notebook provided by the course """
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from pathlib import Path
import os
from surprise import Dataset, Reader

number_of_users, number_of_movies = (10000, 1000)


def get_project_directory():
    directory = Path(__file__).parent.parent.parent
    directory_path = os.path.abspath(directory)
    return directory_path


def get_number_of_users():
    return number_of_users


def get_number_of_movies():
    return number_of_movies


def get_data_statistics(predictions):
    """Return mean and variance given input"""
    # TODO: Do this better (maybe return user- and item specific statistics, or something like that
    mean = np.nanmean(predictions)
    var = np.nanvar(predictions)

    return mean, var


def rmse(x, y):
    return math.sqrt(mean_squared_error(x, y))


def get_score(predictions, target_values):
    """Return the RMSE of the prediction, given the target values"""
    return rmse(predictions, target_values)


def extract_prediction_from_full_matrix(reconstructed_matrix, users, movies):
    """given some users and movies, predictions[i] contains rating of users[i] for movies[i],
    returns predictions for the users-movies combinations specified based on a full m \times n matrix"""

    assert (len(users) == len(movies)), "users-movies combinations specified should have equal length"
    predictions = np.zeros(len(users))

    for i, (user, movie) in enumerate(zip(users, movies)):
        predictions[i] = reconstructed_matrix[user][movie]

    return predictions


def extract_users_items_predictions(data_pd):
    """splits the r44_c1 into "user 44, movie 1" and returns 3 lists: user[i] gave movie[i] rating predictions[i]"""
    users, movies = \
        [np.squeeze(arr) for arr in
         np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


def read_data():
    """ Read the data from ./data/data_train.csv """
    directory_path = get_project_directory()
    data_pd = pd.read_csv(directory_path + '/data/data_train.csv')
    return data_pd


def get_data_mask(users, movies, predictions):
    """ given input data, return a mask containing 1 if the prediction is available,
    and a data matrix containing that prediction (using mean imputation) """
    data = np.full((number_of_users, number_of_movies), np.mean(predictions))
    mask = np.zeros((number_of_users, number_of_movies))  # 0 -> unobserved value, 1->observed value

    for user, movie, pred in zip(users, movies, predictions):
        data[user][movie] = pred
        mask[user][movie] = 1

    return data, mask


def get_users_movies_to_predict():
    """ Return the users and movies that we have to create a prediction for """
    directory_path = get_project_directory()
    submission_pd = pd.read_csv(directory_path + '/data/sampleSubmission.csv')
    sub_users, sub_movies, _ = extract_users_items_predictions(submission_pd)

    return sub_users, sub_movies


def load_surprise_dataframe_from_arrays(users, movies, predictions):
    # Creation of the dataframe. Column names are irrelevant.
    ratings_dict = {'users': users,
                    'items': movies,
                    'predictions': predictions}
    df = pd.DataFrame(ratings_dict)

    # The columns must correspond to user id, item id and ratings (in that order).
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['users', 'items', 'predictions']], reader=reader)
    return data


def load_surprise_dataframe_from_pd(data_pd):
    items, users, predictions = extract_users_items_predictions(data_pd)

    # Creation of the dataframe. Column names are irrelevant.
    ratings_dict = {'users': users,
                    'items': items,
                    'predictions': predictions}
    df = pd.DataFrame(ratings_dict)

    # The columns must correspond to user id, item id and ratings (in that order).
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['users', 'items', 'predictions']], reader=reader)
    return data


def create_submission_file(sub_users, sub_movies, predictions, name='submission'):
    """ predictions, create a file to submit to kaggle and store it under name.csv """

    directory_path = get_project_directory()

    pred_pd = pd.DataFrame(columns=['Id', 'Prediction'])
    pred_pd['Id'] = [f'r{v[0] + 1}_c{v[1] + 1}' for v in zip(sub_users, sub_movies)]
    pred_pd['Prediction'] = predictions

    # export to file:
    # archive_name is required to create working zip on my computer
    pred_pd.to_csv(directory_path + '/data/submissions/' + name + '.csv.zip',
                   index=False,
                   compression={'method': 'zip', 'archive_name': name + '.csv'}
                   )

    # submit from terminal:

    # submit from terminal:
    # !kaggle competitions submit cil-collaborative-filtering-2021 -f ./data/submissions/name.csv.zip -m '<message>'


def create_users_dict(users, movies, ratings):
    unique_users = np.unique(users)
    movies_ratings_dict = {}
    user_mean_rating_dict = {}
    for user_key in unique_users:
        user_list = []
        user_mean_rating = 0
        for idx, user in enumerate(users):
            if user == user_key:
                user_list.append((movies[idx], ratings[idx]))
                user_mean_rating += ratings[idx]

        movies_ratings_dict[user_key] = user_list
        user_mean_rating_dict[user_key] = user_mean_rating/len(user_list)

    return movies_ratings_dict, user_mean_rating_dict

def create_movies_dict(users, movies, ratings):
    unique_movies = np.unique(movies)
    users_ratings_dict = {}
    movie_mean_rating_dict = {}
    for movie_key in unique_movies:
        movie_list = []
        movie_mean_rating = 0
        for idx, movie in enumerate(movies):
            if movie == movie_key:
                movie_list.append((users[idx], ratings[idx]))
                movie_mean_rating += ratings[idx]

        users_ratings_dict[movie_key] = movie_list
        movie_mean_rating_dict[movie_key] = movie_mean_rating/len(movie_list)

    return users_ratings_dict, movie_mean_rating_dict