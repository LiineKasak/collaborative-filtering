import numpy as np

from utils import data_processing
import pandas as pd
from surprise import Dataset, Reader
import torch
from torch.utils.data import DataLoader, TensorDataset


class DatasetWrapper:
    """
    Dataset-class to store information about the dataset at hand. Provides methods to efficiently access the data, and
    information about the statistics of the data.

    Parameters:
        users, movies, ratings: extracted users, items, ratings lists from the input data_pd
        users_dict: For a given user, store a list pairs of (movie, rating) provided by that user
        items_dict: For a given user, store a list pairs of (movie, rating) provided by that user
        triples: A list of triples [(u, m, r)], where user u gave movie m the rating r


    """

    def __init__(self, *args):
        if len(args) == 1:
            self.data_pd = args[0]
            self.users, self.movies, self.ratings = data_processing.extract_users_items_predictions(self.data_pd)
        else:
            self.users, self.movies, self.ratings = args[0], args[1], args[2]

        self.data_matrix, self.mask = data_processing.get_data_mask(self.users, self.movies, self.ratings, impute=False)

        self.movie_dict, self.user_dict = data_processing.create_dicts(self.users, self.movies, self.ratings)
        self.triples = list(zip(self.users, self.movies, self.ratings))

        self.num_users, self.num_movies = data_processing.get_number_of_users(), data_processing.get_number_of_movies()

        self.user_per_movie_encodings = None

        self.movies_per_user_representation()

        self.movie_means = np.nanmean(self.data_matrix, axis=0)
        self.user_means = np.nanmean(self.data_matrix, axis=1)

    def rating_available(self, user, query_movie):
        """ Determine if this user has already rated the query movie"""
        user_dict = self.user_dict[user]
        for (movie, rating) in user_dict:
            if movie == query_movie:
                return rating
            else:
                return 0

    def get_users_movies_predictions(self):
        """ Return lists of users, movies and predictions """
        return self.users, self.movies, self.ratings

    # def get_data_statistics(self):
    #     """ Return the overall mean over all users and all movies as 2 separate dictionaries"""
    #     return self.user_mean_ratings, self.movie_mean_ratings

    def get_data_and_mask(self):
        """ Return the data-matrix (users x movies) and the corresponding mask of available ratings """
        return self.data_matrix, self.mask

    def create_dataloader(self, batch_size, device=None):
        """ Create a pytorch dataloader of this dataset """
        if device is None:
            device = torch.device("cpu")

        users_torch = torch.tensor(self.users, device=device)
        movies_torch = torch.tensor(self.movies, device=device)
        predictions_torch = torch.tensor(self.ratings, device=device)

        dataloader = DataLoader(
            TensorDataset(users_torch, movies_torch, predictions_torch), batch_size=batch_size)
        return dataloader

    def create_surprise_trainset(self):
        """ create a surprise-trainset of this dataset """
        # Creation of the dataframe. Column names are irrelevant.
        ratings_dict = {'users': self.users,
                        'items': self.movies,
                        'predictions': self.ratings}
        df = pd.DataFrame(ratings_dict)

        # The columns must correspond to user id, item id and ratings (in that order).
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['users', 'items', 'predictions']], reader=reader)

        return data.build_full_trainset()

    def get_user_vector(self, user_id):
        return self.user_to_movie_vector(self.user_dict[user_id]).reshape(1, -1)

    def user_to_movie_vector(self, user_array):
        user_per_movie_encoding = np.zeros(self.num_movies)
        for (movie, rating) in user_array:
            user_per_movie_encoding[movie] = rating

        return user_per_movie_encoding

    def movies_per_user_representation(self):
        self.user_per_movie_encodings = np.zeros((self.num_users, self.num_movies))
        for user in range(self.num_users):
            self.user_per_movie_encodings[user] = self.user_to_movie_vector(self.user_dict[user])

        return self.user_per_movie_encodings

