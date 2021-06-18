import data_processing
import pandas as pd
from surprise import Dataset, Reader
import torch
from torch.utils.data import DataLoader, TensorDataset


class Dataset:
    """
    Dataset-class to store information about the dataset at hand. Provides methods to efficiently access the data, and
    information about the statistics of the data.

    Parameters:
        users, movies, ratings: extracted users, items, ratings lists from the input data_pd
        users_dict: For a given user, store a list pairs of (movie, rating) provided by that user
        items_dict: For a given user, store a list pairs of (movie, rating) provided by that user
        triples: A list of triples [(u, m, r)], where user u gave movie m the rating r


    """

    def __init__(self, data_pd):
        self.data_pd = data_pd
        self.users, self.movies, self.ratings = data_processing.extract_users_items_predictions(data_pd)

        self.data, self.mask = data_processing.get_data_mask(self.users, self.movies, self.ratings)

        self.user_dict, self.user_mean_ratings = data_processing.create_users_dict(self.users, self.movies,
                                                                                   self.ratings)

        self.movie_dict, self.movie_mean_ratings = data_processing.create_movies_dict(self.users, self.movies,

                                                                                      self.ratings)
        self.triples = list(zip(self.users, self.movies, self.ratings))

    def get_users_movies_predictions(self):
        """ Return lists of users, movies and predictions """
        return self.users, self.movies, self.ratings

    def get_data_statistics(self):
        """ Return the overall mean over all users and all movies as 2 separate dictionaries"""
        return self.user_mean_ratings, self.movie_mean_ratings

    def get_data_and_mask(self):
        """ Return the data-matrix (users x movies) and the corresponding mask of available ratings """
        return self.data, self.mask

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