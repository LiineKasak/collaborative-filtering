import src.utils.data_processing as data_processing
import pandas as pd
from surprise import Dataset, Reader
import torch
from torch.utils.data import DataLoader, TensorDataset


class DatasetClass:
    """
    Dataset-class to store information about the dataset at hand. Provides methods to efficiently access the data, and
    information about the statistics of the data.

    Parameters:
        users, movies, ratings: extracted users, items, ratings lists from the input data_pd
        users_dict: For a given user, store a list pairs of (movie, rating) provided by that user
        items_dict: For a given user, store a list pairs of (movie, rating) provided by that user
        triples: A list of triples [(u, m, r)], where user u gave movie m the rating r


    """

    def __init__(self, users, movies, ratings):
        print("start setting up dataset")
        self.users, self.movies, self.ratings = users, movies, ratings
        print("1")
        self.data, self.mask = data_processing.get_imputed_data_mask(self.users, self.movies, self.ratings)
        print("2")

        self.movie_dict, self.user_dict = data_processing.create_dicts(self.users, self.movies, self.ratings)
        print("3")

        self.triples = list(zip(self.users, self.movies, self.ratings))
        print("5")

        self.tuples = list(zip(self.users, self.movies))

        print("done setting up dataset")




    def get_users_movies_predictions(self):
        """ Return lists of users, movies and predictions """
        return self.users, self.movies, self.ratings

    # def get_data_statistics(self):
    #     """ Return the overall mean over all users and all movies as 2 separate dictionaries"""
    #     return self.user_mean_ratings, self.movie_mean_ratings

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
