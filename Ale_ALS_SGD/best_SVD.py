import pandas as pd
import numpy as np
import math
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from sklearn.model_selection import train_test_split

# prepare data
def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions

data_pd = pd.read_csv('data_train.csv')
train_size = 0.9
train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)
test_users, test_movies, test_predictions = extract_users_items_predictions(test_pd)
ratings_dict = {'itemID': test_movies,
                'userID': test_users,
                'rating': test_predictions}
df = pd.DataFrame(ratings_dict)

# run improved SVD via surprise
algo = SVD(n_factors=2)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)