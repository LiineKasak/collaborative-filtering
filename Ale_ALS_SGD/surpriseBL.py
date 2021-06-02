import pandas as pd
import numpy as np
import math
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split


# prepare data for surprise
def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract(
            'r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions

data_pd = pd.read_csv('data_train.csv')
users, movies, predictions = extract_users_items_predictions(
    data_pd)
ratings_dict = {'userID': users,
                'itemID': movies,
                'rating': predictions}
df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

# split data
test_size = 0.25
trainset, testset = train_test_split(data, test_size=.25)

print('#############################   starting... #############################')

# use algorithm
bsl_options = {'method': 'sgd', 'n_epochs': 5}
algo = BaselineOnly(bsl_options=bsl_options)
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

print('#############################   done! #############################')
