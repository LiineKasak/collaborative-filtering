import pandas as pd
import numpy as np
import math
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate


# prepare data for surprise
def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract(
            'r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions

import time
start_time = time.time()

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
trainset, testset = train_test_split(data, test_size=test_size)

print('#############################  run code SVD surprise... #############################')

# use algorithm
algo = SVD(n_factors=2, n_epochs=1)
#algo.fit(trainset)
#predictions = algo.test(testset)

# Then compute RMSE
#accuracy.rmse(predictions)

# Do 3-fold cross-validation
res = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=False)

print(np.mean(res['test_rmse']))
print("--- %s seconds ---" % (time.time() - start_time))
print('#############################   done! #############################')
