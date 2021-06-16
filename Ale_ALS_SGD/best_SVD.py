import pandas as pd
import numpy as np
import math
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from surprise.model_selection import GridSearchCV

# prepare data


def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract(
            'r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


data_pd = pd.read_csv('data_train.csv')
test_users, test_movies, test_predictions = extract_users_items_predictions(
    data_pd)
ratings_dict = {'itemID': test_movies,
                'userID': test_users,
                'rating': test_predictions}
df = pd.DataFrame(ratings_dict)

# run improved SVD via surprise
algo = SVD(n_factors=1, n_epochs=50, lr_all=0.001, reg_bu=0.005, reg_bi=0.005, 
reg_pu=0.06, reg_qi=0.06)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

param_grid = {'init_mean': [0, 1], 'init_std_dev': [0.0, 0.1], 'n_factors': [1,2], 'n_epochs': [20], 'lr_all': [0.001], 'reg_bu': [0.01,0.02, 0.005],
              'reg_bi': [0.01, 0.02, 0.005], 'reg_all': [0.02, 0.06]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

print('finished new new')
