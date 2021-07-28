import data_processing
from random import randint
from sklearn.model_selection import KFold


with open('data/movielens/users.dat', 'w') as f:
    for i in range(1, 10001):
        f.write(f'{i}::F::2::0::00000\n')

with open('data/movielens/movies.dat', 'w') as f:
    for i in range(1, 1001):
        f.write(f'{i}::n{i}::g\n')


data_pd = data_processing.read_data()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for fold_n, (train_index, test_index) in enumerate(kfold.split(data_pd)):
    train_users, train_movies, train_predictions = data_processing.extract_users_items_predictions(
        data_pd.iloc[train_index])
    val_users, val_movies, val_predictions = data_processing.extract_users_items_predictions(
        data_pd.iloc[test_index])
    with open(f'data/movielens/ratings_{fold_n}.dat', 'w') as f:
        for u, i, p in zip(train_users, train_movies, train_predictions):
            f.write(f'{u+1}::{i+1}::{p}::{randint(1, 999999999)}\n')
        for u, i, p in zip(val_users, val_movies, val_predictions):
            f.write(f'{u+1}::{i+1}::{p}::{randint(1, 999999999)}\n')
