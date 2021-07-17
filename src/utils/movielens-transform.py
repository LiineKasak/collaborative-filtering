import data_processing
from random import randint

with open('movielens/users.dat', 'w') as f:
    for i in range(1, 10001):
        f.write(f'{i}::F::2::0::00000\n')

with open('movielens/movies.dat', 'w') as f:
    for i in range(1, 1001):
        f.write(f'{i}::n{i}::g\n')

data_pd = data_processing.read_data()
users, items, predictions = data_processing.extract_users_items_predictions(data_pd)
with open('movielens/ratings.dat', 'w') as f:
    for u, i, p in zip(users, items, predictions):
        f.write(f'{u+1}::{i+1}::{p}::{randint(1, 999999999)}\n')
