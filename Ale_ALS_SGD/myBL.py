import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from collections import defaultdict

# functions
#extracts user_id u, item_id i and rating value r_ui
def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract(
            'r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions

# read data
data_pd = pd.read_csv('data_train.csv')

#split data
train_size = 0.75
train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)

train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)
test_users, test_movies, test_predictions = extract_users_items_predictions(test_pd)

def create_adjacency_lists(train_user_ids, train_movie_ids, train_ratings):
    ur = defaultdict(list)
    ir = defaultdict(list)

    for i in range(len(train_users)):
        ur[train_users[i]].append((train_movies[i], train_predictions[i]))
    
    for i in range(len(train_movies)):
        ir[train_movies[i]].append((train_users[i], train_predictions[i]))

    return (ur, ir)

def fit_model_als(ur, ir, train_user_ids, train_movie_ids, train_ratings, num_epochs):
    global_average = np.mean(train_ratings)
    reg_u = 15
    reg_i = 10

    bi = np.zeros(len(ir.keys()))
    bu = np.zeros(len(ur.keys()))

    for dummy in range(num_epochs):
        for i in ir:
            sum_i = 0
            for idx in range(len(ir[i])):
                user_idx = ir[i][idx][0]
                rating = ir[i][idx][1]
                sum_i += rating - global_average - bu[user_idx]
            bi[i] = (sum_i/(reg_i+len(ir[i])))

        for u in ur:
            sum_u = 0
            for idx in range(len(ur[u])):
                item_idx = ur[u][idx][0]
                rating = ur[u][idx][1]
                sum_u += rating - global_average - bi[item_idx]
            bu[u] = (sum_u/(reg_u+len(ur[u])))

        

    return (bu, bi, global_average)



#fit model to data
def fit_model_sgd(ur, ir, train_user_ids, train_movie_ids, train_ratings, num_epochs):
    global_average = np.mean(train_ratings)
    learning_rate = 0.005
    reg = 0.02

    bu = np.zeros(len(ur.keys()))
    bi = np.zeros(len(ir.keys()))

    for dummy in range(num_epochs):
        for i in range(len(train_ratings)):
            error = train_ratings[i] - (global_average + bu[train_user_ids[i]] + bi[train_movie_ids[i]])
            bi[train_movie_ids[i]] += learning_rate * (error - reg*bi[train_movie_ids[i]])
            bu[train_user_ids[i]] += learning_rate * (error - reg*bu[train_user_ids[i]])
            
    
    return (bu, bi, global_average)

def predict(ur, ir, train_user_ids, train_movie_ids, test_user_ids, test_movie_ids, bu, bi, global_mean):
    predictions = np.zeros(len(test_user_ids))

    for i in range(len(test_user_ids)):
        predictions[i] = global_mean
        # wait....
        # I need to find out:
        #    - is the test_user_id_i an element of the train_user_ids array?
        #    - if yes, what is its position value?
        if( test_user_ids[i] in ur.keys()):
            predictions[i] += bu[test_user_ids[i]]
        if( test_movie_ids[i] in ir.keys()):
            predictions[i] += bi[test_movie_ids[i]]
    
    return predictions


(ur, ir) = create_adjacency_lists(train_users, train_movies, train_predictions)

print('#############################   starting... #############################')
(bu, bi, global_mean) = fit_model_als(ur, ir, train_users, train_movies, train_predictions, 1)
print('#############################   fit done... #############################')
print('#############################   start predictions... #############################')
predictions = predict(ur, ir, train_users, train_movies, test_users, test_movies, bu, bi, global_mean)

print('#############################   predictions done.... #############################')
print('#############################   calc rmse.... #############################')

from sklearn.metrics import mean_squared_error

print(mean_squared_error(test_predictions, predictions)**(1/2))

print('#############################   done! #############################')