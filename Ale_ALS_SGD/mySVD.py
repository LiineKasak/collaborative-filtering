import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from joblib import Parallel
from joblib import delayed

# functions
#extracts user_id u, item_id i and rating value r_ui
def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract(
            'r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions

def create_adjacency_lists(train_user_ids, train_movie_ids, train_ratings):
    ur = defaultdict(list)
    ir = defaultdict(list)

    for i in range(len(train_user_ids)):
        ur[train_user_ids[i]].append((train_movie_ids[i], train_ratings[i]))
    
    for i in range(len(train_movie_ids)):
        ir[train_movie_ids[i]].append((train_user_ids[i], train_ratings[i]))

    return (ur, ir)

def fit_model_baseline_als(ur, ir, train_user_ids, train_movie_ids, train_ratings, num_epochs):
    global_average = np.mean(train_ratings)
    reg_u = 15
    reg_i = 10

    bi = np.zeros(len(ir.keys())+1)
    bu = np.zeros(len(ur.keys())+1)

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
def fit_model_svd(bu, bi, ur, ir, train_user_ids, train_movie_ids, train_ratings, num_epochs):
    global_average = np.mean(train_ratings)
    learning_rate = 0.001
    reg = 0.02
    reg_b = 0.005
    reg_pq = 0.06
    nr_factors = 2
    nr_users = len(ur.keys())+1
    nr_items = len(ir.keys())+1
    init_mean = 0.0
    init_std = 0.1

    pu = np.random.normal(init_mean, init_std, size=(nr_factors, nr_users))
    qi = np.random.normal(init_mean, init_std, size=(nr_factors, nr_items))

    for dummy in range(num_epochs):
        for i in range(len(train_ratings)):
            u_id = train_user_ids[i]
            m_id = train_movie_ids[i]
            r_i = train_ratings[i]
            bu_i = bu[u_id]
            bi_i = bi[m_id]
            pu_i = pu[:,u_id]
            qi_i = qi[:,m_id]

            dot = 0
            #compute dot by hand because you index the vector by element
            for f in range(nr_factors):
                dot += qi[f, m_id] * pu[f, u_id]

            error = r_i - (global_average + bu_i + bi_i + dot)
            bi[m_id] += learning_rate * (error - reg_b*bi_i)
            bu[u_id] += learning_rate * (error - reg_b*bu_i)

            for f in range(nr_factors):
                puf = pu[f,u_id]
                qif = qi[f, m_id]
                pu[f,u_id] += learning_rate * (error * qif - reg_pq * puf)
                qi[f,m_id] += learning_rate * (error * puf - reg_pq * qif)
    
    #return (bu, bi, pu, qi, global_average)
    return(1,1,1,1,1)

def predict_svd(ur, ir, train_user_ids, train_movie_ids, test_user_ids, test_movie_ids, bu, bi, pu, qi, global_mean):
    predictions = np.zeros(len(test_user_ids))
    nr_features = pu.shape[0]

    for i in range(len(test_user_ids)):
        predictions[i] = global_mean
        u = test_user_ids[i]
        m = test_movie_ids[i]
        # wait....
        # I need to find out:
        #    - is the test_user_id_i an element of the train_user_ids array?
        #    - if yes, what is its position value?
        is_user_known = u in ur.keys()
        is_movie_known = m in ir.keys()
        if(is_user_known):
            predictions[i] += bu[u]
        if(is_movie_known):
            predictions[i] += bi[m]
        if(is_user_known and is_movie_known):
            predictions[i] += np.dot(pu[:, u].T, qi[:, m])
    
    return predictions


def fit_and_score(train_index, test_index):
    train_users, train_movies, train_predictions = extract_users_items_predictions(data_pd.iloc[train_index])
    val_users, val_movies, val_predictions = extract_users_items_predictions(data_pd.iloc[test_index])
    print('#############################   starting my SVD regular... #############################')
    (ur, ir) = create_adjacency_lists(train_users, train_movies, train_predictions)
    bu = np.zeros(len(ur.keys())+1)
    bi = np.zeros(len(ir.keys())+1)
    #(bu, bi, global_mean) = fit_model_baseline_als(ur, ir, train_users, train_movies, train_predictions, 1)
    (bu, bi, pu, qi, global_mean) = fit_model_svd(bu, bi, ur, ir, train_users, train_movies, train_predictions, 1)
    print('#############################   start predictions... #############################')
    predictions = predict_svd(ur, ir, train_users, train_movies, val_users, val_movies, bu, bi, pu, qi, global_mean)
    error = mean_squared_error(val_predictions, predictions)**(1/2)
    print('error... ', error)
    return error

def cross_validate(data_pd, nr_folds):
    kfold = KFold(n_splits=nr_folds, shuffle=True, random_state=41)

    errors = []
    n_jobs = 1
    delayed_list = (delayed(fit_and_score)(train_index, test_index)
                    for (train_index, test_index) in kfold.split(data_pd))
    out = Parallel(n_jobs=n_jobs, pre_dispatch=2*n_jobs)(delayed_list)
    print("testing error...", np.mean(np.array(out)))

print("#################################### run code mySVD regular ####################################")
import time
start_time = time.time()


# read data
data_pd = pd.read_csv('data_train.csv')

#split data
train_size = 0.75
train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)
train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)
val_users, val_movies, val_predictions = extract_users_items_predictions(test_pd)
(ur, ir) = create_adjacency_lists(train_users, train_movies, train_predictions)
bu = np.zeros(len(ur.keys())+1)
bi = np.zeros(len(ir.keys())+1)
#(bu, bi, pu, qi, global_mean) = fit_model_svd(bu, bi, ur, ir, train_users, train_movies, train_predictions, 1)
#print('#############################   start predictions... #############################')
#predictions = predict_svd(ur, ir, train_users, train_movies, val_users, val_movies, bu, bi, pu, qi, global_mean)
#error = mean_squared_error(val_predictions, predictions)**(1/2)

#cross_validate(data_pd, 5)

print("--- %s seconds ---" % (time.time() - start_time))

print('#############################   done! #############################')