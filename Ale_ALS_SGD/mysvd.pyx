import pandas as pd
cimport numpy as np
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

def calc_u_means(ur):
    u_means = {}
    for key in ur.keys():
        sum_r = 0
        for _,r in ur[key]:
            sum_r += r
        u_means[key] = sum_r/max(1,len(ur[key]))
    return u_means

def calc_u_vars(ur, u_means):
    u_vars = {}
    for key in ur.keys():
        sum_r = 0
        for _,r in ur[key]:
            sum_r += (r-u_means[key])**2
        val = sum_r/max(1,len(ur[key]))
        u_vars[key] = (val)**(1.0/2.0)
    return u_vars

def calc_i_means(ir):
    i_means = {}
    for key in ir.keys():
        sum_r = 0
        for _,r in ir[key]:
            sum_r += r
        i_means[key] = sum_r/max(1,len(ir[key]))
    return i_means

def mean_u_normalize(u_means, u_vars, train_users, train_ratings):
    mean_train_ratings = []
    for i in range(len(train_ratings)):
        u_id = train_users[i]
        if(u_vars[u_id] != 0):
            mean_train_ratings.append((train_ratings[i] - u_means[u_id])/u_vars[u_id])
        else:
            mean_train_ratings.append(0.0)
    return mean_train_ratings

def mean_u_denormalize(u_means, u_vars, test_users, test_ratings):
    mean_test_ratings = []
    for i in range(len(test_ratings)):
        u_id = test_users[i]
        mean_test_ratings.append((test_ratings[i] * u_vars[u_id]) + u_means[u_id])
    return mean_test_ratings

def mean_i_normalize(i_means, train_items, train_ratings):
    mean_train_ratings = []
    for i in range(len(train_ratings)):
        i_id = train_items[i]
        mean_train_ratings.append((train_ratings[i] - i_means[i_id]))
    return mean_train_ratings

def mean_i_denormalize(i_means, test_items, test_ratings):
    mean_test_ratings = []
    for i in range(len(test_ratings)):
        i_id = test_items[i]
        mean_test_ratings.append((test_ratings[i] + i_means[i_id]))
    return mean_test_ratings


def fit_model_baseline_als(ur, ir, train_user_ids, train_movie_ids, train_ratings, num_epochs):
    cdef double global_average = np.mean(train_ratings)
    cdef double reg_u = 15.0
    cdef double reg_i = 10.0

    cdef np.ndarray[np.double_t] bi = np.zeros(len(ir.keys())+1)
    cdef np.ndarray[np.double_t] bu = np.zeros(len(ur.keys())+1)

    cdef double sum_i
    cdef double sum_u
    cdef double rating
    cdef int user_idx
    cdef int item_idx

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
def fit_model_svd(mybu, mybi, ur, ir, train_user_ids, train_movie_ids, train_ratings, num_epochs):
    cdef double global_average = np.mean(train_ratings)
    cdef double learning_rate = 0.001
    #cdef double learning_rate = 0.005
    cdef double reg = 0.02
    cdef double reg_b = 0.005
    cdef double reg_pq = 0.06
    #reg_b = reg
    #reg_pq = reg
    cdef int nr_factors = 2
    cdef int nr_users = len(ur.keys())+1
    cdef int nr_items = len(ir.keys())+1
    cdef double init_mean = 0.0
    cdef double init_std = 0.1
    
    cdef np.ndarray[np.double_t] bu
    cdef np.ndarray[np.double_t] bi

    cdef np.ndarray[np.double_t, ndim=2] pu
    cdef np.ndarray[np.double_t, ndim=2] qi

    bu = mybu
    bi = mybi

    pu = np.random.normal(init_mean, init_std, size=(nr_factors, nr_users))
    qi = np.random.normal(init_mean, init_std, size=(nr_factors, nr_items))

    cdef int u_id, m_id, f
    cdef double r_i, error, dot, puf, qif

    #für was bruch ich ur???

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
    
    return (bu, bi, pu, qi, global_average)

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

def predict_bl(ur, ir, train_user_ids, train_movie_ids, test_user_ids, test_movie_ids, bu, bi, global_mean):
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

def fit_and_score(train_index, test_index):
    train_users, train_movies, train_predictions = extract_users_items_predictions(data_pd.iloc[train_index])
    val_users, val_movies, val_predictions = extract_users_items_predictions(data_pd.iloc[test_index])
    (ur, ir) = create_adjacency_lists(train_users, train_movies, train_predictions)
    (ur_val, ir_val) = create_adjacency_lists(val_users, val_movies, val_predictions)
    print('####################### with u normalization ########################')
    u_means = calc_u_means(ur)
    u_vars = calc_u_vars(ur, u_means)
    train_predictions = mean_u_normalize(u_means, u_vars, train_users, train_predictions)
    u_means_val = calc_u_means(ur_val)
    u_vars_val = calc_u_vars(ur_val, u_means_val)
    val_predictions = mean_u_normalize(u_means_val, u_vars_val, val_users, val_predictions)
    

    (ur, ir) = create_adjacency_lists(train_users, train_movies, train_predictions)
    #cdef np.ndarray[np.double_t] bu
    #cdef np.ndarray[np.double_t] bi
    #bu = np.zeros(len(ur.keys())+1)
    #bi = np.zeros(len(ir.keys())+1)
    print('#############################   starting fit 1 epochs... #############################')
    (bu, bi, global_mean) = fit_model_baseline_als(ur, ir, train_users, train_movies, train_predictions, 1)
    (bu, bi, pu, qi, global_mean) = fit_model_svd(bu, bi, ur, ir, train_users, train_movies, train_predictions, 100)
    print('#############################   start predictions... #############################')
    predictions = predict_svd(ur, ir, train_users, train_movies, val_users, val_movies, bu, bi, pu, qi, global_mean)
    #predictions = predict_bl(ur, ir, train_users, train_movies, val_users, val_movies, bu, bi, global_mean)
    cdef double error
    for i in range(len(predictions)):
        error += (predictions[i] - val_predictions[i])**2
    error = (error/len(predictions))**(1.0/2.0)
    print("rmse ", error)
    return error

def cross_validate(data_pd, nr_folds):
    kfold = KFold(n_splits=nr_folds, shuffle=True, random_state=41)

    n_jobs = 1
    delayed_list = (delayed(fit_and_score)(train_index, test_index)
                    for (train_index, test_index) in kfold.split(data_pd))
    out = Parallel(n_jobs=n_jobs, pre_dispatch=2*n_jobs)(delayed_list)
    print("testing error...", np.mean(np.array(out)))

print("#################################### run code mySVD-C ####################################")
import time
start_time = time.time()

# read data
data_pd = pd.read_csv('data_train.csv')

cross_validate(data_pd, 5)


print("--- %s seconds ---" % (time.time() - start_time))

print('#############################   done! #############################')