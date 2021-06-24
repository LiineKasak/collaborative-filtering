from sklearn.model_selection import train_test_split

from utils import data_processing, dataset, prediction_analysis
from models.matrix_factorization import SVD
from models.ncf import NCF
from models.ensemble import Bagging
from models.knn import KNN, KNNBag, KNNUserMovie
""" 
    Run Cross-Validation for multiple approaches at the same time.

"""
train_size = 0.9
# -------------
#
# Variables
#
# -------------
number_of_users = data_processing.get_number_of_users()
number_of_movies = data_processing.get_number_of_movies()


data_pd = data_processing.read_data()
train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)

train_data_wrapper = dataset.DatasetWrapper(train_pd)

test_users, test_movies, test_predictions = data_processing.extract_users_items_predictions(test_pd)
train_data_wrapper.movies_per_user_representation()


knn5 = KNN(n_neighbors=500, metric='cosine', method_name='KNN 5 cosine')
knn6 = KNN(n_neighbors=500, metric='euclidean', method_name='KNN 5 euclidean')
knn3 = KNN(n_neighbors=500, metric='manhattan', method_name='KNN 5 manhattan')
knn4 = KNN(n_neighbors=500, metric='chebyshev',method_name='KNN 5 chebyshev')


knnBag = KNNBag()

knn_new = KNNUserMovie(n_user_neighbors=200, n_movie_neighbors=20)
# approaches = [knn5, knn6, knn3, knn4]
approaches = [knn_new]
for knn in approaches:
    print("================================")
    print(knn.method_name)
    print("================================\n")
    knn.fit(train_data_wrapper)
    pred = knn.predict(test_users, test_movies)
    pa = prediction_analysis.PredictionAnalyser(test_users, test_movies, test_predictions, pred, knn=knn)
    pa.analyze_prediction()



# # rsmes = bag.cross_validate(data_pd)
# # print(rsmes)

