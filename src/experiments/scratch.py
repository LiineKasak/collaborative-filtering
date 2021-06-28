from sklearn.model_selection import train_test_split
import numpy as np

from models.svd_sgd import SVD_SGD
from utils import data_processing, dataset, prediction_analysis
from models.matrix_factorization import SVD
from models.ncf import NCF
from models.ensemble import Bagging
from models.knn import KNN, KNNBag, KNNUserMovie, KNNSVD_Embeddings, KNNSVD_Biases
from models.kmeans import KMeansRecommender
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

#
# knn5 = KNN(n_neighbors=500, metric='cosine', method_name='KNN 5 cosine')
# knn6 = KNN(n_neighbors=500, metric='euclidean', method_name='KNN 5 euclidean')
# knn3 = KNN(n_neighbors=500, metric='manhattan', method_name='KNN 5 manhattan')
# knn4 = KNN(n_neighbors=500, metric='chebyshev',method_name='KNN 5 chebyshev')
#
#
# knnBag = KNNBag()
#
# knn_new = KNNUserMovie(n_user_neighbors=200, n_movie_neighbors=20)
# knn_svd2_cosine_movie = KNNSVD_Embeddings(user_based=False, metric='cosine', n_neighbors=2,
#                                           method_name="movie based KNNSVD 2 cosine")
# knn_svd3_cosine_movie = KNNSVD_Embeddings(user_based=False, metric='cosine', n_neighbors=3,
#                                           method_name="movie based KNNSVD 3 cosine")
# knn_svd5_cosine_movie = KNNSVD_Embeddings(user_based=False, metric='cosine', n_neighbors=5,
#                                           method_name="movie based KNNSVD 5 cosine")
#
# knn_svd2_euclidean_movie = KNNSVD_Embeddings(user_based=False, metric='euclidean', n_neighbors=2,
#                                              method_name="movie based KNNSVD 2 euclidean")
# knn_svd3_euclidean_movie = KNNSVD_Embeddings(user_based=False, metric='euclidean', n_neighbors=3,
#                                              method_name="movie based KNNSVD 3 euclidean")
# knn_svd5_euclidean_movie = KNNSVD_Embeddings(user_based=False, metric='euclidean', n_neighbors=5,
#                                              method_name="movie based KNNSVD 5 euclidean")
#
# knn_svd2_cosine = KNNSVD_Embeddings(user_based=True, metric='cosine', n_neighbors=2,
#                                     method_name="user based KNNSVD 2 cosine")
# knn_svd3_cosine = KNNSVD_Embeddings(user_based=True, metric='cosine', n_neighbors=3,
#                                     method_name="user based KNNSVD 3 cosine")
# knn_svd5_cosine = KNNSVD_Embeddings(user_based=True, metric='cosine', n_neighbors=5,
#                                     method_name="user based KNNSVD 5 cosine")
#
# knn_svd2_euclidean = KNNSVD_Embeddings(user_based=True, metric='euclidean', n_neighbors=2,
#                                        method_name="user based KNNSVD 2 euclidean")
# knn_svd3_euclidean = KNNSVD_Embeddings(user_based=True, metric='euclidean', n_neighbors=3,
#                                        method_name="user based KNNSVD 3 euclidean")
# knn_svd5_euclidean = KNNSVD_Embeddings(user_based=True, metric='euclidean', n_neighbors=5,
#                                        method_name="user based KNNSVD 5 euclidean")
# # knn_svd50 = KNNSVD_Embeddings(n_neighbors=50, method_name="KNNSVD 50")
# knn_svd200 = KNNSVD_Embeddings(n_neighbors=200, method_name="KNNSVD 200")

# approaches = [knn5, knn6, knn3, knn4]
#approaches = [knn_svd2_cosine_movie,
              # knn_svd3_cosine_movie,
              # knn_svd5_cosine_movie,
              #
              # knn_svd2_euclidean_movie,
              # knn_svd3_euclidean_movie,
              # knn_svd5_euclidean_movie,
              #
              # knn_svd2_cosine,
              # knn_svd3_cosine,
              # knn_svd5_cosine,
              #
              # knn_svd2_euclidean,
              # knn_svd3_euclidean,
              # knn_svd5_euclidean
          #    ]



#sv: 1, neighbors:  2


# k = 10
# epochs = 2
# sgd = SVD_SGD(k_singular_values=k, epochs=epochs, verbal=True)
#
# rsmes = sgd.cross_validate(data_pd)
# print(rsmes)

svd = KNNSVD_Biases(k=10, n_neighbors=2)
# rsmes = svd.cross_validate(data_pd)

full_data_wrapper = dataset.DatasetWrapper(data_pd)

svd.fit(data_wrapper=full_data_wrapper)
svd.predict_for_submission("knn_svd")



best_score = 5
best_k = 0
best_n = 0
user_item = 'none'

# for n in [1, 2, 5, 10, 50]:
#     for k in range(1,12,2):
#         print("================================")
#         print("singular values: ", k, "neighbors: ", n)
#         print("================================\n")
#         knn = KNNSVD_Embeddings(k=k, user_based=True, metric='cosine', n_neighbors=n, method_name="user based KNNSVD 2 cosine")
#
#         rsmes = knn.cross_validate(data_pd)
#         score = np.mean(rsmes)
#         print("mean rsme: ", score)
#
#         if score < best_score:
#             best_score = score
#             best_k = k
#             best_n = n
#             user_item = 'user'

