import comet_ml
import numpy as np

from models.svd_sgd import SVD_SGD
from utils import dataset
from utils import data_processing
from models.knn import KNN, KNNBag, KNNSVD_Embeddings, KNNSVD_Biases, KNNImprovedSVDEmbeddings
from sklearn.model_selection import train_test_split

""" 
    Run Cross-Validation for multiple approaches at the same time.
    
"""

TRACK = False
# -------------
#
# Variables
#
# -------------
number_of_users = data_processing.get_number_of_users()
number_of_movies = data_processing.get_number_of_movies()

train_size = 0.9


data_pd = data_processing.read_data()
train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)

train_data_wrapper = dataset.DatasetWrapper(train_pd)

test_users, test_movies, test_predictions = data_processing.extract_users_items_predictions(test_pd)


# -------------
#
# APPROACHES
#
# -------------
# knnbag = KNNBag(method_name='knn_bag')
# knn_normal = KNN(n_neighbors=100, metric='cosine', method_name='KNN100_cosine')
# knn_emb = KNNSVD_Embeddings(method_name="KNNSVd_Embeddings")
# knn_decompose = KNNSVD_Biases(method_name="KNN_SVD_decompose_reconstruction_matrix")
#
# knn_improved_embeddings = KNNImprovedSVDEmbeddings()

svd_sgd = SVD_SGD(epochs=10, k_singular_values=5)
# approaches = [knn3, knn4, knn5, knn6]
# approaches = [knn_normal, knn_emb, knn_decompose, knnbag, knn_improved_embeddings]
approaches = [svd_sgd]
# # # Read the data from the file:
data_pd = data_processing.read_data()
#
# cross validate all approaches
for approach in approaches:
    print("================================================================\n")
    print(approach.method_name)

    approach.fit(train_data_wrapper)
    pred = approach.predict(test_users, test_movies)


    data_processing.create_validation_file(test_users, test_movies, pred, test_predictions,
                                           name=approach.method_name+'_validation')

    rmses = approach.cross_validate(data_pd, folds=5)
    print(f"RMSE: {np.mean(rmses)}")



# print("Now running crossvalidation on the liine thing")
# rsmes = knn_improved_embeddings.cross_validate(data_pd)
# print(f"RSME: {np.mean(rsmes)}")
