import comet_ml
import numpy as np
from utils import dataset
from utils import data_processing
from models.knn import KNN, KNNBag

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

# -------------
#
# APPROACHES
#
# -------------
knn5 = KNN(n_neighbors=100, metric='cosine', method_name='KNN 5 cosine')
knn6 = KNN(n_neighbors=100, metric='euclidean', method_name='KNN 5 euclidean')
knn3 = KNN(n_neighbors=100, metric='manhattan', method_name='KNN 5 manhattan')
knn4 = KNN(n_neighbors=100, metric='chebyshev',method_name='KNN 5 chebyshev')

knnbag = KNNBag()
# approaches = [knn3, knn4, knn5, knn6]
approaches = [knnbag, knn3, knn4, knn5, knn6]

# Read the data from the file:
data_pd = data_processing.read_data()

# cross validate all approaches
for approach in approaches:
    print("================================")
    print(approach.method_name)
    print("================================\n")

    rmse = approach.cross_validate(data_pd, folds=5)
    print("RMSE ", np.mean(rmse))


knnbag.fit(data_wrapper=dataset.DatasetWrapper(data_pd))
knnbag.predict_for_submission("knnbag")

