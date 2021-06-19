import comet_ml
import numpy as np
from utils import data_processing
from models.knn import KNN

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
knn5 = KNN(n_neighbors=5, method_name='KNN 5')
knn10 = KNN(n_neighbors=10, method_name='KNN 10')
knn50 = KNN(n_neighbors=50, method_name='KNN 50')
knn100 = KNN(n_neighbors=100, method_name='KNN 100')

approaches = [knn5, knn10, knn50, knn100]

# Read the data from the file:
data_pd = data_processing.read_data()

# cross validate all approaches
for approach in approaches:
    print("================================")
    print(approach.method_name)
    print("================================\n")

    rmse = approach.cross_validate(data_pd, folds=5)
    print("RMSE ", np.mean(rmse))

