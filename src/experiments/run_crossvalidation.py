import numpy as np
from utils import dataset
from utils import data_processing
from models.knn import KNNImprovedSVDEmbeddings
from models.classifiers import svdInputClassifier
""" 
    Run Cross-Validation to reproduce the results reported in the paper
    
"""
# -------------
# APPROACHES: All the approaches used as baseline comparisons
# -------------
knn = KNNImprovedSVDEmbeddings()
log_regression = svdInputClassifier()

approaches = [knn, log_regression]


# -------------
# Variables
# -------------
number_of_users = data_processing.get_number_of_users()
number_of_movies = data_processing.get_number_of_movies()
# Read the data from the file:
data_pd = data_processing.read_data()





# cross validate all approaches
for approach in approaches:
    print("================================")
    print(approach.method_name)
    print("================================\n")

    rmse = approach.cross_validate(data_pd, folds=5)
    print("RMSE ", np.mean(rmse))



