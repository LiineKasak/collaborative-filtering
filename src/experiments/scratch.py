
from utils import data_processing, dataset
from models.matrix_factorization import SVD
from models.ncf import NCF
from models.ensemble import Bagging
from models.knn import KNN
""" 
    Run Cross-Validation for multiple approaches at the same time.

"""
# -------------
#
# Variables
#
# -------------
number_of_users = data_processing.get_number_of_users()
number_of_movies = data_processing.get_number_of_movies()


data_pd = data_processing.read_data()
data = dataset.DatasetWrapper(data_pd)

data.movies_per_user_representation()
#
knn = KNN()

output = knn.cross_validate(data_pd)
print(output)
# # rsmes = bag.cross_validate(data_pd)
# # print(rsmes)

