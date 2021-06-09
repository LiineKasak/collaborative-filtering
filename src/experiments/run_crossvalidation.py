from utils import data_processing
from models.svd import SVD
from models.ncf import NCF

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

# -------------
#
# APPROACHES
#
# -------------
svd = SVD(k_singular_values=2)
ncf = NCF(
    number_of_users=number_of_users,
    number_of_movies=number_of_movies,
    embedding_size=16,
)

approaches = [svd, ncf]

# Read the data from the file:
data_pd = data_processing.read_data()


# cross validate all approaches
for approach in approaches:
    rmses = approach.cross_validate(data_pd, folds=5)
    print(approach.__class__.__name__, ": ")
    print(rmses)
