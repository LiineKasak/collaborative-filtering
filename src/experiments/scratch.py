
from utils import data_processing
from models.matrix_factorization import SVD
from models.ncf import NCF
from models.ensemble import Bagging
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

methods = [svd, ncf]

bag = Bagging(methods)

data_pd = data_processing.read_data()
rsmes = svd.cross_validate(data_pd=data_pd)


# rsmes = bag.cross_validate(data_pd)
print(rsmes)

