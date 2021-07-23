from utils import data_processing
from models import matrix_factorization
""" 
    Example file to show how to use the AlgoBase functions 
    First, run: 
     export PYTHONPATH=/path-to-project/collaborative-filtering:$PYTHONPATH
     e.g.
     export PYTHONPATH=/Users/veronique/Documents/ETHZ/Master/Semester2/CIL/Project/collaborative-filtering:$PYTHONPATH
     
    or run "auxiliary/init_local.sh

"""



# Read the data from the file:
data_pd = data_processing.read_data()

svd = matrix_factorization.SVD(k_singular_values=2)

# cross_validation, can also set number of folds (default is 5), and randomstate (default is 42)
rmses = svd.cross_validate(data_pd)
print("RMSES:", rmses)

# we can also simply fit and predict
users, movies, predictions = data_processing.extract_users_items_predictions(data_pd)
svd.fit(users, movies, predictions)

pred_users, pred_movies = data_processing.get_users_movies_to_predict()
predictions = svd.predict(pred_users, pred_movies)
print("predictions: ", predictions[:5])

# alterantively we can predict and directly generate the submission file:
svd.fit(users, movies, predictions)
svd.predict_for_submission(name="submission_name")
