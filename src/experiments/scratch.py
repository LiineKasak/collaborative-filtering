from sklearn.model_selection import train_test_split
import numpy as np
from models.classifiers import BaysianClassifier, svdInputClassifier
from models.svd_sgd import SVD_SGD
from utils import data_processing, dataset, data_analysis
from models.matrix_factorization import SVD
from models.ncf import NCF
from models.ensemble import Bagging
from models.knn import KNN, KNNBag, KNNUserMovie, KNNSVD_Embeddings, KNNSVD_Biases, KNNImprovedSVDEmbeddings, \
    KNNFancyFeatures
import pandas as pd
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
# number_of_users = data_processing.get_number_of_users()
# number_of_movies = data_processing.get_number_of_movies()
#
# data_pd = data_processing.read_data()
# train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)
#
# train_data_wrapper = dataset.DatasetWrapper(train_pd)
# data_wrapper = dataset.DatasetWrapper(data_pd)
#
# test_users, test_movies, test_predictions = data_processing.extract_users_items_predictions(test_pd)
# train_data_wrapper.movies_per_user_representation()


model = KNNFancyFeatures()
epochs = 75
k = 12
# k = 100
directory_path = data_processing.get_project_directory()
storing_directory = f"{directory_path}/data/precomputed_svd_e{epochs}_k{k}"

train_pd = pd.read_csv(f"{storing_directory}/train_pd.csv")
test_pd = pd.read_csv(f"{storing_directory}/test_pd.csv")

train_dw = dataset.DatasetWrapper(train_pd)
test_dw = dataset.DatasetWrapper(test_pd)

test_users, test_movies, test_predictions = test_dw.users, test_dw.movies, test_dw.ratings

model.fit(train_dw)
pred = model.predict(test_users, test_movies)
rmse = data_processing.get_score(pred, test_predictions)
print(f"score: {rmse}")
pa = data_analysis.PredictionAnalyser(test_movies=test_movies, test_predictions=test_predictions,
                                      test_users=test_users, output_predictions=pred)

pa.analyze_prediction()

data_processing.create_validation_file(test_users, test_movies, pred, test_predictions)
da = data_analysis.DataAnalyzer(train_pd)
da.create_validation_histograms(pred, test_predictions)
da.create_validation_scatterplot(pred, test_predictions)
# clf.fit(datawrapper=data_wrapper)
# clf.predict_for_submission(name='histgradientboosting_svd12')
