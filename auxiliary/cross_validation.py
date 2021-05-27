
from comet_ml import Experiment
from auxiliary import data_processing
from sklearn.model_selection import train_test_split
from src.run_crossvalidation import max_iterations, num_eigenvalues, train_size
import numpy as np
import torch

def cross_validate_factorization(data_pd, fact, cv=5, comet_log=False):
    if comet_log:
        experiment = Experiment(
            api_key="rISpuwcLQoWU6qan4jRCAPy5s",
            project_name="cil-experiments",
            workspace="veroniquek",
        )
    rmses = []
    rs = 42
    for fold in range(cv):
        # split train and test set
        train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=rs)
        train_users, train_movies, train_predictions = data_processing.extract_users_items_predictions(train_pd)
        train_data, mask = data_processing.get_data_mask(train_users, train_movies, train_predictions)

        train_data = torch.from_numpy(train_data)
        train_data.float()


        # run on train set
        U, V = fact(train_data, max_iterations, num_eigenvalues)
        reconstructed_matrix = U @ V.t()

        # test on test set
        test_users, test_movies, test_predictions = data_processing.extract_users_items_predictions(test_pd)
        predictions = data_processing.extract_prediction_from_full_matrix(reconstructed_matrix, test_users, test_movies)

        rmse = data_processing.get_score(predictions,  test_predictions)
        rmses.append(rmse)
        rs += 1

    if comet_log:
        experiment.log_metrics(
            {
                "root_mean_squared_error": data_processing.get_score(predictions, test_predictions)
            }
        )


    print("Mean RMSE = ", np.mean(rmses))
