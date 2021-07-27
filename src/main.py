import argparse
import numpy as np
from models.svd_sgd import SVD_SGD
from models.aumf import AuMF
from models.log_reg import LogisticRegression
from models.knn import KNNImprovedSVDEmbeddings
from utils import data_processing
from utils.dataset import DatasetWrapper
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Train (, validate and save) a collaborative filtering model.')
parser.add_argument('model', type=str, help='selected model', choices=['aumf', 'svd_sgd', 'log_reg', 'knn'])  # TODO
parser.add_argument('--train_split', '-split', type=float, default=0.8,
                    help='Train split size, 0 < split <= 1. 1 for no validation set. (Default: 0.9)')
parser.add_argument('--submission', type=str, help='Submission file name if should be created.')
parser.add_argument('--cross_validate', '-cv', action='store_true', help='Flag for cross-validation.')
parser.add_argument('--verbal', '-v', type=bool)

# often used arguments
parser.add_argument('--epochs', '-e', type=int)
parser.add_argument('--learning_rate', '-lr', type=float)
parser.add_argument('--regularization', '-r', type=float)
parser.add_argument('--batch_size', '-b', type=int)
parser.add_argument('--device', '-d', type=str)

# model-specific parameters
parser.add_argument('--k_singular_values', '-k', type=int)
parser.add_argument('--enable_bias', '-bias', type=bool)
parser.add_argument('--n_neighbors', '-n', type=int)    # KNN neighbors
# TODO add other custom params from different models

args = parser.parse_args()
print(args)


def get_params(params, default_params):
    for arg, value in params.__dict__.items():
        if value is None and hasattr(default_params, arg):
            setattr(params, arg, getattr(default_params, arg))
    return params



def get_model(model : str):
    if model == 'svd_sgd':
        return SVD_SGD(get_params(args, SVD_SGD.default_params()))
    elif model == 'aumf':
        return AuMF(get_params(args, AuMF.default_params()))
    elif model == 'knn':
        return KNNImprovedSVDEmbeddings(get_params(args, KNNImprovedSVDEmbeddings.default_params()))
    elif model == 'log_reg':
        return LogisticRegression(get_params(args, LogisticRegression.default_params()))
    # TODO: add models

    else:
        print("This model does not exist.")


model = get_model(args.model)
data_pd = data_processing.read_data()
data = DatasetWrapper(data_pd)

if args.cross_validate:
    nr_folds = int(1. / (1. - args.train_split))
    rmses = model.cross_validate(data_pd, nr_folds)
    print(f'Cross-validation RMSEs: {rmses}')
    print(f'Average RMSE: {np.mean(np.array(rmses))}')
elif 0 < args.train_split < 1:
    train_pd, test_pd = train_test_split(data_pd, train_size=0.9, random_state=42)
    train_data = DatasetWrapper(train_pd)
    model.fit(train_data)

    users_valid, movies_valid, predictions_true = data_processing.extract_users_items_predictions(test_pd)
    predictions_valid = model.predict(users_valid, movies_valid)
    rmse = data_processing.rmse(predictions_valid, predictions_true)
    print("Validation RMSE: ", rmse)
elif args.train_split == 1:
    model.fit(data)

    if args.submission is not None:
        model.predict_for_submission(args.submission)
