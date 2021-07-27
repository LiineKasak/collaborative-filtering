import argparse
import numpy as np
from models.svd_sgd import SVD_SGD
from utils import data_processing
from utils.dataset import DatasetWrapper
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Train (, validate and save) a collaborative filtering model.')
parser.add_argument('model', type=str, help='selected model', choices=['svd_sgd'])  # TODO
parser.add_argument('--train_split', '-split', type=float, default=0.9,
                    help='Train split size, 0 < split <= 1. 1 for no validation set. (Default: 0.9)')
parser.add_argument('--submission', type=str, help='Submission file name if should be created.')
parser.add_argument('--cross_validate', '-cv', action='store_true', help='Flag for cross-validation.')
parser.add_argument('--epochs', '-e', type=int)
parser.add_argument('--learning_rate', '-lr', type=float)
parser.add_argument('--regularization', '-r', type=float)
parser.add_argument('--k_singular_values', '-k', type=int)
parser.add_argument('--verbal', '-v', type=bool)
parser.add_argument('--enable_bias', '-bias', type=bool)
# TODO add other custom params from different models

args = parser.parse_args()
print(args)


def get_params(params, default_params):
    for arg, value in params.__dict__.items():
        if value is None and hasattr(default_params, arg):
            setattr(params, arg, getattr(default_params, arg))
    return params


model_dict = {
    'svd_sgd': SVD_SGD(get_params(args, SVD_SGD.default_params()))
    # TODO: add models
}

model = model_dict[args.model]
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
