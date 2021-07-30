import argparse
import numpy as np
import torch
from models.svd_sgd import SVD_SGD
from models.aumf import AuMF
from models.gmf import GMF
from models.mlp import MLP
from models.ncf import NCF
from models.log_reg import LogisticRegression
from models.knn import KNNImprovedSVDEmbeddings
from models.autoencoder.deep_autoencoder import DeepAutoEncoder
from models.autoencoder.variational_autoencoder import VAE
from models.autoencoder.collaborative_denoising_autoencoder import CDAE
from models.svt import SVT
from models.svd_als_sgd_hybrid import SVD_ALS_SGD
from models.svt_init_svd_als_sgd_hybrid import SVT_INIT_SVD_ALS_SGD
from utils import data_processing
from utils.dataset import DatasetWrapper
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Train (, validate and save) a collaborative filtering model.')
parser.add_argument('model', type=str, help='selected model',
                    choices=['aumf', 'svd', 'svd_sgd', 'log_reg', 'knn', 'gmf', 'mlp', 'ncf', 'vae', 'cdae', 'ae',
                             'svt', 'svt_hybrid', 'svt_init_hybrid'])
parser.add_argument('--mode', '-m', type=str, choices=['val', 'cv', 'submit'], default='val',
                    help='mode: validate, cross-validate (cv) or train for submission (default: \'val\').')
parser.add_argument('--train_split', '-split', type=float, default=0.9,
                    help='Train portion of dataset if mode=\'val\' (default: 0.9). Must satisfy 0 < train_split < 1.')
parser.add_argument('--folds', '-f', type=int, default=5, help='Number of folds if mode=\'cv\' (default: 5).')
parser.add_argument('--submission', type=str, default='submission',
                    help='Submission file name if mode=\'submit\' (default: \'submission\').')
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
parser.add_argument('--n_neighbors', '-n', type=int)  # KNN neighbors
parser.add_argument('--shrink_val', '-s', type=float)

args = parser.parse_args()


def get_params(params, default_params):
    for arg, value in params.__dict__.items():
        if value is None and hasattr(default_params, arg):
            setattr(params, arg, getattr(default_params, arg))
    print('Running with parameters: ', params.__dict__.items())
    return params


def get_device(name):
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_model(model: str):
    if model == 'svd_sgd':
        return SVD_SGD(get_params(args, SVD_SGD.default_params()))
    elif model == 'aumf':
        params = get_params(args, AuMF.default_params())
        use_precomputed = params.mode == 'cv'
        return AuMF(params=params, use_pretrained_svd=use_precomputed)
    elif model == 'knn':
        return KNNImprovedSVDEmbeddings(get_params(args, KNNImprovedSVDEmbeddings.default_params()))
    elif model == 'log_reg':
        return LogisticRegression(get_params(args, LogisticRegression.default_params()))
    elif model == 'gmf':
        params = get_params(args, GMF.default_params())
        device = get_device(params.device)
        return GMF(user_embedding=None,
                   movie_embedding=None,
                   user_bias=None,
                   movie_bias=None,
                   epochs=params.epochs,
                   batch_size=params.batch_size,
                   learning_rate=params.learning_rate,
                   device=device)
    elif model == 'mlp':
        params = get_params(args, GMF.default_params())
        device = get_device(params.device)
        return MLP(user_embedding=None,
                   movie_embedding=None,
                   user_bias=None,
                   movie_bias=None,
                   num_epochs=params.epochs,
                   batch_size=params.batch_size,
                   learning_rate=params.learning_rate,
                   device=device)
    elif model == 'ncf':
        params = get_params(args, NCF.default_params())
        device = get_device(params.device)
        return NCF(device=device,
                   epochs=params.epochs,
                   batch_size=params.batch_size,
                   learning_rate=params.learning_rate)
    elif model == 'vae':
        return VAE(get_params(args, VAE.default_params()))
    elif model == 'cdae':
        return CDAE(get_params(args, CDAE.default_params()))
    elif model == 'ae':
        return DeepAutoEncoder(get_params(args, DeepAutoEncoder.default_params()))
    elif model == 'svt':
        return SVT(get_params(args, SVT.default_params()))
    elif model == 'svt_hybrid':
        return SVD_ALS_SGD(get_params(args, SVD_ALS_SGD.default_params()))
    elif model == 'svt_init_hybrid':
        return SVT_INIT_SVD_ALS_SGD(get_params(args, SVT_INIT_SVD_ALS_SGD.default_params()))
    else:
        print("This model does not exist.")


model = get_model(args.model)
data_pd = data_processing.read_data()
data = DatasetWrapper(data_pd)

if args.mode == 'cv' and args.folds > 0:
    rmses = model.cross_validate(data_pd, args.folds)
    print(f'Cross-validation RMSEs: {rmses}')
    print(f'Average RMSE: {np.mean(np.array(rmses))}')

elif args.mode == 'val' and 0 < args.train_split < 1:
    train_pd, test_pd = train_test_split(data_pd, train_size=args.train_split, random_state=42)
    train_data = DatasetWrapper(train_pd)
    model.fit(train_data)

    users_valid, movies_valid, predictions_true = data_processing.extract_users_items_predictions(test_pd)
    predictions_valid = model.predict(users_valid, movies_valid)
    rmse = data_processing.rmse(predictions_valid, predictions_true)
    print("Validation RMSE: ", rmse)

elif args.mode == 'submit':
    model.fit(data)
    if args.submission is not None:
        model.predict_for_submission(args.submission)
