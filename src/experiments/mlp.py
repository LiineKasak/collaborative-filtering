from argparse import ArgumentParser
from utils.experiment import run_experiment
import pickle
from models.mlp import MLP


def parser_setup(parser: ArgumentParser):
    parser.add_argument('--svd', type=str, help='SVD pickle')


def model_factory(args, device):
    svd = pickle.load(open(args.svd, 'rb'))
    return MLP(svd.pu, svd.qi, svd.bu, svd.bi, num_epochs=args.epochs, batch_size=128, learning_rate=args.lr, device=device)


run_experiment(
    'MLP',
    parser_setup,
    model_factory,
)
