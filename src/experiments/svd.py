from argparse import ArgumentParser
from utils.experiment import run_experiment
from models.svd_sgd import SVD_SGD


def parser_setup(parser: ArgumentParser):
    parser.add_argument('-k', type=int, help='number of singular values')
    parser.add_argument('--bias', dest='bias', action='store_true')
    parser.add_argument('--no-bias', dest='bias', action='store_false')
    parser.set_defaults(bias=True)


def model_factory(args, device):
    return SVD_SGD(verbal=True, k_singular_values=args.k, enable_bias=args.bias, epochs=args.epochs)


run_experiment(
    'SVG-SGD',
    parser_setup,
    model_factory,
)
