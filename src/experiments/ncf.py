from argparse import ArgumentParser
from models.ncf import NCF
from utils.experiment import run_experiment
import torch


def parser_setup(parser: ArgumentParser):
    parser.add_argument('--gmf', type=str, help='GMF state dict')
    parser.add_argument('--mlp', type=str, help='MLP state dict')


def model_factory(args, device):
    gmf = torch.load(args.gmf)
    gmf.eval()

    mlp = torch.load(args.mlp)
    mlp.eval()

    return NCF(gmf, mlp, device, epochs=args.epochs, batch_size=128, learning_rate=args.lr)


run_experiment(
    'NCF',
    parser_setup,
    model_factory,
)
