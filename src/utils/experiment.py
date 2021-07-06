from argparse import ArgumentParser
from models.algobase import AlgoBase
from utils import data_processing
from sklearn.model_selection import train_test_split
from typing import Callable, Any
import torch


def run_experiment(
        name: str,
        parser_setup: Callable[[ArgumentParser], None],
        model_factory: Callable[[Any, torch.device], AlgoBase],
):
    parser = ArgumentParser(description=name)
    parser_setup(parser)
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--batch-size', type=int, dest='batch_size')
    parser.add_argument('--epochs', '-e', type=int, help='nubmer of epochs')
    parser.add_argument('--output', '-o', type=str, help='file to export')
    parser.add_argument('--no-validate', '--noval', dest='validate', action='store_false')
    parser.add_argument('--cross-validate', dest='crossval', action='store_true')
    parser.add_argument('--tune', dest='tune', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.set_defaults(
        validate=True,
        crossval=False,
        tune=False,
        submit=False,
    )
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.tune:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('Using device:', device)

    model = model_factory(args, device)

    data_pd = data_processing.read_data()
    if args.crossval:
        model.cross_validate(data_pd)
    else:
        users, movies, predictions = data_processing.extract_users_items_predictions(data_pd)
        if args.validate or args.tune:
            train_users, test_users, train_movies, test_movies, train_predictions, test_predictions = train_test_split(
                users,
                movies,
                predictions,
                train_size=0.9,
                random_state=42
            )
        else:
            train_users, train_movies, train_predictions = users, movies, predictions
            test_users, test_movies, test_predictions = None, None, None

        if args.tune:
            analysis = model.tune_params(
                train_users,
                train_movies,
                train_predictions,
                test_users,
                test_movies,
                test_predictions,
            )
            print("Best config: ", analysis.get_best_config(metric="rmse"))
        else:
            rmse, epochs = model.fit(
                train_users,
                train_movies,
                train_predictions,
                test_users,
                test_movies,
                test_predictions,
            )
            print(f'Best RMSE {rmse} after {epochs} epochs.')

    if args.output:
        print(f'Saving model in `{args.output}`.')
        model.save(args.output)

    if args.submit:
        model.predict_for_submission(f'{name}_sub')
