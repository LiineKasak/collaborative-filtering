import torch
import torch.nn as nn
import torch.optim as optim
from algobase import AlgoBase
from utils.dataset import DatasetWrapper
from utils.fitting import train
from ray import tune


class GMF(AlgoBase):
    class Model(nn.Module):
        def __init__(self, user_embedding, movie_embedding, user_bias, movie_bias):
            super().__init__()
            self.embedding_users = nn.Embedding.from_pretrained(user_embedding, freeze=False)
            self.embedding_movies = nn.Embedding.from_pretrained(movie_embedding, freeze=False)
            self.bias_users = nn.Parameter(torch.tensor(user_bias, dtype=torch.float32), requires_grad=False)
            self.bias_movies = nn.Parameter(torch.tensor(movie_bias, dtype=torch.float32), requires_grad=False)
            self.weights = nn.Linear(in_features=user_embedding.shape[1], out_features=1)

        def forward(self, users, movies):
            users_embedding = self.embedding_users(users)
            movies_embedding = self.embedding_movies(movies)
            product = torch.mul(users_embedding, movies_embedding)
            return torch.squeeze(self.weights(product)) + self.bias_users[users] + self.bias_movies[movies]

    def __init__(
            self,
            user_embedding,
            movie_embedding,
            user_bias,
            movie_bias,
            num_epochs,
            batch_size,
            learning_rate,
            device,
    ):
        super().__init__()
        self.user_embedding = user_embedding
        self.movie_embedding = movie_embedding
        self.user_bias = user_bias
        self.movie_bias = movie_bias
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.model = self.make_model()

    def make_model(self):
        return self.Model(
            torch.tensor(self.user_embedding, dtype=torch.float32, device=self.device),
            torch.tensor(self.movie_embedding, dtype=torch.float32, device=self.device),
            self.user_bias,
            self.movie_bias,
        ).to(self.device)

    def fit(self, train_data: DatasetWrapper, test_data: DatasetWrapper = None):
        return train(
            train_data,
            test_data,
            device=self.device,
            model=self.model,
            optimizer=optim.SGD(self.model.parameters(), lr=self.learning_rate),
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
        )

    def tune_params(self, train_data: DatasetWrapper, test_data: DatasetWrapper = None):
        def do_train(config, train_data: DatasetWrapper, test_data: DatasetWrapper = None):
            for i in range(5):
                m = self.make_model()
                rmse, epochs = train(
                    train_data,
                    test_data,
                    device=self.device,
                    model=m,
                    optimizer=optim.SGD(m.parameters(), lr=config['lr']),
                    num_epochs=self.num_epochs,
                    batch_size=config['batch_size'],
                )
                tune.report(rmse=rmse)

        return tune.run(
            tune.with_parameters(do_train, train_data=train_data, test_data=test_data),
            mode='min',
            config={
                'lr': tune.grid_search([0.1, 0.01, 0.001, 0.0001]),
                'batch_size': tune.grid_search([32, 64, 128, 256, 512]),
            },
        )

    def predict(self, users, movies):
        return self.model(
            torch.tensor(users, device=self.device),
            torch.tensor(movies, device=self.device),
        ).detach().cpu()

    def save(self, filename: str):
        torch.save(self.model, filename)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from utils.experiment import run_experiment
    import pickle


    def parser_setup(parser: ArgumentParser):
        parser.add_argument('--svd', type=str, help='SVD pickle')


    def model_factory(args, device):
        svd = pickle.load(open(args.svd, 'rb'))
        return GMF(
            svd.pu,
            svd.qi,
            svd.bu,
            svd.bi,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
        )


    run_experiment(
        'GMF',
        parser_setup,
        model_factory,
    )
