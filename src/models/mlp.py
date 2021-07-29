import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.algobase import AlgoBase
from utils.dataset import DatasetWrapper
from utils.fitting import train
from ray import tune


class MLP(AlgoBase):
    """ Multi-Layer Perceptron """

    class Sigmoid(nn.Module):
        def forward(self, x):
            return torch.sigmoid(x) * 3 + 1

    class Tanh(nn.Module):
        def forward(self, x):
            return torch.tanh(x) * 2 + 3

    class Model(nn.Module):

        def __init__(
                self,
                user_embedding,
                freeze_user_embedding,
                movie_embedding,
                freeze_movie_embedding,
                user_bias,
                movie_bias,
                dropouts,
                hidden_activation,
                output_activation,
        ):
            super().__init__()
            self.embedding_users = nn.Embedding.from_pretrained(user_embedding, freeze=freeze_user_embedding)
            self.embedding_movies = nn.Embedding.from_pretrained(movie_embedding, freeze=freeze_movie_embedding)

            self.bias_users = nn.Parameter(torch.tensor(user_bias, dtype=torch.float32), requires_grad=False)
            self.bias_movies = nn.Parameter(torch.tensor(movie_bias, dtype=torch.float32), requires_grad=False)

            self.output_activation = output_activation

            in_features = self.embedding_users.embedding_dim + self.embedding_movies.embedding_dim
            self.mlp = nn.Sequential(
                nn.Dropout(dropouts[0]),
                nn.Linear(in_features=in_features, out_features=16),
                hidden_activation,
                nn.Dropout(dropouts[1]),
                nn.Linear(in_features=16, out_features=4),
                hidden_activation,
                nn.Dropout(dropouts[2]),
                nn.Linear(in_features=4, out_features=1),
            )

        def forward(self, users, movies):
            users_embedding = self.embedding_users(users)
            movies_embedding = self.embedding_movies(movies)
            concat_embedding = torch.cat([
                users_embedding,
                movies_embedding,
            ], dim=1)
            return self.output_activation(
                torch.squeeze(self.mlp(concat_embedding)) + self.bias_users[users] + self.bias_movies[movies]
            )

    @staticmethod
    def default_params():
        return argparse.Namespace(epochs=3, batch_size=64, learning_rate=0.001, device="cpu")

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
            user_embedding_dim=16,
            movie_embedding_dim=16,
    ):
        super().__init__()
        if user_embedding is None:
            self.user_embedding = torch.empty(self.number_of_users, user_embedding_dim)
            nn.init.normal_(self.user_embedding)
            self.freeze_user_embedding = False
        else:
            self.user_embedding = torch.tensor(user_embedding, dtype=torch.float32)
            self.freeze_user_embedding = True

        if movie_embedding is None:
            self.movie_embedding = torch.empty(self.number_of_movies, movie_embedding_dim)
            nn.init.normal_(self.movie_embedding)
            self.freeze_movie_embedding = False
        else:
            self.movie_embedding = torch.tensor(movie_embedding, dtype=torch.float32)
            self.freeze_movie_embedding = True

        if user_bias is None:
            self.user_bias = np.zeros(self.number_of_users)
        else:
            self.user_bias = user_bias

        if movie_bias is None:
            self.movie_bias = np.zeros(self.number_of_movies)
        else:
            self.movie_bias = movie_bias

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.model = self.make_model()

    def make_model(
            self,
            dropout=(0, 0, 0),
            hidden_activation=nn.ReLU(),
            output_activation=Tanh(),
    ):
        return self.Model(
            self.user_embedding,
            self.freeze_user_embedding,
            self.movie_embedding,
            self.freeze_movie_embedding,
            self.user_bias,
            self.movie_bias,
            dropout,
            hidden_activation,
            output_activation,
        ).to(self.device)

    def fit(self, train_data: DatasetWrapper, test_data: DatasetWrapper = None):
        return train(
            train_data,
            test_data,
            device=self.device,
            model=self.model,
            optimizer=optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0),
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
        )

    def tune_params(self, train_data: DatasetWrapper, test_data: DatasetWrapper):
        def do_train(config):
            for i in range(5):
                m = self.make_model(config['dropout'], config['activation'])
                rmse, epochs = train(
                    train_data,
                    test_data,
                    device=self.device,
                    model=m,
                    optimizer=optim.Adam(m.parameters(), lr=config['lr'], weight_decay=config['weight_decay']),
                    num_epochs=self.num_epochs,
                    batch_size=config['batch_size'],
                    verbose=False,
                )
                tune.report(rmse=rmse)
        return tune.run(
            do_train,
            mode='min',
            config={
                'lr': tune.grid_search([0.1, 0.01, 0.001]),
                'batch_size': tune.grid_search([32, 64, 128, 256, 512]),
                'dropout': [
                    tune.grid_search([0, 0.1, 0.2]),
                    tune.grid_search([0, 0.1, 0.2]),
                    tune.grid_search([0, 0.1, 0.2]),
                ],
                'weight_decay': tune.grid_search([0, 0.1, 0.5, 0.9]),
                'activation': tune.grid_search([
                    nn.Identity(),
                    self.Sigmoid(),
                    self.Tanh(),
                ])
            },
        )

    def predict(self, users, movies):
        return self.model(
            torch.tensor(users, device=self.device),
            torch.tensor(movies, device=self.device),
        ).detach().cpu()

    def save(self, filename: str):
        torch.save(self.model, filename)
