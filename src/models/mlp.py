import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from .algobase import AlgoBase
from utils.fitting import train
from ray import tune


class MLP(AlgoBase):

    class Sigmoid(nn.Module):
        def forward(self, x):
            return torch.sigmoid(x) * 4 + 1

    class Tanh(nn.Module):
        def forward(self, x):
            return torch.tanh(x) * 2 + 3

    class Model(nn.Module):
        def __init__(self, user_embedding, movie_embedding, d, act):
            super().__init__()
            self.embedding_users = nn.Embedding.from_pretrained(user_embedding, freeze=False)
            self.embedding_movies = nn.Embedding.from_pretrained(movie_embedding, freeze=False)
            # self.bias_users = nn.Parameter(torch.tensor(user_bias, dtype=torch.float32), requires_grad=True)
            # self.bias_movies = nn.Parameter(torch.tensor(movie_bias, dtype=torch.float32), requires_grad=True)
            self.mlp = nn.Sequential(
                # nn.Linear(in_features=user_embedding.shape[1] + movie_embedding.shape[1] + 2, out_features=16),
                nn.Dropout(d[0]),
                nn.Linear(in_features=self.embedding_users.embedding_dim + self.embedding_movies.embedding_dim, out_features=16),
                nn.ReLU(),
                nn.Dropout(d[1]),
                nn.Linear(in_features=16, out_features=4),
                nn.ReLU(),
                nn.Dropout(d[2]),
                nn.Linear(in_features=4, out_features=1),
                act,
            )

        def forward(self, users, movies):
            users_embedding = self.embedding_users(users)
            movies_embedding = self.embedding_movies(movies)
            concat_embedding = torch.cat([
                users_embedding,
                # self.bias_users[users].view(users.shape[0], 1),
                movies_embedding,
                # self.bias_movies[movies].view(users.shape[0], 1),
            ], dim=1)
            return torch.squeeze(self.mlp(concat_embedding))

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

    def make_model(self, dropout=[0, 0, 0], act=nn.Identity()):
        return self.Model(
            torch.tensor(self.user_embedding, dtype=torch.float32, device=self.device),
            torch.tensor(self.movie_embedding, dtype=torch.float32, device=self.device),
            dropout,
            act,
            # self.user_bias,
            # self.movie_bias,
        ).to(self.device)

    def fit(self, train_users, train_movies, train_predictions, test_users=None, test_movies=None, test_predictions=None):
        return train(
            train_users,
            train_movies,
            train_predictions,
            test_users,
            test_movies,
            test_predictions,
            device=self.device,
            model=self.model,
            optimizer=optim.Adam(self.model.parameters(), lr=self.learning_rate),
            # optimizer=optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.5, dampening=0.5),
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
        )

    def tune_params(self, train_users, train_movies, train_predictions, test_users, test_movies, test_predictions):
        def do_train(config):
            for i in range(5):
                m = self.make_model(config['dropout'], config['activation'])
                rmse, epochs = train(
                    train_users,
                    train_movies,
                    train_predictions,
                    test_users,
                    test_movies,
                    test_predictions,
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
