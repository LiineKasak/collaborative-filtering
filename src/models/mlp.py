import torch
import torch.nn as nn
import torch.optim as optim
from .algobase import AlgoBase
from utils.fitting import train


class MLP(AlgoBase):

    class Model(nn.Module):
        def __init__(self, user_embedding, movie_embedding, user_bias, movie_bias):
            super().__init__()
            self.embedding_users = nn.Embedding.from_pretrained(user_embedding, freeze=False)
            self.embedding_movies = nn.Embedding.from_pretrained(movie_embedding, freeze=False)
            # self.bias_users = nn.Parameter(torch.tensor(user_bias, dtype=torch.float32), requires_grad=True)
            # self.bias_movies = nn.Parameter(torch.tensor(movie_bias, dtype=torch.float32), requires_grad=True)
            self.mlp = nn.Sequential(
                # nn.Linear(in_features=user_embedding.shape[1] + movie_embedding.shape[1] + 2, out_features=16),
                nn.Linear(in_features=self.embedding_users.embedding_dim + self.embedding_movies.embedding_dim, out_features=16),
                nn.ReLU(),
                nn.Linear(in_features=16, out_features=4),
                nn.ReLU(),
                nn.Linear(in_features=4, out_features=1),
                # nn.Sigmoid(),
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
            return torch.squeeze(self.mlp(concat_embedding)) #* 4 + 1

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
        self.model = self.Model(
            torch.tensor(user_embedding, dtype=torch.float32, device=device),
            torch.tensor(movie_embedding, dtype=torch.float32, device=device),
            user_bias,
            movie_bias,
        ).to(device)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

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
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
        )

    def predict(self, users, movies):
        return self.model(users, movies)

    def serialize(self, filename: str):
        torch.save(self.model.state_dict(), filename)
