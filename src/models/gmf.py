import torch
import torch.nn as nn
import torch.optim as optim
from .algobase import AlgoBase
from utils.fitting import train


class GMF(AlgoBase):

    class Model(nn.Module):
        def __init__(self, user_embedding, movie_embedding, user_bias, movie_bias):
            super().__init__()
            self.embedding_users = nn.Embedding.from_pretrained(user_embedding, freeze=False)
            self.embedding_movies = nn.Embedding.from_pretrained(movie_embedding, freeze=False)
            self.bias_users = nn.Parameter(torch.tensor(user_bias, dtype=torch.float32), requires_grad=True)
            self.bias_movies = nn.Parameter(torch.tensor(movie_bias, dtype=torch.float32), requires_grad=True)
            self.weights = nn.Sequential(
                nn.Linear(in_features=user_embedding.shape[1] + 2, out_features=1),
            )

        def forward(self, users, movies):
            users_embedding = self.embedding_users(users)
            movies_embedding = self.embedding_movies(movies)
            product = torch.mul(users_embedding, movies_embedding)
            return torch.squeeze(self.weights(
                torch.cat([
                    product,
                    self.bias_users[users].view(users.shape[0], 1),
                    self.bias_movies[movies].view(movies.shape[0], 1),
                ], dim=1)
            ))

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
            optimizer=optim.SGD(self.model.parameters(), lr=self.learning_rate),
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
        )

    def predict(self, users, movies):
        return self.model(
            torch.tensor(users, device=self.device),
            torch.tensor(movies, device=self.device),
        ).detach().cpu()

    def serialize(self, filename: str):
        torch.save(self.model.state_dict(), filename)