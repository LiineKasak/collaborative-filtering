import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models.gmf import GMF
from models.mlp import MLP
from models.algobase import AlgoBase
from utils.fitting import train
from utils.dataset import DatasetWrapper


class NCF(AlgoBase):
    """ Neural Collaborative Filtering Base"""

    class Model(nn.Module):
        def __init__(self, gmf, mlp):
            super().__init__()
            self.gmf = gmf
            self.mlp = mlp
            self.alpha = nn.Parameter(torch.rand(1, requires_grad=True))

        def forward(self, users, movies):
            return self.alpha * self.gmf(users, movies) + (1 - self.alpha) * self.mlp(users, movies)

    @staticmethod
    def default_params():
        return argparse.Namespace(epochs=2, batch_size=256, learning_rate=0.01)

    def __init__(self, device, epochs=20, batch_size=128, learning_rate=1e-2):
        super().__init__()
        self.gmf = None
        self.mlp = None
        self.model = None
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, train_data: DatasetWrapper, test_data: DatasetWrapper = None):
        self.gmf = GMF(
            user_embedding=None,
            movie_embedding=None,
            user_bias=None,
            movie_bias=None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            device=self.device,
        )
        self.gmf.fit(train_data, test_data)

        self.mlp = MLP(
            user_embedding=None,
            movie_embedding=None,
            user_bias=None,
            movie_bias=None,
            num_epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            device=self.device,
        )
        self.mlp.fit(train_data, test_data)

        self.model = self.Model(self.gmf.model, self.mlp.model)
        return train(
            train_data,
            test_data,
            device=self.device,
            model=self.model,
            optimizer=optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=1),
            num_epochs=self.epochs,
            batch_size=self.batch_size,
        )

    def predict(self, users, movies):
        return self.model(
            torch.tensor(users, device=self.device),
            torch.tensor(movies, device=self.device),
        ).detach().cpu()

    def save(self, filename: str):
        torch.save(self.model, filename)
