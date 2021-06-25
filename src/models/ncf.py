import torch
import torch.nn as nn
import torch.optim as optim
from .algobase import AlgoBase
from utils.fitting import train


class NCF(AlgoBase):

    class Model(nn.Module):
        def __init__(self, gmf, mlp):
            super().__init__()
            self.gmf = gmf
            self.mlp = mlp
            self.alpha = nn.Parameter(torch.rand(1, requires_grad=True))

        def forward(self, users, movies):
            return self.alpha * self.gmf(users, movies) + (1 - self.alpha) * self.mlp(users, movies)

    def __init__(self, gmf, mlp, device, epochs=20, batch_size=128, learning_rate=1e-2):
        super().__init__()
        self.device = device
        self.model = self.Model(gmf, mlp).to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

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
            num_epochs=self.epochs,
            batch_size=self.batch_size,
        )

    def predict(self, users, movies):
        return self.model(users, movies)

    def serialize(self, filename: str):
        torch.save(self.model.state_dict(), filename)
