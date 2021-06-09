import pandas as pd
import numpy as np
from utils import data_processing
from models.algobase import AlgoBase
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import pytorch_lightning as pl


def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)

class NCF(nn.Module, AlgoBase):
    def __init__(self, number_of_users, number_of_movies, embedding_size=16, batch_size=1024, num_epochs=10, learning_rate=10e-3, optimizer = None, track_to_comet=False, method_name=None, api_key="rISpuwcLQoWU6qan4jRCAPy5s", projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        super().__init__(track_to_comet=track_to_comet, method_name=method_name, api_key=api_key, projectname=projectname, workspace=workspace, tag=tag)
        AlgoBase.__init__(self)
        self.embedding_layer_users = nn.Embedding(number_of_users, embedding_size)
        self.embedding_layer_movies = nn.Embedding(number_of_movies, embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=2 * embedding_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),  # maybe predict per category?
            nn.ReLU()
        )

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, users, movies):
        users_embedding = self.embedding_layer_users(users)
        movies_embedding = self.embedding_layer_movies(movies)
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        return torch.squeeze(self.feed_forward(concat))

    def fit(self, users, movies, predictions):
        # Build Dataloaders
        train_dataloader = data_processing.create_dataloader(users, movies, predictions, batch_size=1024)

        step = 0
        for epoch in range(self.num_epochs):
            for users_batch, movies_batch, target_predictions_batch in train_dataloader:
                self.optimizer.zero_grad()
                predictions_batch = self(users_batch, movies_batch)
                loss = mse_loss(predictions_batch, target_predictions_batch)
                loss.backward()
                self.optimizer.step()
                step += 1

    def predict(self, users, movies):
        device = "cpu"

        users_torch = torch.tensor(users, device=device)
        movies_torch = torch.tensor(movies, device=device)
        return self(users_torch, movies_torch).cpu().detach().numpy()
