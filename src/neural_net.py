from comet_ml import Experiment
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import torch.nn.functional as F
from torch.optim import SGD

import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys

directory = Path(__file__).parent.parent
directory_path = os.path.abspath(directory)
sys.path.append(directory_path)

from auxiliary import data_processing
from auxiliary.data_processing import get_score
from src.matrix_factorization import als_factorization, non_negative_matrix_factorization, sgd_factorization

criterion = nn.MSELoss()
#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# global variables
train_size = 0.9
number_of_users, number_of_movies = data_processing.get_number_of_users(), data_processing.get_number_of_users()

# Parameters
batch_size = 1024
num_epochs = 15
show_validation_score_every_epochs = 5
embedding_size = 5
learning_rate = 1e-3



class NCF(pl.LightningModule):
    def __init__(self, number_of_users, number_of_movies, embedding_size):
        super().__init__()
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

        self.loss = rmse_loss
        self.lr = learning_rate

    def forward(self, users, movies):
        users_embedding = self.embedding_layer_users(users)
        movies_embedding = self.embedding_layer_movies(movies)
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        return torch.squeeze(self.feed_forward(concat))

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        movies, users, predictions = train_batch
        logits = self.forward(users, movies)
        loss = self.loss(logits, predictions)
        # experiment.log_metrics(
        #     {
        #         "it_rmse": loss,
        #     }
        # )
        return loss







class vero_nn(pl.LightningModule):
    def __init__(self, number_of_users, number_of_movies, train_data_torch, embedding_size):
        super().__init__()
        U, V = non_negative_matrix_factorization(train_data_torch, 1000, embedding_size)
        self.embedding_layer_users = V
        self.embedding_layer_movies = U

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=2 * embedding_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),  # maybe predict per category?
            nn.ReLU()
        )

        self.loss = rmse_loss
        self.lr = learning_rate




    def forward(self, users, movies):
        users_embedding = self.embedding_layer_users[users]
        movies_embedding = self.embedding_layer_movies[movies]
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        return torch.squeeze(self.feed_forward(concat))

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        movies, users, predictions = train_batch
        logits = self.forward(users, movies)
        loss = self.loss(logits, predictions)
        # experiment.log_metrics(
        #     {
        #         "it_rmse": loss,
        #     }
        # )
        return loss

    def test_step(self, test_batch, batch_idx):
        movies, users, predictions = test_batch
        logits = self.forward(users, movies)
        loss = self.loss(logits, predictions)
        # experiment.log_metrics(
        #     {
        #         "it_rmse": loss,
        #     }
        # )
        return loss

class vero_nn_plus(pl.LightningModule):
    def __init__(self, number_of_users, number_of_movies, train_data_torch, embedding_size):
        super().__init__()
        U, V = non_negative_matrix_factorization(train_data_torch, 1000, embedding_size)
        self.embedding_layer_users = V
        self.embedding_layer_movies = U

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=2 * embedding_size, out_features=batch_size),
            nn.ReLU(),
            nn.Linear(in_features=batch_size, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),  # maybe predict per category?
            nn.ReLU()
        )

        self.loss = rmse_loss
        self.lr = learning_rate




    def forward(self, users, movies):
        users_embedding = self.embedding_layer_users[users]
        movies_embedding = self.embedding_layer_movies[movies]
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        return torch.squeeze(self.feed_forward(concat))

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        movies, users, predictions = train_batch
        logits = self.forward(users, movies)
        loss = self.loss(logits, predictions)
        # experiment.log_metrics(
        #     {
        #         "it_rmse": loss,
        #     }
        # )
        return loss

    def test_step(self, test_batch, batch_idx):
        movies, users, predictions = test_batch
        logits = self.forward(users, movies)
        loss = self.loss(logits, predictions)
        # experiment.log_metrics(
        #     {
        #         "it_rmse": loss,
        #     }
        # )
        return loss


def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)


def rmse_loss(predictions, target):
    return torch.sqrt(torch.mean((predictions - target) ** 2))



def run_vero_nn():
    experiment = Experiment(
        api_key="rISpuwcLQoWU6qan4jRCAPy5s",
        project_name="cil-experiments",
        workspace="veroniquek",
    )
    DATA_PATH = directory_path + '/data/data_train.csv'
    data_pd = pd.read_csv(DATA_PATH)
    train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)
    train_users, train_movies, train_predictions = data_processing.extract_users_items_predictions(train_pd)
    test_users, test_movies, test_predictions = data_processing.extract_users_items_predictions(test_pd)

    train_data, mask = data_processing.get_data_mask(train_users, train_movies, train_predictions)
    train_data_torch = torch.from_numpy(train_data)
    # ------------
    # Load the data
    # ------------
    train_users_torch = torch.tensor(train_users, device=device)
    train_movies_torch = torch.tensor(train_movies, device=device)
    train_predictions_torch = torch.tensor(train_predictions, device=device)

    train_dataloader = DataLoader(
        TensorDataset(train_users_torch, train_movies_torch, train_predictions_torch),
        num_workers=0,
        batch_size=batch_size)

    test_users_torch = torch.tensor(test_users, device=device)
    test_movies_torch = torch.tensor(test_movies, device=device)
    test_predictions_torch = torch.tensor(test_predictions, device=device)

    test_dataloader = DataLoader(
        TensorDataset(test_users_torch, test_movies_torch, test_predictions_torch),
        num_workers=0,
        batch_size=batch_size)

    # ------------
    # Model 1
    # ------------
    ncf = vero_nn(
        number_of_users=number_of_users,
        number_of_movies=number_of_movies,
        train_data_torch=train_data_torch,
        embedding_size=embedding_size,
    )

    # ------------
    # Training
    # ------------
    # We want to log to Tensorboard.

    # Make sure to connect to gpu runtime!
    trainer = pl.Trainer(gpus=0,
                         max_epochs=num_epochs)

    trainer.fit(ncf, train_dataloader=train_dataloader)

    predictions = ncf(test_movies_torch, test_users_torch).cpu().detach().numpy()
    predictions = ncf(test_movies_torch, test_users_torch).cpu().detach().numpy()
    rmse = get_score(predictions, test_predictions)
    print("vero_nn score is", get_score(predictions, test_predictions))
    experiment.log_metrics(
        {
            "root_mean_squared_error": rmse
        }
    )


def run_nn_plus():
    experiment = Experiment(
        api_key="rISpuwcLQoWU6qan4jRCAPy5s",
        project_name="cil-experiments",
        workspace="veroniquek",
    )

    DATA_PATH = directory_path + '/data/data_train.csv'
    data_pd = pd.read_csv(DATA_PATH)
    train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)
    train_users, train_movies, train_predictions = data_processing.extract_users_items_predictions(train_pd)
    test_users, test_movies, test_predictions = data_processing.extract_users_items_predictions(test_pd)

    train_data, mask = data_processing.get_data_mask(train_users, train_movies, train_predictions)
    train_data_torch = torch.from_numpy(train_data)
    # ------------
    # Load the data
    # ------------
    train_users_torch = torch.tensor(train_users, device=device)
    train_movies_torch = torch.tensor(train_movies, device=device)
    train_predictions_torch = torch.tensor(train_predictions, device=device)

    train_dataloader = DataLoader(
        TensorDataset(train_users_torch, train_movies_torch, train_predictions_torch),
        num_workers=0,
        batch_size=batch_size)

    test_users_torch = torch.tensor(test_users, device=device)
    test_movies_torch = torch.tensor(test_movies, device=device)
    test_predictions_torch = torch.tensor(test_predictions, device=device)

    test_dataloader = DataLoader(
        TensorDataset(test_users_torch, test_movies_torch, test_predictions_torch),
        num_workers=0,
        batch_size=batch_size)

    # ------------
    # Model 1
    # ------------
    ncf = NCF(
        number_of_users=number_of_users,
        number_of_movies=number_of_movies,
        embedding_size=embedding_size,
    )

    # ------------
    # Training
    # ------------
    # We want to log to Tensorboard.

    # Make sure to connect to gpu runtime!
    trainer = pl.Trainer(gpus=0,
                         max_epochs=num_epochs)

    trainer.fit(ncf, train_dataloader=train_dataloader)

    predictions = ncf(test_movies_torch, test_users_torch).cpu().detach().numpy()
    rmse = get_score(predictions, test_predictions)
    print("vero_nn_plus score is", get_score(predictions, test_predictions))
    experiment.log_metrics(
        {
            "root_mean_squared_error": rmse
        }
    )

def run_ncf_nn():
    experiment = Experiment(
        api_key="rISpuwcLQoWU6qan4jRCAPy5s",
        project_name="cil-experiments",
        workspace="veroniquek",
    )

    DATA_PATH = directory_path + '/data/data_train.csv'
    data_pd = pd.read_csv(DATA_PATH)
    train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)
    train_users, train_movies, train_predictions = data_processing.extract_users_items_predictions(train_pd)
    test_users, test_movies, test_predictions = data_processing.extract_users_items_predictions(test_pd)

    train_data, mask = data_processing.get_data_mask(train_users, train_movies, train_predictions)
    train_data_torch = torch.from_numpy(train_data)
    # ------------
    # Load the data
    # ------------
    train_users_torch = torch.tensor(train_users, device=device)
    train_movies_torch = torch.tensor(train_movies, device=device)
    train_predictions_torch = torch.tensor(train_predictions, device=device)

    train_dataloader = DataLoader(
        TensorDataset(train_users_torch, train_movies_torch, train_predictions_torch),
        num_workers=0,
        batch_size=batch_size)

    test_users_torch = torch.tensor(test_users, device=device)
    test_movies_torch = torch.tensor(test_movies, device=device)
    test_predictions_torch = torch.tensor(test_predictions, device=device)

    test_dataloader = DataLoader(
        TensorDataset(test_users_torch, test_movies_torch, test_predictions_torch),
        num_workers=0,
        batch_size=batch_size)

    # ------------
    # Model 1
    # ------------
    ncf = NCF(
        number_of_users=number_of_users,
        number_of_movies=number_of_movies,
        embedding_size=embedding_size,
    )

    # ------------
    # Training
    # ------------
    # We want to log to Tensorboard.

    # Make sure to connect to gpu runtime!
    trainer = pl.Trainer(gpus=0,
                         max_epochs=num_epochs)

    trainer.fit(ncf, train_dataloader=train_dataloader)

    predictions = ncf(test_movies_torch, test_users_torch).cpu().detach().numpy()
    rmse = get_score(predictions, test_predictions)
    print("ncf score is", get_score(predictions, test_predictions))
    experiment.log_metrics(
        {
            "root_mean_squared_error": rmse
        }
    )
