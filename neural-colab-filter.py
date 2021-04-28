import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from constants import *
from utils import DatasetUtil
import numpy as np


class NCF(nn.Module):
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

    def forward(self, users, movies):
        users_embedding = self.embedding_layer_users(users)
        movies_embedding = self.embedding_layer_movies(movies)
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        return torch.squeeze(self.feed_forward(concat))


def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)


def get_dataloaders(data_util):
    train_users, train_movies, train_predictions = data_util._extract_users_items_predictions(data_util._train_pd)
    test_users, test_movies, test_truth = data_util._extract_users_items_predictions(data_util._test_pd)

    # Build Dataloaders
    train_users_torch = torch.tensor(train_users, device=device)
    train_movies_torch = torch.tensor(train_movies, device=device)
    train_predictions_torch = torch.tensor(train_predictions, device=device)

    train_dataloader = DataLoader(
        TensorDataset(train_users_torch, train_movies_torch, train_predictions_torch),
        batch_size=batch_size)

    test_users_torch = torch.tensor(test_users, device=device)
    test_movies_torch = torch.tensor(test_movies, device=device)

    test_dataloader = DataLoader(
        TensorDataset(test_users_torch, test_movies_torch),
        batch_size=batch_size)
    return train_dataloader, test_dataloader


def get_predictions(ncf, dataloader, is_train=False):
    with torch.no_grad():
        all_predictions = []
        if is_train:
            for users_batch, movies_batch, _ in dataloader:
                predictions_batch = ncf(users_batch, movies_batch)
                all_predictions.append(predictions_batch)
        else:
            for users_batch, movies_batch in dataloader:
                predictions_batch = ncf(users_batch, movies_batch)
                all_predictions.append(predictions_batch)

    all_predictions = torch.cat(all_predictions)
    return all_predictions.cpu().numpy()


def train():
    data_util = DatasetUtil()
    train_dataloader, test_dataloader = get_dataloaders(data_util)

    ncf = NCF(NR_USERS, NR_MOVIES, embedding_size).to(device)

    optimizer = optim.Adam(ncf.parameters(), lr=learning_rate)

    ncf_logdir = f'./tensorboard/{model_name}'
    writer_train = SummaryWriter(f'{ncf_logdir}_train')
    writer_test = SummaryWriter(f'{ncf_logdir}_test')

    step = 0
    with tqdm(total=len(train_dataloader) * num_epochs) as pbar:
        for epoch in range(num_epochs):
            for users_batch, movies_batch, target_predictions_batch in train_dataloader:
                optimizer.zero_grad()
                predictions_batch = ncf(users_batch, movies_batch)
                loss = mse_loss(predictions_batch, target_predictions_batch)
                loss.backward()
                optimizer.step()

                writer_train.add_scalar('loss', loss, step)
                pbar.update(1)
                step += 1

            if epoch % show_validation_score_every_epochs == 0:
                train_predictions = get_predictions(ncf, train_dataloader, is_train=True)
                test_predictions = get_predictions(ncf, test_dataloader)

                train_rmse, test_rmse = data_util.rmse_scores(train_predictions, test_predictions)
                pbar.set_description('At epoch {:3d} loss is {:.4f}'.format(epoch, test_rmse))

                writer_train.add_scalar('reconstuction_rmse', train_rmse, step)
                writer_test.add_scalar('reconstuction_rmse', test_rmse, step)

    all_predictions = np.append(get_predictions(ncf, train_dataloader, is_train=True),
                                get_predictions(ncf, test_dataloader))
    data_util.save_predictions(all_predictions, model_name)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    # change model name with every new training
    model_name = 'ncf_v1'
    # Parameters
    batch_size = 1024
    num_epochs = 25
    show_validation_score_every_epochs = 1
    embedding_size = 16
    learning_rate = 1e-3

    train()
