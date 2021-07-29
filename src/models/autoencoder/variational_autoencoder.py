import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import argparse

from utils import data_processing
from utils.dataset import DatasetWrapper
from models.autoencoder.torch_model_base import TorchModelTrainer


class VAEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = data_processing.get_number_of_movies()
        dim1 = 512
        dim2 = 16

        self.fc1 = nn.Linear(self.input_dim, dim1)
        self.fc21 = nn.Linear(dim1, dim2)
        self.fc22 = nn.Linear(dim1, dim2)
        self.fc3 = nn.Linear(dim2, dim1)
        self.fc4 = nn.Linear(dim1, self.input_dim)

    def encode(self, x):
        h1 = nn.functional.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    @staticmethod
    def re_parameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = nn.functional.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.re_parameterize(mu, log_var)
        return self.decode(z), mu, log_var


class VAE(TorchModelTrainer):
    """
    Inspired by:
    Auto-Encoding Variational Bayes
    https://arxiv.org/abs/1312.6114
    """

    def __init__(self, params: argparse.Namespace):
        super().__init__('variational_autoencoder', epochs=params.epochs, batch_size=params.batch_size,
                         learning_rate=params.learning_rate, regularization=params.regularization,
                         re_feeding=False, verbal=params.verbal)

    @staticmethod
    def default_params():
        return argparse.Namespace(epochs=105, batch_size=128, learning_rate=0.0005, regularization=0.001, verbal=True)

    def build_model(self):
        return VAEModel().to(self.device)

    @staticmethod
    def bce_kld_loss(x, recon_x, mu, log_var):
        # Reconstruction + KL divergence losses summed over all elements and batch
        bce = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return bce + kld

    def get_dataloader(self, data: DatasetWrapper):
        data, mask = data_processing.get_data_mask(data.users, data.movies, data.ratings)

        self.data_torch = (torch.tensor(data, device=self.device).float() - 1) / 4
        self.mask_torch = torch.tensor(mask, device=self.device)
        return DataLoader(TensorDataset(self.data_torch, self.mask_torch), batch_size=self.batch_size)

    def train_step(self, batch, re_feeding=False):
        data_batch, mask_batch = batch
        self.optimizer.zero_grad()
        output_batch, mu, log_var = self.model(data_batch)
        loss = self.bce_kld_loss(data_batch, output_batch, mu, log_var)
        loss.backward()
        self.optimizer.step()
        return output_batch, loss

    def reconstruct_whole_matrix(self, matrix):
        shape = (data_processing.get_number_of_users(), data_processing.get_number_of_movies())
        data_reconstructed = np.zeros(shape)

        with torch.no_grad():
            for i in range(0, shape[0], self.batch_size):
                upper_bound = min(i + self.batch_size, shape[0])
                data_reconstructed[i:upper_bound] = self.model(matrix[i:upper_bound])[0].detach().cpu().numpy()

        return data_reconstructed

    def predict(self, users, movies):
        reconstructed_matrix = self.reconstruct_whole_matrix(self.data_torch) * 4 + 1
        return data_processing.extract_prediction_from_full_matrix(reconstructed_matrix, users, movies)
