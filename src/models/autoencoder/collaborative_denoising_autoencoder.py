import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.utils import data_processing
from src.models.autoencoder.torch_model_base import TorchModelTrainer


class CDAEModel(torch.nn.Module):
    def __init__(self, factors, dropout):
        super().__init__()
        self.factors = factors
        self.dropout = dropout
        self._build_model()

    def _build_model(self):
        self.user_score = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(in_features=data_processing.get_number_of_movies(), out_features=self.factors),
            nn.SELU()
        )

        self.user_content = nn.Sequential(
            nn.Embedding(data_processing.get_number_of_users() + 1, self.factors),
            nn.Flatten()
        )

        self.model = nn.Sequential(
            nn.Linear(in_features=self.factors, out_features=data_processing.get_number_of_movies()),
            nn.Sigmoid()
        )

    def forward(self, movies, user_id):
        user_score = self.user_score(movies)
        user_content = self.user_content(user_id)
        return self.model(torch.add(user_score, user_content)) * 4 + 1


class CDAE(TorchModelTrainer):
    """
    Inspired by:
    Collaborative Denoising Auto-Encoders for Top-N Recommender Systems
    http://alicezheng.org/papers/wsdm16-cdae.pdf
    """

    def __init__(self, epochs=25, verbal=True):
        self.dropout = 0.9
        self.factors = 16
        super().__init__('auto_encoder', epochs=epochs, batch_size=64, learning_rate=0.0005,
                         regularization=0.001, dropout=self.dropout, re_feeding=False, verbal=verbal)

    def build_model(self):
        return CDAEModel(self.factors, self.dropout).to(self.device)

    @staticmethod
    def mse(target, predictions):
        return torch.mean((target - predictions) ** 2)

    def get_dataloader(self, data: tuple):
        users, movies, predictions = data
        unknown_users, unknown_movies, unknown_predictions = self.get_unknown()
        data, _ = data_processing.get_data_mask(np.append(users, unknown_users), np.append(movies, unknown_movies),
                                                np.append(predictions, unknown_predictions))
        users = np.arange(data_processing.get_number_of_users())

        self.data_torch = torch.tensor(data, device=self.device).float()
        self.users_torch = torch.tensor(users, device=self.device).int()
        return DataLoader(TensorDataset(self.data_torch, self.users_torch), batch_size=self.batch_size)

    def train_step(self, batch, re_feeding=False):
        data_batch, users_batch = batch
        self.optimizer.zero_grad()
        output_batch = self.model(data_batch, users_batch)
        loss = self.mse(data_batch, output_batch)
        loss.backward()
        self.optimizer.step()
        return output_batch, loss

    def reconstruct_whole_matrix(self, matrix, users):
        shape = (data_processing.get_number_of_users(), data_processing.get_number_of_movies())
        data_reconstructed = np.zeros(shape)

        with torch.no_grad():
            for i in range(0, shape[0], self.batch_size):
                upper_bound = min(i + self.batch_size, shape[0])
                data_reconstructed[i:upper_bound] = self.model(matrix[i:upper_bound],
                                                               users[i:upper_bound]).detach().cpu().numpy()

        return data_reconstructed

    def predict(self, users, movies):
        reconstructed_matrix = self.reconstruct_whole_matrix(self.data_torch, self.users_torch)
        return data_processing.extract_prediction_from_full_matrix(reconstructed_matrix, users, movies)
