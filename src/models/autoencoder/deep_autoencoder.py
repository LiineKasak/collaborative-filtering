import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.utils import data_processing
from src.models.autoencoder.torch_model_base import TorchModelTrainer


class AutoEncoderModel(torch.nn.Module):
    def __init__(self, dropout=None):
        super().__init__()

        layer_dims = [1000, 16, 32, 16, 1000]
        self.layers = nn.ModuleList()

        # Encoder
        k = int(len(layer_dims) / 2)
        current_dim = layer_dims[0]
        for layer_dim in layer_dims[1:k + 1]:
            self.layers.append(nn.Linear(in_features=current_dim, out_features=layer_dim))
            self.layers.append(nn.SELU())
            current_dim = layer_dim

        if dropout:
            self.layers.append(nn.Dropout(dropout))

        # Decoder
        for layer_dim in layer_dims[k + 1:-1]:
            self.layers.append(nn.Linear(in_features=current_dim, out_features=layer_dim))
            self.layers.append(nn.SELU())
            current_dim = layer_dim

        # Output
        self.layers.append(nn.Linear(in_features=current_dim, out_features=layer_dims[-1]))
        self.layers.append(nn.Sigmoid())

    def forward(self, data):
        for layer in self.layers:
            data = layer(data)
        return data * 4 + 1


class DeepAutoEncoder(TorchModelTrainer):
    """
    Inspired by:
    Training Deep AutoEncoders for Collaborative Filtering
    https://arxiv.org/pdf/1708.01715.pdf
    """

    def __init__(self, epochs=150, dropout=0.9, verbal=True):
        self.dropout = dropout
        super().__init__('auto_encoder', epochs=epochs, batch_size=128, learning_rate=0.0001,
                         regularization=0.01, dropout=self.dropout, re_feeding=False, verbal=verbal)

    def build_model(self):
        return AutoEncoderModel(self.dropout).to(self.device)

    @staticmethod
    def weighted_masked_mse(data_batch, output_batch, mask_batch):
        mask_weight = 0.6
        known_mse = mask_weight * torch.mean(mask_batch * (data_batch - output_batch) ** 2)
        unknown_mse = (1 - mask_weight) * torch.mean((1 - mask_batch) * (data_batch - output_batch) ** 2)
        return known_mse + unknown_mse

    def get_dataloader(self, data: tuple):
        users, movies, predictions = data
        _, mask = data_processing.get_data_mask(users, movies, predictions)
        unknown_users, unknown_movies, unknown_predictions = self.get_unknown()
        data, _ = data_processing.get_data_mask(np.append(users, unknown_users), np.append(movies, unknown_movies),
                                                np.append(predictions, unknown_predictions))

        self.data_torch = torch.tensor(data, device=self.device).float()
        self.mask_torch = torch.tensor(mask, device=self.device)
        return DataLoader(TensorDataset(self.data_torch, self.mask_torch), batch_size=self.batch_size)

    def train_step(self, batch, re_feeding=False):
        if re_feeding:
            data_batch, (_, mask_batch) = batch
        else:
            data_batch, mask_batch = batch
        self.optimizer.zero_grad()
        output_batch = self.model(data_batch)
        loss = self.weighted_masked_mse(data_batch, output_batch, mask_batch)
        loss.backward()
        self.optimizer.step()
        return output_batch, loss

    def reconstruct_whole_matrix(self, matrix):
        shape = (data_processing.get_number_of_users(), data_processing.get_number_of_movies())
        data_reconstructed = np.zeros(shape)

        with torch.no_grad():
            for i in range(0, shape[0], self.batch_size):
                upper_bound = min(i + self.batch_size, shape[0])
                data_reconstructed[i:upper_bound] = self.model(matrix[i:upper_bound]).detach().cpu().numpy()

        return data_reconstructed

    def predict(self, users, movies):
        reconstructed_matrix = self.reconstruct_whole_matrix(self.data_torch)
        return data_processing.extract_prediction_from_full_matrix(reconstructed_matrix, users, movies)
