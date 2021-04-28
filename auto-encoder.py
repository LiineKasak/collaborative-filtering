from constants import *
from utils import DatasetUtil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, input_dimension, encoded_dimension=16):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=input_dimension, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=encoded_dimension),
            nn.ReLU()
        )

    def forward(self, data):
        return self.model(data)


class Decoder(nn.Module):
    def __init__(self, output_dimensions, encoded_dimension=16):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=encoded_dimension, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_dimensions),
            nn.ReLU()  # How does the output look like? What about if you had first centered the data?!
        )

    def forward(self, encodings):
        return self.model(encodings)


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(self.encoder(data))


# L2 loss between original ratings and reconstructed ratings for the observed values
def loss_function(original, reconstructed, mask):
    return torch.mean(mask * (original - reconstructed) ** 2)


# reconstuct the whole array
def reconstruct_whole_matrix(data_torch, autoencoder):
    data_reconstructed = np.zeros((NR_USERS, NR_MOVIES))

    with torch.no_grad():
        for i in range(0, NR_USERS, batch_size):
            upper_bound = min(i + batch_size, NR_USERS)
            data_reconstructed[i:upper_bound] = autoencoder(data_torch[i:upper_bound]).detach().cpu().numpy()

    return data_reconstructed


def train():
    data_util = DatasetUtil()
    autoencoder = AutoEncoder(
        encoder=Encoder(
            input_dimension=NR_MOVIES,
            encoded_dimension=encoded_dimension,
        ),
        decoder=Decoder(
            output_dimensions=NR_MOVIES,
            encoded_dimension=encoded_dimension,
        )
    ).to(device)

    optimizer = optim.Adam(autoencoder.parameters(),
                           lr=learning_rate)

    # Build Dataloaders
    data_torch = torch.tensor(data_util.data, device=device).float()
    mask_torch = torch.tensor(data_util.mask, device=device)

    dataloader = DataLoader(
        TensorDataset(data_torch, mask_torch),
        batch_size=batch_size)

    autoencoder_logdir = f'./tensorboard/{model_name}'
    writer = SummaryWriter(autoencoder_logdir)

    step = 0
    with tqdm(total=len(dataloader) * num_epochs) as pbar:
        for epoch in range(num_epochs):
            for data_batch, mask_batch in dataloader:
                optimizer.zero_grad()
                reconstructed_batch = autoencoder(data_batch)
                loss = loss_function(data_batch, reconstructed_batch, mask_batch)
                loss.backward()
                optimizer.step()

                writer.add_scalar('loss', loss, step)
                pbar.update(1)
                step += 1

            if epoch % show_validation_score_every_epochs == 0:
                reconstructed_matrix = reconstruct_whole_matrix(data_torch, autoencoder)
                _, test_rmse = data_util.rmse_scores_from_matrix(reconstructed_matrix)
                pbar.set_description('At epoch {:3d} loss is {:.4f}'.format(epoch, test_rmse))

                writer.add_scalar('reconstuction_rmse', test_rmse, step)

    reconstructed_matrix = reconstruct_whole_matrix(data_torch, autoencoder)
    data_util.save_predictions_from_matrix(reconstructed_matrix, model_name)


if __name__ == '__main__':
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    # change model name with every new training for new log graph
    model_name = 'autoencoder_v1'
    # Parameters
    batch_size = 64
    num_epochs = 1000
    show_validation_score_every_epochs = 5
    encoded_dimension = 16
    learning_rate = 1e-3

    train()
