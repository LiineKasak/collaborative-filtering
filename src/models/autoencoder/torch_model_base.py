import time
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import data_processing
from utils.dataset import DatasetWrapper
from models.algobase import AlgoBase


class TorchModelTrainer(AlgoBase):
    """ Trainer base used for Autoencoder implementations. """

    def predict(self, users, movies):
        raise NotImplementedError("predict-function has to be implemented! ")

    def __init__(self, model_name, epochs=50, batch_size=128, re_feeding=False, learning_rate=0.001,
                 regularization=0.01, dropout=0, verbal=True):
        AlgoBase.__init__(self)
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.re_feeding = re_feeding
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.dropout = dropout
        self.verbal = verbal

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                          weight_decay=self.regularization)

    def rebuild_model(self):
        self.model = self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                          weight_decay=self.regularization)

    def get_unknown(self):
        # results from SVD_SGD
        unknown_values_pd = pd.read_csv(data_processing.get_project_directory() + '\data\svd_sgd_valid_seed42.csv.zip')
        return data_processing.extract_users_items_predictions(unknown_values_pd)

    def build_model(self):
        raise NotImplementedError("build_model-function has to be implemented! ")

    def get_dataloader(self, data_pd):
        raise NotImplementedError("get_dataloader-function has to be implemented! ")

    def train_step(self, batch, re_feeding=False):
        raise NotImplementedError("train_step-function has to be implemented! ")

    def fit(self, data: DatasetWrapper):
        self.train(data)

    def train(self, data: DatasetWrapper, val_data: DatasetWrapper = None):
        self.rebuild_model()
        train_dataloader = self.get_dataloader(data)

        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f'./logs/{self.model_name}_{time_string}'
        writer = SummaryWriter(log_dir)

        step = 0
        with tqdm(total=len(train_dataloader) * self.epochs, disable=not self.verbal) as pbar:
            for epoch in range(self.epochs):
                for batch in train_dataloader:
                    output, loss = self.train_step(batch)
                    if self.re_feeding:
                        output, loss = self.train_step((output.detach(), batch), True)

                    writer.add_scalar('loss', loss, step)
                    pbar.update(1)
                step += 1

                predictions = self.predict(data.users, data.movies)
                rmse = data_processing.get_score(predictions, data.ratings)
                writer.add_scalar('rmse', rmse, step)

                if val_data is not None:
                    predictions = self.predict(val_data.users, val_data.movies)
                    reconstruction_rmse = data_processing.get_score(predictions, val_data.ratings)
                    pbar.set_description('Epoch {:3d}: val_loss is {:.4f}'.format(epoch, reconstruction_rmse))

                    writer.add_scalar('val_rmse', reconstruction_rmse, step)
