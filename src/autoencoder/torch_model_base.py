import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from auxiliary import data_processing
from src.algobase import AlgoBase


class TorchModelTrainer(AlgoBase):

    def predict(self, users, movies):
        raise NotImplementedError("predict-function has to be implemented! ")

    def __init__(self, model_name, epochs=50, batch_size=128, re_feeding=False, learning_rate=0.001,
                 regularization=0.01, dropout=0, logging=True):
        AlgoBase.__init__(self)
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.re_feeding = re_feeding
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.dropout = dropout
        self.logging = logging

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                          weight_decay=self.regularization)

    def rebuild_model(self):
        self.model = self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                          weight_decay=self.regularization)

    def build_model(self):
        raise NotImplementedError("build_model-function has to be implemented! ")

    def get_dataloader(self, data_pd):
        raise NotImplementedError("get_dataloader-function has to be implemented! ")

    def train_step(self, batch, re_feeding=False):
        raise NotImplementedError("train_step-function has to be implemented! ")

    def fit(self, users, movies, predictions):
        self.train((users, movies, predictions))

    def train(self, train_data: tuple, validation_data: tuple = None):
        # data as tuple (users, movies, predictions)
        if len(train_data) != 3:
            raise AttributeError("train_data has to be a tuple consisting of users, movies, predictions")
        self.rebuild_model()
        train_dataloader = self.get_dataloader(train_data)

        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f'./logs/{self.model_name}_{time_string}'
        writer = SummaryWriter(log_dir)

        step = 0
        with tqdm(total=len(train_dataloader) * self.epochs, disable=not self.logging) as pbar:
            for epoch in range(self.epochs):
                for batch in train_dataloader:
                    output, loss = self.train_step(batch)
                    if self.re_feeding:
                        output, loss = self.train_step((output.detach(), batch), True)

                    writer.add_scalar('loss', loss, step)
                    pbar.update(1)
                step += 1

                if validation_data:
                    valid_users, valid_movies, valid_predictions = validation_data
                    predictions = self.predict(valid_users, valid_movies)
                    reconstruction_rmse = data_processing.get_score(predictions, valid_predictions)
                    pbar.set_description('At epoch {:3d} loss is {:.4f}'.format(epoch, reconstruction_rmse))

                    writer.add_scalar('val_rmse', reconstruction_rmse, step)
