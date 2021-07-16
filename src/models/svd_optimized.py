import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import data_processing
from src.models.algobase import AlgoBase
from src.models.svd import SVD
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset


class SVD_optimized(AlgoBase):
    """
    Running optimizers on SVD initialized embeddings.
    """

    def __init__(self, k_singular_values=17, epochs=100, batch_size=64, learning_rate=0.001, regularization=0.05,
                 verbal=False,
                 track_to_comet=False, optimizer='SGD'):
        AlgoBase.__init__(self, track_to_comet)

        self.k = k_singular_values  # number of singular values to use
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.verbal = verbal
        self.optimizer = optimizer

        self.matrix = np.zeros((self.number_of_users, self.number_of_movies))
        self.reconstructed_matrix = np.zeros((self.number_of_users, self.number_of_movies))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device {self.device}.')
        self.pu = torch.zeros(self.number_of_users, self.k, requires_grad=True, device=self.device)  # user embedding
        self.qi = torch.zeros(self.number_of_movies, self.k, requires_grad=True, device=self.device)  # item embedding
        self.bu = torch.zeros(self.number_of_users, requires_grad=True, device=self.device)  # user bias
        self.bi = torch.zeros(self.number_of_movies, requires_grad=True, device=self.device)  # item bias
        self.mu = 0

    @staticmethod
    def _rmse(predictions, target_values):
        # for torch tensors
        return torch.sqrt(torch.mean(predictions - target_values) ** 2)

    def _optimizer(self):
        if self.optimizer == 'SGD':
            return torch.optim.SGD([self.pu, self.qi, self.bu, self.bi], lr=self.learning_rate,
                                   weight_decay=self.regularization)
        elif self.optimizer == 'Adadelta':
            return torch.optim.Adadelta([self.pu, self.qi, self.bu, self.bi], lr=self.learning_rate,
                                        weight_decay=self.regularization)
        elif self.optimizer == 'Adam':
            return torch.optim.Adam([self.pu, self.qi, self.bu, self.bi], lr=self.learning_rate,
                                    weight_decay=self.regularization)

    def _update_reconstructed_matrix(self):
        pu = self.pu.detach().cpu().numpy()
        qi = self.qi.detach().cpu().numpy()
        bu = self.bu.detach().cpu().numpy()
        bi = self.bi.detach().cpu().numpy()
        dot_product = pu.dot(qi.T)
        user_biases_matrix = np.reshape(bu, (self.number_of_users, 1))
        movie_biases_matrix = np.reshape(bi, (1, self.number_of_movies))
        self.reconstructed_matrix = dot_product + user_biases_matrix + movie_biases_matrix + self.mu

    def fit(self, users, movies, ground_truth, valid_users=None, valid_movies=None, valid_ground_truth=None):
        self.matrix, _ = data_processing.get_data_mask(users, movies, ground_truth)
        pu, qi = SVD.get_embeddings(self.k, self.matrix)
        self.pu = torch.tensor(pu, requires_grad=True, device=self.device)
        self.qi = torch.tensor(qi, requires_grad=True, device=self.device)
        self._update_reconstructed_matrix()

        run_validation = valid_users is not None and valid_movies is not None and valid_ground_truth is not None
        indices = np.arange(len(users))

        users = torch.tensor(users, device=self.device)
        movies = torch.tensor(movies, device=self.device)
        ground_truth = torch.tensor(ground_truth, device=self.device)
        train_dataloader = DataLoader(TensorDataset(users, movies, ground_truth), batch_size=self.batch_size)

        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f'./logs/SGD_{time_string}'
        writer = SummaryWriter(log_dir)

        optimizer = self._optimizer()

        with tqdm(total=self.epochs * len(train_dataloader), disable=not self.verbal) as pbar:
            for epoch in range(self.epochs):
                np.random.shuffle(indices)
                # print(self.pu.detach().cpu().numpy())
                for users_batch, movies_batch, truth_batch in train_dataloader:
                    predictions = torch.tensor(self.predict(users_batch, movies_batch), device=self.device)
                    loss = self._rmse(predictions, truth_batch).requires_grad_(True)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    self._update_reconstructed_matrix()
                    pbar.update(1)

                rmse_loss = torch.tensor(data_processing.get_score(predictions, ground_truth), requires_grad=True,
                                         device=self.device)

                writer.add_scalar('rmse', rmse_loss, epoch)
                self._update_reconstructed_matrix()

                if run_validation:
                    valid_predictions = self.predict(valid_users, valid_movies)
                    reconstruction_rmse = data_processing.get_score(valid_predictions, valid_ground_truth)
                    pbar.set_description(f'Epoch {epoch}:  rmse {rmse_loss:.4f}, val_rmse {reconstruction_rmse:.4f}')
                    writer.add_scalar('val_rmse', reconstruction_rmse, epoch)
                else:
                    pbar.set_description(f'Epoch {epoch}:  rmse {rmse_loss}')

    def predict(self, users, movies):
        with torch.no_grad():
            users = users.detach().cpu().numpy()
            movies = movies.detach().cpu().numpy()
            predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)
            predictions[predictions > 5] = 5
            predictions[predictions < 1] = 1
            return predictions


if __name__ == '__main__':
    data_pd = data_processing.read_data()
    k = 10
    epochs = 1
    sgd = SVD_optimized(k_singular_values=k, epochs=epochs, verbal=True, optimizer='Adam')

    submit = False

    if submit:
        users, movies, predictions = data_processing.extract_users_items_predictions(data_pd)
        sgd.fit(users, movies, predictions)
        sgd.predict_for_submission(f'svd_sgd_norm_k{k}_{epochs}')
    else:
        train_pd, test_pd = train_test_split(data_pd, train_size=0.9, random_state=42)
        users, movies, predictions = data_processing.extract_users_items_predictions(train_pd)
        val_users, val_movies, val_predictions = data_processing.extract_users_items_predictions(test_pd)
        sgd.fit(users, movies, predictions, val_users, val_movies, val_predictions)
