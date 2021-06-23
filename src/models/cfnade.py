from utils import data_processing
from .algobase import AlgoBase
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random


def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)


class CFNADE(AlgoBase):

    class Model(nn.Module):
        def __init__(self, number_of_items, scores_tensor, hidden_units):
            super().__init__()
            self.number_of_items = number_of_items
            self.scores = scores_tensor

            self.hidden_units = hidden_units
            self.hidden_bias = nn.Parameter(torch.rand(self.hidden_units, requires_grad=True))
            self.hidden_W = nn.Parameter(torch.rand((scores_tensor.shape[0], number_of_items, self.hidden_units), requires_grad=True))

            self.score_bias = nn.Parameter(torch.rand((scores_tensor.shape[0], number_of_items), requires_grad=True))
            self.score_V = nn.Parameter(torch.rand((scores_tensor.shape[0], number_of_items, self.hidden_units), requires_grad=True))

        def hidden(self, history: torch.Tensor):
            biases = self.hidden_bias.repeat(history.shape[0], 1)
            history_view = history.view(history.shape[0], self.scores.shape[0], 1, self.number_of_items)
            res = biases + torch.matmul(history_view, self.hidden_W).sum(axis=1).view(history.shape[0], self.hidden_units)
            return torch.tanh(res)

        def dist(self, item: torch.Tensor, history: torch.Tensor):
            v = self.score_V[:, item, :].view(self.scores.shape[0], item.shape[0], 1, self.hidden_units)
            h = self.hidden(history).view(history.shape[0], self.hidden_units, 1)
            dots = torch.matmul(v, h).view(self.scores.shape[0], item.shape[0])
            scores = self.score_bias[:, item] + dots
            scores = torch.cumsum(scores, dim=0)
            return nn.Softmax(dim=0)(scores).T

        def forward(self, item: torch.Tensor, history: torch.Tensor):
            d = self.dist(item, history)
            return torch.matmul(d, self.scores)


    def __init__(self, hidden_units, device, track_to_comet=False):
        AlgoBase.__init__(self, track_to_comet)
        self.number_of_ratings = 5
        scores = torch.tensor(range(self.number_of_ratings), device=device) + 1.0
        self.model = self.Model(self.number_of_movies, scores, hidden_units).to(device)
        self.history = []
        self.device = device

    def make_history(self, users, movies, predictions):
        self.history = torch.zeros((self.number_of_users, self.number_of_ratings, self.number_of_movies), device=self.device)
        for (u, m, r) in zip(users, movies, predictions):
            for i in range(r):
                self.history[u, i, m] = 1

    def fit(self, users, movies, predictions, num_epochs=20, batch_size=128, learning_rate=1e-2):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        users_train, users_test, movies_train, movies_test, pred_train, pred_test =\
            train_test_split(
                users,
                movies,
                predictions,
                train_size=0.9,
                random_state=42
            )
        user_movie_rating = torch.zeros((self.number_of_users, self.number_of_movies, self.number_of_ratings), device=self.device)
        for u, m, r in zip(users_train, movies_train, pred_train):
            user_movie_rating[u, m, r - 1] = 1
        test_dataloader = DataLoader(
            TensorDataset(
                torch.tensor(users_test, device=self.device), torch.tensor(movies_test, device=self.device)),
            batch_size=batch_size,
        )
        res = None
        best_rmse = 100
        step = 0
        with tqdm(total=len(user_movie_rating) * num_epochs) as pbar:
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                losses = torch.zeros(1, device=self.device)
                for r in user_movie_rating:
                    nz = r.nonzero(as_tuple=True)[0]
                    nnz = len(nz)
                    if nnz == 0:
                        continue
                    num_right = random.randint(1, nnz)
                    num_left = nnz - num_right
                    lt_i = torch.cat([torch.ones(num_left, dtype=torch.bool, device=self.device), torch.zeros(num_right, dtype=torch.bool, device=self.device)])
                    lt_i = lt_i[torch.randperm(nnz)]
                    m_o_lt_i = torch.zeros(len(r), dtype=torch.bool, device=self.device)
                    m_o_lt_i[nz[lt_i]] = True
                    m_o_lt_i = m_o_lt_i.view(self.number_of_movies, 1)
                    m_o_j = nz[~lt_i]
                    probs = (self.model.dist(m_o_j, (r * m_o_lt_i).T.repeat(len(m_o_j), 1, 1)) * r[m_o_j]).sum(axis=1)
                    losses -= nnz / len(m_o_j) * torch.log(probs).sum()
                    pbar.update(1)
                    step += 1
                losses.backward()
                optimizer.step()
                with torch.no_grad():
                    all_predictions = []
                    for users_batch, movies_batch in test_dataloader:
                        predictions_batch = self.model(movies_batch, user_movie_rating[users_batch].transpose(1, 2))
                        all_predictions.append(predictions_batch)
                    all_predictions = torch.cat(all_predictions)
                reconstruction_rmse = data_processing.get_score(all_predictions.cpu().numpy(), pred_test)
                pbar.set_description('At epoch {:3d} loss is {:.4f}'.format(epoch, reconstruction_rmse))
                if reconstruction_rmse < best_rmse:
                    res = self.model.state_dict()
        self.model.load_state_dict(res)

    def predict(self, users, movies):
        return self.model(movies, self.history[users])
