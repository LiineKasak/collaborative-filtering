from utils import data_processing
from .algobase import AlgoBase
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)


class CFNADE(AlgoBase):

    class Model(nn.Module):
        def __init__(self, number_of_items, scores_tensor, hidden_units):
            super().__init__()
            self.number_of_items = number_of_items
            self.scores = scores_tensor

            self.register_buffer('score_add_mask', torch.triu(torch.ones((scores_tensor.shape[0], scores_tensor.shape[0]))), persistent=False)

            self.hidden_units = hidden_units
            self.hidden_bias = nn.Parameter(torch.rand(self.hidden_units, requires_grad=True))
            self.hidden_W = nn.Parameter(torch.rand((scores_tensor.shape[0], number_of_items, self.hidden_units), requires_grad=True))

            self.score_bias = nn.Parameter(torch.rand((scores_tensor.shape[0], number_of_items), requires_grad=True))
            self.score_V = nn.Parameter(torch.rand((scores_tensor.shape[0], number_of_items, self.hidden_units), requires_grad=True))

        def hidden(self, history: torch.Tensor):
            biases = self.hidden_bias.repeat(history.shape[0], 1)
            history_view = history.view(history.shape[0], self.scores.shape[0], 1, self.number_of_items)
            return biases + torch.matmul(history_view, self.hidden_W).sum(axis=1).view(history.shape[0], self.hidden_units)

        def dist(self, item: torch.Tensor, history: torch.Tensor):
            v = self.score_V[:, item, :].view(self.scores.shape[0], item.shape[0], 1, self.hidden_units)
            h = self.hidden(history).view(history.shape[0], self.hidden_units, 1)
            dots = torch.matmul(v, h).view(self.scores.shape[0], item.shape[0])
            scores = self.score_bias[:, item] + dots
            scores = torch.matmul(scores.T, self.score_add_mask)
            return nn.Softmax(dim=1)(scores)

        def forward(self, item: torch.Tensor, history: torch.Tensor):
            d = self.dist(item, history)
            return torch.matmul(d, self.scores)


    def __init__(self, hidden_units, device, track_to_comet=False):
        AlgoBase.__init__(self, track_to_comet)
        self.number_of_scores = 5
        scores = torch.tensor(range(self.number_of_scores), device=device) + 1.0
        self.model = self.Model(self.number_of_movies, scores, hidden_units).to(device)
        self.history = []
        self.device = device

    def make_history(self, users, movies, predictions):
        self.history = torch.zeros((self.number_of_users, self.number_of_scores, self.number_of_movies), device=self.device)
        for (u, m, r) in zip(users, movies, predictions):
            for i in range(r):
                self.history[u, i, m] = 1

    def fit(self, users, movies, predictions, num_epochs=20, batch_size=128, learning_rate=1e-2):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        users_train, users_test, movies_train, movies_test, pred_train, pred_test =\
            train_test_split(
                torch.tensor(users, device=self.device),
                torch.tensor(movies, device=self.device),
                torch.tensor(predictions, device=self.device),
                train_size=0.9,
                random_state=42
            )
        self.make_history(users_train, movies_train, pred_train)
        train_dataloader = DataLoader(
            TensorDataset(users_train, movies_train, pred_train),
            batch_size=batch_size,
        )
        test_dataloader = DataLoader(
            TensorDataset(users_test, movies_test),
            batch_size=batch_size,
        )
        res = None
        best_rmse = 100
        step = 0
        with tqdm(total=len(train_dataloader) * num_epochs) as pbar:
            for epoch in range(num_epochs):
                for users_batch, movies_batch, target_predictions_batch in train_dataloader:
                    optimizer.zero_grad()
                    predictions_batch = self.model(movies_batch, self.history[users_batch])
                    loss = mse_loss(predictions_batch, target_predictions_batch)
                    loss.backward()
                    optimizer.step()
                    pbar.update(1)
                    step += 1
                with torch.no_grad():
                    all_predictions = []
                    for users_batch, movies_batch in test_dataloader:
                        predictions_batch = self.model(movies_batch, self.history[users_batch])
                        all_predictions.append(predictions_batch)
                    all_predictions = torch.cat(all_predictions)
                reconstruction_rmse = data_processing.get_score(all_predictions.cpu().numpy(), pred_test.cpu().numpy())
                pbar.set_description('At epoch {:3d} loss is {:.4f}'.format(epoch, reconstruction_rmse))
                if reconstruction_rmse < best_rmse:
                    res = self.model.state_dict()
        self.model.load_state_dict(res)

    def predict(self, users, movies):
        return self.model(movies, self.history[users])
