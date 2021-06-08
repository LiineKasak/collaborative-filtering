from utils import data_processing
from .algobase import AlgoBase
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def batch_dot(a, b):
    return torch.bmm(
        a.view(a.shape[0], 1, a.shape[1]),
        b.view(a.shape[0], a.shape[1], 1)
    ).view(a.shape[0])


def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)


class CFNADE(AlgoBase):

    class Model(nn.Module):
        def __init__(self, number_of_items, number_of_scores):
            super().__init__()
            self.number_of_scores = number_of_scores
            self.scores_tensor = torch.tensor(range(number_of_scores))

            self.hidden_units = 10
            self.hidden_bias = nn.Parameter(torch.rand(self.hidden_units, requires_grad=True))
            self.hidden_W = nn.Parameter(torch.rand((number_of_scores, self.hidden_units, number_of_items), requires_grad=True))

            self.score_bias = nn.Parameter(torch.rand((number_of_scores, number_of_items), requires_grad=True))
            self.score_V = nn.Parameter(torch.rand((number_of_scores, number_of_items, self.hidden_units), requires_grad=True))

        def hidden(self, histories):
            res = self.hidden_bias.repeat(len(histories), 1)
            for i, history in enumerate(histories):
                for (item, rating) in history:
                    res[i] += self.hidden_W[rating, :, item]
                res[i] = torch.tanh(res[i])
            return res

        def score(self, item, history, k):
            return self.score_bias[k,item] + batch_dot(self.score_V[k, item], self.hidden(history))

        def dist(self, item, history):
            scores = torch.stack([self.score(item, history, k) for k in range(self.number_of_scores)])
            return nn.Softmax(dim=0)(scores)

        def forward(self, item, history):
            res = torch.zeros(item.shape[0])
            for k in range(self.number_of_scores):
                res += k * self.dist(item, history)[k]
            return res


    def __init__(self, track_to_comet=False):
        AlgoBase.__init__(self, track_to_comet)
        self.model = self.Model(self.number_of_movies, 5)
        self.history = []

    def make_history(self, users, movies, predictions):
        self.history = [[] for u in range(self.number_of_users)]
        for (u, m, r) in zip(users, movies, predictions):
            self.history[u].append((m, r))

    def get_history(self, users):
        return [self.history[user] for user in users]

    def fit(self, users, movies, predictions, num_epochs=20, batch_size=128, learning_rate=1e-2):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        users_train, users_test, movies_train, movies_test, pred_train, pred_test =\
            train_test_split(torch.tensor(users), torch.tensor(movies), torch.tensor(predictions), train_size=0.9, random_state=42)
        self.make_history(users_train, movies_train, pred_train)
        dataloader = DataLoader(
            TensorDataset(users_train, movies_train, pred_train),
            batch_size=batch_size,
        )
        res = None
        best_rmse = 100
        step = 0
        with tqdm(total=len(dataloader) * num_epochs) as pbar:
            for epoch in range(num_epochs):
                for users_batch, movies_batch, target_predictions_batch in dataloader:
                    optimizer.zero_grad()
                    predictions_batch = self.model(movies_batch, self.get_history(users_batch))
                    loss = mse_loss(predictions_batch, target_predictions_batch)
                    loss.backward()
                    optimizer.step()
                    pbar.update(1)
                    step += 1
                with torch.no_grad():
                    all_predictions = self.model(movies_test, self.get_history(users_test))
                reconstuction_rmse = data_processing.get_score(all_predictions.cpu().numpy(), pred_test)
                pbar.set_description('At epoch {:3d} loss is {:.4f}'.format(epoch, reconstuction_rmse))
                if reconstuction_rmse < best_rmse:
                    res = self.model.state_dict()
        self.model.load_state_dict(res)

    def predict(self, users, movies):
        return self.model(movies, self.get_history(users))
