import pandas as pd
import numpy as np
import math
import datetime
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm



# !pip install kaggle

# !mkdir ~/.kaggle

# kaggle_username = "..." #@param {type:"string"}
# kaggle_api_key = "..." #@param {type:"string"}

# assert len(kaggle_username) > 0 and len(kaggle_api_key) > 0

# api_token = {"username": kaggle_username,"key": kaggle_api_key}

# with open('kaggle.json', 'w') as file:
    # json.dump(api_token, file)

# !mv kaggle.json ~/.kaggle/kaggle.json

# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle competitions download -c cil-collaborative-filtering-2021

# !unzip data_train.csv.zip 
# !unzip sampleSubmission.csv.zip 


number_of_users, number_of_movies = (10000, 1000)

data_pd = pd.read_csv('data_train.csv')
print(data_pd.head(5))
print()
print('Shape', data_pd.shape)

submission_pd = pd.read_csv('sampleSubmission.csv')
print(submission_pd.head(5))
print()
print('Shape', submission_pd.shape)


# Split the dataset into train and test

train_size = 0.9

train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)


def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions

train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)

# also create full matrix of observed values
data = np.full((number_of_users, number_of_movies), np.mean(train_pd.Prediction.values))
mask = np.zeros((number_of_users, number_of_movies)) # 0 -> unobserved value, 1->observed value

for user, movie, pred in zip(train_users, train_movies, train_predictions):
    data[user - 1][movie - 1] = pred
    mask[user - 1][movie - 1] = 1


rmse = lambda x, y: math.sqrt(mean_squared_error(x, y))

test_users, test_movies, test_predictions = extract_users_items_predictions(test_pd)

# test our predictions with the true values
def get_score(predictions, target_values=test_predictions):
    return rmse(predictions, target_values)

def extract_prediction_from_full_matrix(reconstructed_matrix, users=test_users, movies=test_movies):
    # returns predictions for the users-movies combinations specified based on a full m \times n matrix
    assert(len(users) == len(movies)), "users-movies combinations specified should have equal length"
    predictions = np.zeros(len(test_users))

    for i, (user, movie) in enumerate(zip(users, movies)):
        predictions[i] = reconstructed_matrix[user][movie]

    return predictions


#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)


class GMF(nn.Module):
  def __init__(self, number_of_users, number_of_movies, embedding_size):
    super().__init__()
    self.embedding_users = nn.Embedding(number_of_users, embedding_size)
    self.embedding_movies = nn.Embedding(number_of_movies, embedding_size)

    self.weights = nn.Sequential(
      nn.Linear(in_features=embedding_size, out_features=1),
      nn.Sigmoid(),
    )

  def forward(self, users, movies):
    users_embedding = self.embedding_users(users)
    movies_embedding = self.embedding_movies(movies)
    product = torch.mul(users_embedding, movies_embedding)
    return torch.squeeze(self.weights(product)) * 4 + 1


class MLP(nn.Module):
  def __init__(self, number_of_users, number_of_movies, embedding_size):
    super().__init__()
    self.embedding_users = nn.Embedding(number_of_users, embedding_size)
    self.embedding_movies = nn.Embedding(number_of_movies, embedding_size)

    self.mlp = nn.Sequential(
      nn.Linear(in_features=2 * embedding_size, out_features=64),
      nn.ReLU(),
      nn.Linear(in_features=64, out_features=32),
      nn.ReLU(),
      nn.Linear(in_features=32, out_features=16),
      nn.ReLU(),
      nn.Linear(in_features=16, out_features=1),
      nn.Sigmoid(),
    )

  def forward(self, users, movies):
    users_embedding = self.embedding_users(users)
    movies_embedding = self.embedding_movies(movies)
    concat_embedding = torch.cat([users_embedding, movies_embedding], dim=1)
    return torch.squeeze(self.mlp(concat_embedding)) * 4 + 1


class NCF(nn.Module):
    def __init__(self, gmf: GMF, mlp: MLP, alpha):
        super().__init__()
        self.gmf = gmf
        self.mlp = mlp
        self.alpha = alpha

    def forward(self, users, movies):
        gmf_res = gmf.forward(users, movies)
        mlp_res = mlp.forward(users, movies)
        return alpha * gmf_res + (1 - alpha) * mlp_res



# Parameters
batch_size = 128
num_epochs = 50
show_validation_score_every_epochs = 1
gmf_embedding_size = 2
mlp_embedding_size = 8
alpha = 0.5
gmf_learning_rate = 1e-2
mlp_learning_rate = 1e-2
ncf_learning_rate = 1e-1


def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)

# Build Dataloaders
train_users_torch = torch.tensor(train_users, device=device)
train_movies_torch = torch.tensor(train_movies, device=device)
train_predictions_torch = torch.tensor(train_predictions, device=device)

train_dataloader = DataLoader(
    TensorDataset(train_users_torch, train_movies_torch, train_predictions_torch),
    batch_size=batch_size)

test_users_torch = torch.tensor(test_users, device=device)
test_movies_torch = torch.tensor(test_movies, device=device)

test_dataloader = DataLoader(
    TensorDataset(test_users_torch, test_movies_torch),
    batch_size=batch_size)


def train_model(optimizer, model, num_epochs, log_dir):
  res = None
  writer = SummaryWriter(log_dir)
  best_rmse = 100
  step = 0
  with tqdm(total=len(train_dataloader) * num_epochs) as pbar:
    for epoch in range(num_epochs):
      for users_batch, movies_batch, target_predictions_batch in train_dataloader:
        optimizer.zero_grad()
        predictions_batch = model(users_batch, movies_batch)
        loss = mse_loss(predictions_batch, target_predictions_batch)
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss', loss, step)
        pbar.update(1)
        step += 1
      if epoch % show_validation_score_every_epochs == 0:
        with torch.no_grad():
          all_predictions = []
          for users_batch, movies_batch in test_dataloader:
            predictions_batch = model(users_batch, movies_batch)
            all_predictions.append(predictions_batch)
        all_predictions = torch.cat(all_predictions)
        reconstuction_rmse = get_score(all_predictions.cpu().numpy())
        pbar.set_description('At epoch {:3d} loss is {:.4f}'.format(epoch, reconstuction_rmse))
        if reconstuction_rmse < best_rmse:
          res = model.state_dict()
        writer.add_scalar('reconstuction_rmse', reconstuction_rmse, step)
  return res


gmf = GMF(number_of_users, number_of_movies, gmf_embedding_size).to(device)
best_gmf_state = train_model(
  optimizer=optim.Adam(gmf.parameters(), lr=gmf_learning_rate),
  model=gmf,
  num_epochs=num_epochs,
  log_dir='./tensorboard/gmf/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)
gmf.load_state_dict(best_gmf_state)


mlp = MLP(number_of_users, number_of_movies, mlp_embedding_size).to(device)
best_mlp_state = train_model(
  optimizer=optim.Adam(mlp.parameters(), lr=mlp_learning_rate),
  model=mlp,
  num_epochs=num_epochs,
  log_dir='./tensorboard/mlp/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)
mlp.load_state_dict(best_mlp_state)


ncf = NCF(gmf, mlp, alpha).to(device)
best_ncf_state = train_model(
  optimizer=optim.SGD(ncf.parameters(), lr=ncf_learning_rate),
  model=ncf,
  num_epochs=num_epochs,
  log_dir='./tensorboard/ncf/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)
torch.save(best_ncf_state, 'best-ncf.pt') 
