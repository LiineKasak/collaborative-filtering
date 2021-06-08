from utils import data_processing
from models.cfnade import CFNADE
import torch

data_pd = data_processing.read_data()
users, movies, predictions = data_processing.extract_users_items_predictions(data_pd)
model = CFNADE()
model.fit(users, movies, predictions)
torch.save(model.model.state_dict(), 'cfnade_state.pt')
