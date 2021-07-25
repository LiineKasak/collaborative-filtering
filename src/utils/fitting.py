import torch
from utils import data_processing
from utils.dataset import DatasetWrapper
from tqdm import tqdm
from copy import deepcopy


def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)


def train(
        train_data: DatasetWrapper,
        test_data: DatasetWrapper,
        device,
        model,
        optimizer,
        num_epochs,
        batch_size,
        verbose=True,
):
    train_users_torch = torch.tensor(train_data.users, device=device)
    train_movies_torch = torch.tensor(train_data.movies, device=device)

    if test_data:
        test_users_torch = torch.tensor(test_data.users, device=device)
        test_movies_torch = torch.tensor(test_data.movies, device=device)

    train_dataloader = train_data.create_dataloader(batch_size, device)

    best_state = None
    best_rmse = 100
    best_epochs = 0
    step = 0
    with tqdm(total=len(train_dataloader) * num_epochs, disable=not verbose) as pbar:
        for epoch in range(num_epochs):
            for users_batch, movies_batch, target_predictions_batch in train_dataloader:
                optimizer.zero_grad()
                predictions_batch = model(users_batch, movies_batch)
                loss = mse_loss(predictions_batch, target_predictions_batch)
                loss.backward()
                optimizer.step()
                pbar.update(1)
                step += 1
            with torch.no_grad():
                train_score_hat = model(train_users_torch, train_movies_torch)
                train_rmse = data_processing.get_score(train_score_hat.cpu().numpy(), train_data.ratings)
                if test_data:
                    test_score_hat = model(test_users_torch, test_movies_torch)
                    test_rmse = data_processing.get_score(test_score_hat.cpu().numpy(), test_data.ratings)
                    pbar.set_description(f'Epoch {epoch}: train loss {train_rmse:.4f}, test loss {test_rmse:.4f}')
                else:
                    pbar.set_description(f'Epoch {epoch}: train loss {train_rmse:.4f}')
                    test_rmse = train_rmse
                if test_rmse < best_rmse:
                    best_state = deepcopy(model.state_dict())
                    best_rmse = test_rmse
                    best_epochs = epoch + 1
    model.load_state_dict(best_state)
    model.eval()
    return best_rmse, best_epochs
