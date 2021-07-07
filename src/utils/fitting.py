import torch
from utils import data_processing
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from copy import deepcopy


def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)


def train(
        train_users,
        train_movies,
        train_predictions,
        test_users,
        test_movies,
        test_predictions,
        device,
        model,
        optimizer,
        num_epochs,
        batch_size,
        verbose=True,
):
    train_users_torch = torch.tensor(train_users, device=device)
    train_movies_torch = torch.tensor(train_movies, device=device)
    train_predictions_torch = torch.tensor(train_predictions, device=device)

    if test_users is None:
        do_validate = False
    else:
        do_validate = True
        test_users_torch = torch.tensor(test_users, device=device)
        test_movies_torch = torch.tensor(test_movies, device=device)

    train_dataloader = DataLoader(
        TensorDataset(train_users_torch, train_movies_torch, train_predictions_torch),
        batch_size=batch_size,
    )

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
                train_rmse = data_processing.get_score(train_score_hat.cpu().numpy(), train_predictions)
                if do_validate:
                    test_score_hat = model(test_users_torch, test_movies_torch)
                    test_rmse = data_processing.get_score(test_score_hat.cpu().numpy(), test_predictions)
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
