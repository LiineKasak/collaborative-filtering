import comet_ml
from comet_ml import Experiment

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

from auxiliary import data_processing

eps = 1e-6

# Return user and item biases
def compute_baselines_als(data, mask, n_epochs=10):
    num_users, num_items = data.shape()
    num_entries = len(users)
    global_mean = np.mean(predictions)
    for e in n_epochs:
        for idx_u in num_users:
            for idx_v in num_items:
                r = data[idx_u, idx_i]




def plot_curve(loss_curve):
    plt.plot(loss_curve)
    plt.title('Loss curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')


def non_negative_matrix_factorization(X, max_iterations, k):
    K = k  # Number of features
    X = X.float()

    # Initialize W and Z from a uniform distribution U(0, 1)
    # Additionally, the matrices are scaled by 1/sqrt(K) to make the variance of the resulting product independent of K
    U = torch.rand(X.shape[0], K).mul_(1 / math.sqrt(K)).requires_grad_()
    V = torch.rand(X.shape[1], K).mul_(1 / math.sqrt(K)).requires_grad_()

    optimizer = optim.SGD([U, V], lr=0.9, momentum=0.8)  # need momentum to degrease the loss faster
    loss_fn = nn.MSELoss()

    loss_curve = []
    for i in range(max_iterations):
        optimizer.zero_grad()
        loss = torch.sqrt(loss_fn(U @ V.t(), X) + eps)  # add eps in case of 0
        loss.backward()
        optimizer.step()

        # Project onto valid set of solutions
        U.data.clamp_(min=0)
        V.data.clamp_(min=0)

        loss_val = loss.item()
        loss_curve.append(loss_val)

    plot_curve(loss_curve)
    return U, V


def sgd_factorization(X, max_iterations, k):
    (n, p) = X.shape

    X = X.float()

    U = torch.rand(n, k).mul_(1 / math.sqrt(k)).requires_grad_()
    V = torch.rand(p, k).mul_(1 / math.sqrt(k)).requires_grad_()

    # optimizer = optim.SGD([U, V], lr=0.005)
    # loss_fn = nn.MSELoss()
    optimizer = optim.SGD([U, V], lr=0.5, momentum=0.95)  # need momentum to decrease the loss faster
    loss_fn = nn.MSELoss()
    loss_old = -1
    same = 0
    loss_curve = []
    with tqdm(range(max_iterations), desc='SGD_factorization', unit="batch") as pb:

        for i in pb:
            optimizer.zero_grad()
            loss = torch.sqrt(loss_fn(U @ V.t(), X) + eps)  # add eps in case of 0
            loss.backward()
            optimizer.step()


            if loss_old - loss.item() < 1e-10:  # early stopping
                same += 1
                if same > 50:
                    break

            loss_old = loss.item()


            pb.set_postfix(loss=loss.item())

            loss_curve.append(loss.item())

    return U, V


def als_factorization(X, max_iterations, k, experiment):
    (n, p) = X.shape

    X = X.float()

    U = torch.rand(n, k).mul_(1 / math.sqrt(k)).requires_grad_()
    V = torch.rand(p, k).mul_(1 / math.sqrt(k)).requires_grad_()

    # optimizer = optim.SGD([U, V], lr=0.005)
    # loss_fn = nn.MSELoss()
    optimizer_U = optim.Adam([U], lr=0.005)
    optimizer_V = optim.Adam([V], lr=0.005)

    loss_fn_U = nn.MSELoss()
    loss_fn_V = nn.MSELoss()




    loss_curve = []
    for i in tqdm(range(max_iterations), desc='ALS_factorization'):
        # optimize U
        optimizer_U.zero_grad()
        loss_u = torch.sqrt(loss_fn_U(U @ V.t(), X) + eps)  # add eps in case of 0
        loss_u.backward()
        optimizer_U.step()

        # optimize V
        optimizer_V.zero_grad()
        loss_v = torch.sqrt(loss_fn_V(U @ V.t(), X) + eps)  # add eps in case of 0
        loss_v.backward()
        optimizer_V.step()

        # Project onto valid set of solutions
        U.data.clamp_(min=0)
        V.data.clamp_(min=0)

        experiment.log_metrics(
            {
                "loss_u": loss_u,
                "loss_v": loss_v
            }
        )

    return U, V
