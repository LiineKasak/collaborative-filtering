import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

from helpers import data_processing

eps = 1e-6


def plot_curve(loss_curve):
    plt.plot(loss_curve)
    plt.title('Loss curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')


def non_negative_matrix_factorization(X, max_iterations, k):
    K = 49  # Number of features
    X = X.float()

    # Initialize W and Z from a uniform distribution U(0, 1)
    # Additionally, the matrices are scaled by 1/sqrt(K) to make the variance of the resulting product independent of K
    U = torch.rand(X.shape[0], K).mul_(1 / math.sqrt(K)).requires_grad_()
    V = torch.rand(X.shape[1], K).mul_(1 / math.sqrt(K)).requires_grad_()

    optimizer = optim.Adam([U, V], lr=0.005)
    loss_fn = nn.MSELoss()

    loss_curve = []
    for i in range(5000):
        optimizer.zero_grad()
        loss = torch.sqrt(loss_fn(U @ V.t(), X) + eps) # add eps in case of 0
        loss.backward()
        optimizer.step()

        # Project onto valid set of solutions
        U.data.clamp_(min=0)
        V.data.clamp_(min=0)

        loss_val = loss.item()
        loss_curve.append(loss_val)

        if i % 1000 == 0:
            print(f'[{i}] loss: {loss_val:.05f}')


    plot_curve(loss_curve)
    return U, V

def sgd_factorization(X, max_iterations, k):
    (n, p) = X.shape

    X = X.float()

    U = torch.rand(n, k).mul_(1 / math.sqrt(k)).requires_grad_()
    V = torch.rand(p, k).mul_(1 / math.sqrt(k)).requires_grad_()

    # optimizer = optim.SGD([U, V], lr=0.005)
    # loss_fn = nn.MSELoss()
    optimizer = optim.SGD([U, V], lr=0.9, momentum=0.8) # need momentum to degrease the loss faster
    loss_fn = nn.MSELoss()

    loss_curve = []
    for i in tqdm(range(max_iterations), desc='SGD_factorization'):
        optimizer.zero_grad()
        loss = torch.sqrt(loss_fn(U @ V.t(), X) + eps) # add eps in case of 0

        loss.backward()
        optimizer.step()

        loss_curve.append(loss.item())
        if (i % (max_iterations/10) == 0):
            print("loss: ", loss.item())
    return U,V

def als_factorization(X, max_iterations, k):
    (n, p) = X.shape

    X = X.float()

    U = torch.rand(n, k).mul_(1 / math.sqrt(k)).requires_grad_()
    V = torch.rand(p, k).mul_(1 / math.sqrt(k)).requires_grad_()

    # optimizer = optim.SGD([U, V], lr=0.005)
    # loss_fn = nn.MSELoss()
    optimizer = optim.Adam([U, V], lr=0.005)
    loss_fn = nn.MSELoss()

    loss_curve = []
    for i in tqdm(range(max_iterations), desc='SGD_factorization'):
        optimizer.zero_grad()
        loss = torch.sqrt(loss_fn(U @ V.t(), X) + eps) # add eps in case of 0

        loss.backward()
        optimizer.step()

        loss_curve.append(loss.item())

        if (i % 1000 == 0):
            print("loss: ", loss.item())
    return U,V