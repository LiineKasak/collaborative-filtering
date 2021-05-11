import numpy as np
import math
from tqdm import tqdm
from helpers import data_processing


def shrink(Y, tau):
    U, S, V = np.linalg.svd(Y, full_matrices=False)
    S = np.maximum(S - tau, 0)
    X = U * S @ V

    return X


# map X to the space of observed entries (Omega)
def map_to_omega(X, mask):
    return mask * X


def svd_completion(input_matrix, k_singular_values):
    U, s, Vt = np.linalg.svd(input_matrix, full_matrices=False)
    number_of_users, number_of_movies = (data_processing.get_number_of_users(), data_processing.get_number_of_movies())
    number_of_singular_values = min(number_of_users, number_of_movies)
    assert (k_singular_values <= number_of_singular_values), "choose correct number of singular values"  # only accept a number smaller than the actual number of singular values

    S = np.zeros((number_of_users, number_of_movies))
    S[:k_singular_values, :k_singular_values] = np.diag(s[:k_singular_values])  # only use k singular values

    reconstructed_matrix = U * S @ Vt
    return reconstructed_matrix


# Singular Value Thresholding
# adapted from
# https://github.com/psh150204/matrix-completion/blob/d875b5280b49fd6e075a27a2be8793f7d9b9d0bc/mat_comp.py#L3
def svt(input_matrix, mask, max_iterations, eps=0.10, disable=False):
    delta = 1.2 * np.prod(input_matrix.shape) / np.sum(mask)
    tau = 5 * np.sum(input_matrix.shape) / 2
    Y = np.zeros_like(input_matrix)

    for k in tqdm(range(max_iterations), desc='svt', disable=disable):
        X = shrink(Y, tau)
        Y += delta * map_to_omega(input_matrix - X, mask)

        recon_error = np.linalg.norm(map_to_omega(X - input_matrix, mask)) / np.linalg.norm(map_to_omega(input_matrix, mask))

        if recon_error < eps:
            print("svt terminated early, at k = ", k)
            break

    return X


# Adaptive Singular Value Thresholding
# This is my own version, and it is kept as similar to SVT as possible
# (e.g. it uses a for-loop unlike the paper, which uses while)
def asvt(input_matrix, mask, max_iterations, a, b, eps=0.10, disable=False):
    delta = 1.2 * np.prod(input_matrix.shape) / np.sum(mask)  # TODO: vary deltas
    deltas = np.full(shape=max_iterations, fill_value=delta)
    Y = np.zeros_like(input_matrix)

    for k in tqdm(range(max_iterations), desc='asvt', disable=disable):
        tau_k = b * math.exp(-a * k)
        X = shrink(Y, tau_k)
        Y += deltas[k] * map_to_omega(input_matrix - X, mask)

        recon_error = np.linalg.norm(map_to_omega(X - input_matrix, mask)) / np.linalg.norm(map_to_omega(input_matrix, mask))

        if recon_error < eps:
            print("asvt terminated early, at k = ", k)
            break

    return X
