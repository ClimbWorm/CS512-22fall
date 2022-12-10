import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
import torch
from scipy.sparse import linalg
import scipy.sparse as sp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = ""


def gen_adj_mat(dist: pd.DataFrame, sensor_ids: List[int], k: float = 0.1) -> Tuple[Dict[int, int], np.ndarray]:
    """Generate the adjacent matrix and sensor id mapping based on the data.
    :param dist: DataFrame of distance. Columns: [from(sensor id), to(sensor id), distance(float)]
    :param sensor_ids: list of sensor ids
    :param k: Threshold to set element i to 0 if i < k after norm
    :return: Dict map the sensor id to its index in the adj_mat, and the adj_mat
    """
    num_sensors = len(sensor_ids)
    dist_mat: np.ndarray = np.inf * np.ones(num_sensors, num_sensors)
    sensor_dict = {sensor_id: idx for idx, sensor_id in enumerate(sensor_ids)}
    for from_id, to_id, distance in dist.iterrows():
        if from_id in sensor_dict and to_id in sensor_dict:
            dist_mat[sensor_dict[from_id], sensor_dict[to_id]] = distance

    std = dist_mat[~np.isinf(dist_mat)].flatten().std()
    adj_mat = np.exp(-np.square(dist_mat / std))  # this adj_mat is not symmetric
    adj_mat[adj_mat < k] = 0
    return sensor_dict, adj_mat


# --------following need to be rewritten----------
def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)
