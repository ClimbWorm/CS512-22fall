import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
import torch
from scipy.sparse import linalg
import scipy.sparse as sp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = ""
CP_PATH = "checkpoints"


class EarlyStopper:
    """Reference: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch"""

    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def compute_mae_loss(pred, label):
    """Masked MAE loss"""
    mask = (label != 0).float()
    mask /= mask.mean()
    loss = torch.abs(pred - label) * mask
    loss[loss != loss] = 0  # nan to 0
    return loss.mean()


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


"""
Input: dataframe df, input time length, output(predict) time length
Output: train data x, result(label) y
"""


# This only works for time_in_day. Each time t= 5mins
# x_offsets: input data time length, ex: x_length = 12 = 12*5 = 60 mins, 
# x_offsets = [-11, -10, ..., 0].
# y_offsets:  output data time length, ex: y_length = 12 = 12*5 = 60 mins,
# y_offsets[1, 2, 3, ..., 12]. Predict next one hour.
def generate_data(df, x_length, y_length):
    n_samples, n_nodes = df.shape
    x_offsets = np.sort(np.arange(-x_length + 1, 1, 1))
    y_offsets = np.sort(np.arange(1, y_length + 1, 1))

    # Slice df by rows(different time slot)
    # This only contain the speed data for 319 sensors at the same time.
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]

    # Rewrite datetime format into numbers.
    # And pivoted the new time array.(a time data for each sensor)
    time_in_number = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_array = np.tile(time_in_number, [1, n_nodes, 1]).transpose((2, 1, 0))
    data_list.append(time_array)
    data = np.concatenate(data_list, axis=-1)

    # Split datalist by x,ys' offset.
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(n_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y
