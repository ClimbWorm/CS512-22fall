import pandas as pd
from typing import List, Tuple, Dict
import numpy as np


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
