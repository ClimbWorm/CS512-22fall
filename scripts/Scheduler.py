import torch
from torch.utils.data.dataset import T_co
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from scripts.utils import DEVICE, LOG_PATH
from model.DCRNN import DCRNN


class SensorDataset(IterableDataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels

    def __iter__(self):
        return iter(zip(self.features, self.labels))


class TrainScheduler:
    def __init__(self, model: DCRNN, dataloader: DataLoader):
        self.model: DCRNN = model
        # self.writer = SummaryWriter(f"runs/{LOG_PATH}")
        self.data = dataloader
