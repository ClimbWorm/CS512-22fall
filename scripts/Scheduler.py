import torch
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scripts.utils import DEVICE, LOG_PATH, EarlyStopper, compute_mae_loss
from typing import Optional, Union, Any, Dict, Tuple
from model.DCRNN import DCRNN
from tqdm import tqdm


class SensorDataloader:
    def __init__(self, features, labels, batch_size, pad: bool = True, shuffle: bool = True):
        self.batch_size = batch_size
        if pad:
            # pad the last batch using the last sample
            pad_size = (batch_size - (len(features) % batch_size)) % batch_size
            self.features = np.concatenate([features, np.repeat(features[-1:], pad_size, axis=0)], axis=0)
            self.labels = np.concatenate([labels, np.repeat(labels[-1:], pad_size, axis=0)], axis=0)
        else:
            self.features = features
            self.labels = labels
        self.num_rows = len(self.features)  # after padding
        if shuffle:
            perm = np.random.permutation(self.num_rows)
            self.features, self.labels = self.features[perm], self.labels[perm]
        self.num_batches = self.num_rows // self.batch_size

    def gen_sample(self):
        for b in range(self.num_batches):
            st, ed = b * self.batch_size, (b + 1) * self.batch_size
            yield self.features[st: ed, ...], self.labels[st: ed, ...]

    def __iter__(self):
        return self.gen_sample()


class TrainScheduler:
    def __init__(self, model: Union[DCRNN, str], train_loader: SensorDataloader, val_loader: SensorDataloader,
                 test_loader: SensorDataloader, input_dim: int, output_dim: int, horizon: int, seq_size: int,
                 num_sensors: int, std, mean):
        self.model: DCRNN = model if type(model) is not str else self.load_model(model)
        self.model.to(DEVICE)
        # self.writer = SummaryWriter(f"runs/{LOG_PATH}")
        self.train_loader: SensorDataloader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = train_loader.batch_size
        self.horizon = horizon
        self.seq_size = seq_size
        self.num_sensors = num_sensors
        self.std = std
        self.mean = mean

    def inv_transform(self, data):
        return (data * self.std) + self.mean

    @staticmethod
    def load_model(model_path: str) -> DCRNN:
        return DCRNN().load_state_dict(torch.load(model_path))

    def save_model(self, model_path: str):
        torch.save(self.model.state_dict(), model_path)

    def format_input(self, feature, label):
        feature = torch.from_numpy(feature).float().permute(1, 0, 2, 3).view(self.seq_size, self.batch_size,
                                                                             self.num_sensors * self.input_dim)
        label = torch.from_numpy(label).float().permute(1, 0, 2, 3)[..., :self.output_dim].view(self.horizon,
                                                                                                self.batch_size,
                                                                                                self.num_sensors * self.output_dim)
        return feature.to(DEVICE), label.to(DEVICE)

    def compute_loss(self, pred, label):
        return compute_mae_loss(self.inv_transform(pred), self.inv_transform(label))

    def train(self, epoch: int = 20, lr: float = 0.001, eps: float = 1e-8, steps: Tuple[int] = (20, 40),
              lr_decay: float = 0.1, grad_clipping: float = 1, early_stop: Optional[EarlyStopper] = None):
        if self.train_loader is None:
            raise RuntimeError("DataLoader not specified. call config_dataloader first")
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=eps)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=opt, milestones=steps, gamma=lr_decay)
        for e in range(epoch):
            self.model.train()
            losses = []
            for batch, (feature, label) in enumerate(tqdm(self.train_loader, desc=f"Epoch {e}/Train: ")):
                feature, label = self.format_input(feature, label)
                opt.zero_grad()
                out = self.model(feature, label, batch)
                if batch == 0:
                    # FIXME what is this??
                    opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=eps)
                loss = self.compute_loss(out, label)
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), grad_clipping)
                opt.step()
            lr_scheduler.step()
            val_loss = self.validate(e)
            if early_stop is not None and early_stop.early_stop(val_loss):
                break

    def validate(self, epoch) -> float:
        with torch.no_grad():
            self.model.eval()
            losses = []
            for batch, (feature, label) in enumerate(tqdm(self.val_loader, desc=f"Epoch {epoch}/validate: ")):
                feature, label = self.format_input(feature, label)
                out = self.model(feature)
                loss = self.compute_loss(out, label)
                losses.append(loss.item())
        return np.mean(losses)
