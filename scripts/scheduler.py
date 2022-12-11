import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scripts.utils import DEVICE, LOG_PATH, EarlyStopper, compute_mae_loss, CP_PATH, SensorDataloader
from typing import Optional, Union, Any, Dict, Tuple
from model.DCRNN import DCRNN
from tqdm import tqdm


class TrainScheduler:
    def __init__(self, adj_mat: np.ndarray, train_loader: SensorDataloader, val_loader: SensorDataloader,
                 test_loader: SensorDataloader, input_dim: int, output_dim: int, horizon: int, seq_size: int,
                 num_sensors: int, std, mean, trained_epoch: int = 0):
        self.model: DCRNN = DCRNN(adj_mat, {}).to(DEVICE)
        if trained_epoch != 0:
            self.load_model(trained_epoch)
        self.writer = SummaryWriter(f"runs/{LOG_PATH}")
        self.dataloader = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = train_loader.batch_size
        self.horizon = horizon
        self.seq_size = seq_size
        self.num_sensors = num_sensors
        self.std = std
        self.mean = mean
        self.trained_batch = trained_epoch * train_loader.num_batches

    def inv_transform(self, data):
        return (data * self.std) + self.mean

    def load_model(self, epoch: int):
        return self.model.load_state_dict(torch.load(f"{CP_PATH}/{epoch}.pth", map_location="cpu"))

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

    def train(self, epoch: int = 100, lr: float = 0.01, eps: float = 1e-3, steps: Tuple[int] = (20, 30, 40, 50),
              lr_decay: float = 0.1, grad_clipping: float = 3, early_stop: Optional[EarlyStopper] = None,
              save_per_epoch: int = 3, test_per_epoch: int = 5):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=eps)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=opt, milestones=steps, gamma=lr_decay)
        for e in range(epoch):
            self.model.train()
            losses = []
            batch_sofar = self.batch_size
            for feature, label in tqdm(self.dataloader["train"], desc=f"Epoch {e}/Train: "):
                feature, label = self.format_input(feature, label)
                opt.zero_grad()
                out = self.model(feature, label, batch_sofar)
                if batch_sofar == 0:
                    # FIXME what is this??
                    opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=eps)
                loss = self.compute_loss(out, label)
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), grad_clipping)
                opt.step()
            batch_sofar += self.dataloader["train"].num_batches
            lr_scheduler.step()
            val_loss = self.evaluate(e, "val")
            self.writer.add_scalar("Validation loss", val_loss, batch_sofar)
            self.writer.add_scalar("Train loss", np.mean(losses), batch_sofar)
            if early_stop is not None and early_stop.early_stop(val_loss):
                break
            if e % test_per_epoch + 1 == test_per_epoch:
                test_loss = self.evaluate(e, "test")
                self.writer.add_scalar("Test loss", test_loss, batch_sofar)
            if e % save_per_epoch + 1 == save_per_epoch:
                self.save_model(f"checkpoints/epoch_{e}.pth")

    def evaluate(self, epoch, dataset: str) -> float:
        with torch.no_grad():
            self.model.eval()
            losses = []
            for batch, (feature, label) in enumerate(tqdm(self.dataloader[dataset], desc=f"Epoch {epoch}/{dataset}: ")):
                feature, label = self.format_input(feature, label)
                out = self.model(feature)
                loss = self.compute_loss(out, label)
                losses.append(loss.item())
        return np.mean(losses)
