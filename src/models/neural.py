import time
import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# 1. PyTorch Tabular MLP Architecture
# -------------------------------------------------------------
class ChessTabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], num_classes: int = 4, dropout: float = 0.25):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.BatchNorm1d(h_d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_d = h_d
        layers.append(nn.Linear(in_d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------------------------------------
# 2. PyTorch 8x8x12 ResNet CNN Architecture
# -------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))


class ChessResNetCNN(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResBlock(64)
        self.trans = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.res2 = ResBlock(128)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.trans(x)
        x = self.res2(x)
        return self.head(x)


# -------------------------------------------------------------
# 3. Model Wrappers with Fit / Predict / Early Stopping
# -------------------------------------------------------------
class NeuralModelWrapper:
    def __init__(self, name: str, model: nn.Module, device: str = "cpu"):
        self.name = name
        self.model = model.to(device)
        self.device = device
        self.training_time_sec = 0.0
        self.inference_time_ms = 0.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 25,
        batch_size: int = 128,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 5,
    ) -> "NeuralModelWrapper":
        t0 = time.time()
        self.model.train()

        # Compute pos_weight for BCEWithLogits to counteract label imbalance
        pos_counts = np.maximum(y_train.sum(axis=0), 1)
        neg_counts = len(y_train) - pos_counts
        pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        t_X_train = torch.tensor(X_train, dtype=torch.float32)
        t_y_train = torch.tensor(y_train, dtype=torch.float32)
        t_X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        t_y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(t_X_train, t_y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(batch_x)

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(t_X_val)
                val_loss = criterion(val_logits, t_y_val).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"{self.name}: Early stopping triggered at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.training_time_sec = time.time() - t0
        logger.info(f"Trained {self.name} in {self.training_time_sec:.2f}s (Best val loss: {best_val_loss:.4f})")
        return self

    def predict_proba(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        t0 = time.time()
        self.model.eval()
        t_X = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(t_X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        probs_list = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probs = torch.sigmoid(logits)
                probs_list.append(probs.cpu().numpy())

        all_probs = np.vstack(probs_list)
        self.inference_time_ms = (time.time() - t0) * 1000.0 / max(1, len(X))
        return np.clip(all_probs, 0.0, 1.0)
