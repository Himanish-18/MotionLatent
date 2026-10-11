"""
Nonlinear Autoencoder Representation -- Phase 3.

Implements a compact, reproducible neural autoencoder in PyTorch with
deterministic initialization, training loop, early stopping, and encoding/decoding
interfaces.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MotionAutoencoder(nn.Module):
    """Compact symmetric MLP Autoencoder for 6,912-d motion window vectors."""

    def __init__(self, input_dim: int = 6912, hidden_dim: int = 256, latent_dim: int = 16) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


class AutoencoderRepresentation:
    """Wrapper managing training, projection, and reconstruction for MotionAutoencoder."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        input_dim: int = 6912,
        lr: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 15,
        patience: int = 3,
        random_seed: int = 42,
    ) -> None:
        self.name = f"Autoencoder (d={latent_dim})"
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_seed = random_seed

        # Set seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.device = torch.device("cpu")
        self.model = MotionAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(self.device)

        self.is_fitted = False
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
    ) -> "AutoencoderRepresentation":
        """Train Autoencoder exclusively on training data with early stopping."""
        torch.manual_seed(self.random_seed)
        self.model.train()

        # If no validation set provided, split off 10% of training data deterministically
        if X_val is None:
            n_val = max(1, int(len(X_train) * 0.10))
            train_data = torch.from_numpy(X_train[:-n_val].astype(np.float32))
            val_data = torch.from_numpy(X_train[-n_val:].astype(np.float32))
        else:
            train_data = torch.from_numpy(X_train.astype(np.float32))
            val_data = torch.from_numpy(X_val.astype(np.float32))

        train_loader = DataLoader(
            TensorDataset(train_data),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training epoch
            self.model.train()
            running_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(batch_x)

            epoch_train_loss = running_loss / len(train_data)
            self.train_losses.append(float(epoch_train_loss))

            # Validation evaluation
            self.model.eval()
            with torch.no_grad():
                val_x = val_data.to(self.device)
                val_recon = self.model(val_x)
                val_loss = criterion(val_recon, val_x).item()
                self.val_losses.append(float(val_loss))

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode input into latent space (N, d)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform.")
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
            # Evaluate in batches if large
            if len(X) > 512:
                latents = []
                for i in range(0, len(X), 512):
                    latents.append(self.model.encode(x_tensor[i : i + 512]).cpu().numpy())
                return np.concatenate(latents, axis=0)
            return self.model.encode(x_tensor).cpu().numpy().astype(np.float32)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Decode latent codes back to input space (N, 6912)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before inverse_transform.")
        self.model.eval()
        with torch.no_grad():
            z_tensor = torch.from_numpy(Z.astype(np.float32)).to(self.device)
            if len(Z) > 512:
                recons = []
                for i in range(0, len(Z), 512):
                    recons.append(self.model.decode(z_tensor[i : i + 512]).cpu().numpy())
                return np.concatenate(recons, axis=0)
            return self.model.decode(z_tensor).cpu().numpy().astype(np.float32)

    def get_param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def get_model_size_bytes(self) -> int:
        """Model weight size in bytes (float32)."""
        return self.get_param_count() * 4

    def benchmark_latency(self, X: np.ndarray, n_repeats: int = 5) -> float:
        """Measure average encoding latency in milliseconds per window."""
        self.model.eval()
        x_sample = torch.from_numpy(X[: min(100, len(X))].astype(np.float32)).to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_repeats):
                _ = self.model.encode(x_sample)
        dt = (time.perf_counter() - t0) / (n_repeats * len(x_sample))
        return float(dt * 1000.0)  # ms per window
