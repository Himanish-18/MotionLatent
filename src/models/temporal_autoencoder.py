"""
Lightweight Temporal Autoencoders -- Phase 4.

Implements compact temporal convolutional autoencoders for resource-constrained
human activity recognition (PAMAP2 27-channel IMU windows of length 256):
1. Conv1DAutoencoder: Standard 1D Temporal Convolutional Autoencoder.
2. DSCAutoencoder: Depthwise-Separable 1D Temporal Convolutional Autoencoder (MobileNet-style).

Both models:
- Train unsupervised using Mean Squared Error (MSE) reconstruction loss.
- Report encoder-only vs full-model parameters and memory footprints.
- Support deterministic training, early stopping, and standard projection interfaces.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Neural Modules: Conv1D and Depthwise-Separable Conv1D
# ---------------------------------------------------------------------------

class Conv1DEncoder(nn.Module):
    """Temporal 1D Convolutional Encoder."""

    def __init__(self, in_channels: int = 27, latent_dim: int = 16) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            # Stage 1: (27, 256) -> (32, 128)
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Stage 2: (32, 128) -> (64, 64)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Stage 3: (64, 64) -> (64, 32)
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Global temporal pooling -> (64, 1) -> (64,)
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Conv1DDecoder(nn.Module):
    """Temporal 1D Convolutional Decoder."""

    def __init__(self, out_channels: int = 27, latent_dim: int = 16) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 64 * 32)
        self.deconv = nn.Sequential(
            # Stage 1: (64, 32) -> (64, 64)
            nn.ConvTranspose1d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Stage 2: (64, 64) -> (32, 128)
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Stage 3: (32, 128) -> (27, 256)
            nn.ConvTranspose1d(32, out_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 64, 32)
        return self.deconv(h)


class DSCBlock(nn.Module):
    """Depthwise-Separable 1D Convolution Block."""

    def __init__(self, in_c: int, out_c: int, k: int = 5, s: int = 2, p: int = 2) -> None:
        super().__init__()
        self.dw = nn.Conv1d(in_c, in_c, kernel_size=k, stride=s, padding=p, groups=in_c, bias=False)
        self.pw = nn.Conv1d(in_c, out_c, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class DSCEncoder(nn.Module):
    """Depthwise-Separable Temporal 1D Convolutional Encoder."""

    def __init__(self, in_channels: int = 27, latent_dim: int = 16) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            # Stage 1: (27, 256) -> (32, 128)
            DSCBlock(in_channels, 32, k=7, s=2, p=3),
            # Stage 2: (32, 128) -> (64, 64)
            DSCBlock(32, 64, k=5, s=2, p=2),
            # Stage 3: (64, 64) -> (64, 32)
            DSCBlock(64, 64, k=5, s=2, p=2),
            # Global temporal pooling -> (64,)
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DSCDecoder(nn.Module):
    """Decoder paired with Depthwise-Separable Encoder."""

    def __init__(self, out_channels: int = 27, latent_dim: int = 16) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 64 * 32)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, out_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 64, 32)
        return self.deconv(h)


# ---------------------------------------------------------------------------
# High-Level Wrapper Classes
# ---------------------------------------------------------------------------

class BaseTemporalAutoencoder:
    """Base class for temporal autoencoder representations."""

    def __init__(
        self,
        name: str,
        latent_dim: int = 16,
        in_channels: int = 27,
        seq_len: int = 256,
        lr: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 15,
        patience: int = 3,
        random_seed: int = 42,
    ) -> None:
        self.name = name
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.flat_dim = in_channels * seq_len
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_seed = random_seed

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.device = torch.device("cpu")
        self.encoder: nn.Module
        self.decoder: nn.Module
        self.is_fitted = False
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        """Convert input array (N, 6912) or (N, 256, 27) into (N, 27, 256) float tensor."""
        if X.ndim == 2 and X.shape[1] == self.flat_dim:
            # (N, 6912) -> (N, 256, 27) -> (N, 27, 256)
            X_reshaped = X.reshape(-1, self.seq_len, self.in_channels).transpose(0, 2, 1)
        elif X.ndim == 3 and X.shape[1] == self.seq_len and X.shape[2] == self.in_channels:
            X_reshaped = X.transpose(0, 2, 1)
        elif X.ndim == 3 and X.shape[1] == self.in_channels and X.shape[2] == self.seq_len:
            X_reshaped = X
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")
        return torch.from_numpy(np.ascontiguousarray(X_reshaped, dtype=np.float32)).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
    ) -> "BaseTemporalAutoencoder":
        """Fit autoencoder on training data with validation early stopping."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        X_t = self._to_tensor(X_train)

        # Subject-aware / deterministic validation slice if not provided
        if X_val is None:
            n_total = len(X_t)
            n_val = max(1, int(0.10 * n_total))
            n_train = n_total - n_val
            train_t = X_t[:n_train]
            val_t = X_t[n_train:]
        else:
            train_t = X_t
            val_t = self._to_tensor(X_val)

        train_loader = DataLoader(
            TensorDataset(train_t),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_enc_state = None
        best_dec_state = None
        patience_counter = 0

        self.train_losses.clear()
        self.val_losses.clear()

        for epoch in range(self.max_epochs):
            self.encoder.train()
            self.decoder.train()
            epoch_loss = 0.0

            for (batch_x,) in train_loader:
                optimizer.zero_grad()
                z = self.encoder(batch_x)
                recon = self.decoder(z)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            avg_train_loss = epoch_loss / len(train_t)
            self.train_losses.append(avg_train_loss)

            # Validation loss
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                z_val = self.encoder(val_t)
                recon_val = self.decoder(z_val)
                avg_val_loss = criterion(recon_val, val_t).item()
                self.val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss - 1e-4:
                best_val_loss = avg_val_loss
                best_enc_state = {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()}
                best_dec_state = {k: v.cpu().clone() for k, v in self.decoder.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_enc_state is not None:
            self.encoder.load_state_dict(best_enc_state)
            self.decoder.load_state_dict(best_dec_state)

        self.encoder.eval()
        self.decoder.eval()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode input observations into latent codes Z (N, latent_dim)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform.")
        self.encoder.eval()
        X_t = self._to_tensor(X)
        with torch.no_grad():
            z = self.encoder(X_t)
        return z.cpu().numpy().astype(np.float32)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct 6,912-d flattened observations from latent codes Z."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before inverse_transform.")
        self.decoder.eval()
        z_t = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            recon = self.decoder(z_t)  # (N, 27, 256)
        # Transpose back to (N, 256, 27) and flatten to (N, 6912)
        flat = recon.cpu().numpy().transpose(0, 2, 1).reshape(-1, self.flat_dim)
        return flat.astype(np.float32)

    def get_param_count(self, encoder_only: bool = True) -> int:
        """Return total trainable parameter count."""
        if encoder_only:
            return sum(p.numel() for p in self.encoder.parameters())
        return sum(p.numel() for p in self.encoder.parameters()) + sum(p.numel() for p in self.decoder.parameters())

    def get_model_size_bytes(self, encoder_only: bool = True) -> int:
        """Return parameter storage size in bytes (float32)."""
        return self.get_param_count(encoder_only=encoder_only) * 4

    def benchmark_latency(
        self,
        X: np.ndarray,
        n_repeats: int = 5,
        encoder_only: bool = True,
    ) -> float:
        """Measure average latency in milliseconds per window."""
        self.encoder.eval()
        self.decoder.eval()
        X_t = self._to_tensor(X[:min(len(X), 200)])
        n_samples = len(X_t)

        with torch.no_grad():
            # Warmup
            for _ in range(2):
                z = self.encoder(X_t)
                if not encoder_only:
                    _ = self.decoder(z)

            t0 = time.perf_counter()
            for _ in range(n_repeats):
                z = self.encoder(X_t)
                if not encoder_only:
                    _ = self.decoder(z)
            dt = (time.perf_counter() - t0) / (n_repeats * n_samples)
        return float(dt * 1000.0)


class Conv1DAutoencoderRepresentation(BaseTemporalAutoencoder):
    """1D Convolutional Autoencoder Representation."""

    def __init__(
        self,
        latent_dim: int = 16,
        in_channels: int = 27,
        seq_len: int = 256,
        lr: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 12,
        patience: int = 3,
        random_seed: int = 42,
    ) -> None:
        super().__init__(
            name=f"Conv1D-AE (d={latent_dim})",
            latent_dim=latent_dim,
            in_channels=in_channels,
            seq_len=seq_len,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_seed=random_seed,
        )
        self.encoder = Conv1DEncoder(in_channels=in_channels, latent_dim=latent_dim).to(self.device)
        self.decoder = Conv1DDecoder(out_channels=in_channels, latent_dim=latent_dim).to(self.device)


class DSCAutoencoderRepresentation(BaseTemporalAutoencoder):
    """Depthwise-Separable 1D Convolutional Autoencoder Representation."""

    def __init__(
        self,
        latent_dim: int = 16,
        in_channels: int = 27,
        seq_len: int = 256,
        lr: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 12,
        patience: int = 3,
        random_seed: int = 42,
    ) -> None:
        super().__init__(
            name=f"DSC-AE (d={latent_dim})",
            latent_dim=latent_dim,
            in_channels=in_channels,
            seq_len=seq_len,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_seed=random_seed,
        )
        self.encoder = DSCEncoder(in_channels=in_channels, latent_dim=latent_dim).to(self.device)
        self.decoder = DSCDecoder(out_channels=in_channels, latent_dim=latent_dim).to(self.device)
