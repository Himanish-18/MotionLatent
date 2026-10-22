"""
Representation models package -- Phase 3.
"""

from src.models.baseline import RawBaselineRepresentation
from src.models.pca_model import PCARepresentation
from src.models.ica_model import ICARepresentation
from src.models.autoencoder import MotionAutoencoder, AutoencoderRepresentation
from src.models.temporal_autoencoder import (
    Conv1DAutoencoderRepresentation,
    DSCAutoencoderRepresentation,
)

__all__ = [
    "RawBaselineRepresentation",
    "PCARepresentation",
    "ICARepresentation",
    "MotionAutoencoder",
    "AutoencoderRepresentation",
    "Conv1DAutoencoderRepresentation",
    "DSCAutoencoderRepresentation",
]
