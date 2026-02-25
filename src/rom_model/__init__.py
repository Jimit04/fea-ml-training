"""ROM model definitions and training loop.

Contains the custom ``GCNLayer``, factory functions for building MLP and GCN
Keras models, and the ``ROMTrainer`` class that orchestrates data loading,
training, evaluation, and model persistence.

All public symbols are re-exported here so existing imports like
``from src.rom_model import ROMTrainer`` continue to work.
"""

from src.rom_model.layers import GCNLayer
from src.rom_model.adjacency import build_beam_adjacency
from src.rom_model.architectures import build_mlp, build_gcn
from src.rom_model.trainer import ROMTrainer

__all__ = [
    "GCNLayer",
    "build_beam_adjacency",
    "build_mlp",
    "build_gcn",
    "ROMTrainer",
]
