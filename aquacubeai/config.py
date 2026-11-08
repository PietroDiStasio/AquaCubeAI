"""Paper settings and portable paths. Importing this module performs no I/O."""

import random
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch

from .paths import ROOT, configuration_file, data_directory


def set_seed(seed=5):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    seed: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tur_min: float = 0.0
    tur_max: float = 140.0
    cell_m: float = 15000.0
    train_perc: float = 0.60
    val_perc: float = 0.20
    test_perc: float = 0.20
    split_restarts: int = 4000
    split_seed: int = 5
    split_mean_weight: float = 2.0
    split_size_tolerance: float = 0.02
    augment_enable: bool = False
    pca_enable: bool = False
    pca_components: int = 4
    normalization_enable: bool = True
    norm_params_path: str = str(ROOT / "outputs/training/norm_params_tur.npz")
    hidden_layers: Tuple[int, ...] = (512, 512, 256)
    lr: float = 1e-4
    batch_size: int = 2048
    n_epochs: int = 2000
    early_stop_patience: int = 200
    grad_clip: float = 0.5
    huber_delta: float = 1.0
    best_model_path: str = str(ROOT / "outputs/training/best_model.pth")
    patch_hw: int = 20
    verbose: bool = False
