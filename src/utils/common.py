"""
Shared utilities for Insider Threat Detection pipeline.
"""

import sys
import yaml
import torch
import numpy as np
from pathlib import Path


def load_config(config_path=None):
    """Load YAML configuration."""
    if config_path is None:
        # Default to ../../config/config.yaml relative to this file
        # __file__ = src/utils/common.py
        # parent = src/utils
        # parent.parent = src
        # parent.parent.parent = project_root
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  Seed: {seed}")


def get_device():
    """Get torch device (GPU/CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"  CPU mode")
    return device
