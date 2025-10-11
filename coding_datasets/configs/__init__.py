"""
Configuration management for coding datasets
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import ConfigManager
from default_configs import get_default_config, get_preprocessing_config, get_training_config

__all__ = [
    'ConfigManager',
    'get_default_config',
    'get_preprocessing_config',
    'get_training_config'
]