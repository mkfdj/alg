"""
Coding Datasets Package
A comprehensive system for managing 18 coding datasets from Kaggle
"""

__version__ = "1.0.0"
__author__ = "Code Dataset Manager"

from .dataset_manager import DatasetManager
from .registry import DatasetRegistry

__all__ = ['DatasetManager', 'DatasetRegistry']