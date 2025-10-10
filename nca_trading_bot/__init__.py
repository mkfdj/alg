"""
Neural Cellular Automata Trading Bot

Adaptive AI-powered trading with self-growing neural cellular automata.
"""

__version__ = "0.1.0"
__author__ = "NCA Trading Bot Team"
__email__ = "contact@nca-trading-bot.com"

from .config import Config
from .nca_model import AdaptiveNCA
from .data_handler import DataHandler
from .trader import TradingEnvironment
from .trainer import PPOTrainer

__all__ = [
    "Config",
    "AdaptiveNCA",
    "DataHandler",
    "TradingEnvironment",
    "PPOTrainer"
]