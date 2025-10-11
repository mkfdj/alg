"""
TRM Coding Agent - Tiny Recursive Model for Code Generation and Refinement

This package implements a specialized AI system for recursive code generation
based on the paper "Less is More: Recursive Reasoning with Tiny Networks"
(arXiv:2510.04871), adapted specifically for coding tasks with binary
thinking logic and minimal recursive algorithms.

Main Components:
- TRM Model: Tiny recursive model with binary decision logic
- Data Handler: Multi-dataset loading and processing
- Coder Environment: Code generation and tool simulation
- Trainer: PPO fine-tuning with distributed TPU support
- Utils: JAX setup, tokenization, and helper functions

Usage:
    from trm_coding_agent import TRMModel, DataHandler, Coder, Trainer

    # Initialize configuration
    config = get_config("tpu")

    # Load data
    data_handler = DataHandler(config)
    train_data, val_data = data_handler.load_datasets(["humaneval", "mbpp"])

    # Initialize model
    model = TRMModel(config)

    # Train
    trainer = Trainer(model, config)
    trainer.train(train_data, val_data)

License: MIT
"""

__version__ = "1.0.0"
__author__ = "TRM Coding Agent Team"
__email__ = "contact@trm-coding-agent.ai"

# Import main components
from .config import Config, get_config, DEFAULT_CONFIG
from .trm_model import TRMModel
from .data_handler import DataHandler
from .coder import CoderEnvironment
from .trainer import Trainer
from .utils import setup_tpu, binarize, simulate_code_execution

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",

    # Configuration
    "Config",
    "get_config",
    "DEFAULT_CONFIG",

    # Main components
    "TRMModel",
    "DataHandler",
    "CoderEnvironment",
    "Trainer",

    # Utilities
    "setup_tpu",
    "binarize",
    "simulate_code_execution",
]


def get_package_info():
    """Get package information."""
    return {
        "name": "trm_coding_agent",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "Tiny Recursive Model for Code Generation with Binary Thinking",
        "url": "https://github.com/trm-coding-agent/trm-coding-agent",
        "license": "MIT",
        "python_requires": ">=3.10",
        "install_requires": [
            "jax[tpu]>=0.4.0",
            "flax>=0.8.0",
            "optax>=0.2.0",
            "chex>=0.1.0",
            "tokenizers>=0.15.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "tqdm>=4.65.0",
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
            "lance>=0.8.0",
            "pyarrow>=12.0.0",
            "timeout-decorator>=0.5.0"
        ]
    }


def check_dependencies():
    """Check if all required dependencies are installed."""
    import importlib
    import sys

    required_packages = [
        "jax",
        "flax",
        "optax",
        "chex",
        "tokenizers",
        "pandas",
        "numpy",
        "tqdm"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False

    return True


def setup_environment(config: Config = None):
    """Setup the TRM Coding Agent environment."""
    if config is None:
        config = DEFAULT_CONFIG

    # Check dependencies
    if not check_dependencies():
        raise ImportError("Missing required dependencies")

    # Setup JAX/TPU
    if config.trm.use_tpu:
        setup_tpu()

    # Setup logging
    import logging
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level),
        format=config.logging.log_format,
        filename=config.logging.log_file,
        filemode='a'
    )

    logger = logging.getLogger(__name__)
    logger.info(f"TRM Coding Agent v{__version__} initialized")
    logger.info(f"Configuration: {config.to_dict()}")

    return logger


# Auto-setup on import
if __name__ != "__main__":
    try:
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
    except:
        pass