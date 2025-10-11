"""
Configuration file for TRM Coding Agent.

This module contains all hyperparameters, model settings, and configuration
options for the Tiny Recursive Model coding agent.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import jax.numpy as jnp


@dataclass
class TRMConfig:
    """Configuration for Tiny Recursive Model architecture."""

    # Model Architecture
    hidden_size: int = 512
    num_layers: int = 2  # Tiny 2-layer architecture as per TRM paper
    intermediate_size: int = 2048  # MLP intermediate size
    vocab_size: int = 50257  # GPT-NeoX tokenizer vocab size
    max_position_embeddings: int = 2048

    # Recursive Reasoning Parameters
    recursion_depth: int = 16  # N_sup from TRM paper
    latent_dim: int = 256  # z dimension for reasoning
    binary_threshold: float = 0.5  # Threshold for binary decisions

    # Binary Neural Network Settings
    use_binary_activations: bool = True
    binary_binarization: bool = True  # Binarize embeddings and states
    gradient_free_steps: int = 15  # Steps without gradients

    # Training Hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    # Batch and Training Settings
    batch_size: int = 768  # From TRM paper
    max_samples_per_dataset: int = 10000
    num_epochs: int = 100
    eval_frequency: int = 1000

    # TPU/Distributed Settings
    use_tpu: bool = True
    num_tpu_chips: int = 8
    dtype: jnp.dtype = jnp.float16  # Memory efficiency

    # Early Stopping and ACT
    use_adaptive_computation_time: bool = True
    act_halting_threshold: float = 0.8
    min_recursion_steps: int = 4

    # Regularization
    dropout_rate: float = 0.1
    use_ema: bool = True
    ema_decay: float = 0.999
    use_stable_max_loss: bool = True

    # Tools and Environment
    enable_tool_simulation: bool = True
    max_tool_calls_per_step: int = 3
    tool_execution_timeout: int = 10  # seconds

    # Data Settings
    max_sequence_length: int = 1024
    prompt_max_length: int = 512
    code_max_length: int = 512

    # RL Fine-tuning (PPO)
    use_rl_finetuning: bool = False
    ppo_clip_ratio: float = 0.2
    ppo_value_loss_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95

    # Logging and Monitoring
    log_frequency: int = 100
    save_frequency: int = 5000
    tensorboard_log_dir: str = "logs/tensorboard"
    wandb_project: Optional[str] = None

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_model: bool = True
    keep_last_n_checkpoints: int = 5


@dataclass
class DatasetConfig:
    """Configuration for datasets."""

    # Dataset paths
    dataset_dir: str = "datasets"

    # Dataset configurations
    datasets: List[Dict[str, str]] = field(default_factory=lambda: [
        {
            "name": "humaneval",
            "path": "handcrafted-dataset-for-code-generation-models",
            "format": "csv",
            "url": "https://www.kaggle.com/datasets/thedevastator/handcrafted-dataset-for-code-generation-models",
            "enabled": True
        },
        {
            "name": "mbpp",
            "path": "mbppjsonl",
            "format": "jsonl",
            "url": "https://www.kaggle.com/datasets/mpwolke/mbppjsonl",
            "enabled": True
        },
        {
            "name": "codeparrot_1m",
            "path": "codeparrot-1m",
            "format": "lance",
            "url": "https://www.kaggle.com/datasets/heyytanay/codeparrot-1m",
            "enabled": False  # Disable by default due to size
        },
        {
            "name": "alpaca_python",
            "path": "python-code-instructions-18k-alpaca",
            "format": "json",
            "url": "https://www.kaggle.com/datasets/nikitakudriashov/python-code-instructions-18k-alpaca",
            "enabled": True
        },
        {
            "name": "glaive_qa",
            "path": "glaive-python-code-qa-dataset",
            "format": "csv",
            "url": "https://www.kaggle.com/datasets/thedevastator/glaive-python-code-qa-dataset",
            "enabled": True
        },
        {
            "name": "livecodebench",
            "path": "livecodebench",
            "format": "json",
            "url": "https://www.kaggle.com/datasets/open-benchmarks/livecodebench",
            "enabled": True
        }
    ])

    # Data augmentation
    use_data_augmentation: bool = True
    augmentation_factor: int = 3  # Number of augmented versions per sample

    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True

    # Validation split
    validation_split: float = 0.1
    test_split: float = 0.1

    # Binary thinking augmentation
    use_synthetic_binary_decisions: bool = True
    binary_decision_threshold: int = 200  # Character length threshold


@dataclass
class ToolConfig:
    """Configuration for tool simulation and execution."""

    # Available tools
    available_tools: List[str] = field(default_factory=lambda: [
        "code_execution",
        "syntax_checker",
        "import_resolver",
        "debugger",
        "test_runner",
        "documentation_search",
        "error_analyzer"
    ])

    # Tool simulation settings
    simulate_code_execution: bool = True
    simulate_web_search: bool = True
    safe_execution_sandbox: bool = True

    # Code execution settings
    execution_timeout: int = 10
    max_execution_memory: str = "512MB"
    allowed_imports: List[str] = field(default_factory=lambda: [
        "json", "math", "random", "datetime", "itertools", "collections",
        "functools", "operator", "re", "string", "typing"
    ])

    # Testing settings
    test_timeout: int = 5
    max_test_cases: int = 10
    generate_synthetic_tests: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""

    # Basic logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/trm_coding_agent.log"

    # Advanced monitoring
    use_tensorboard: bool = True
    use_wandb: bool = False
    use_mlflow: bool = False

    # Metrics to track
    metrics_to_track: List[str] = field(default_factory=lambda: [
        "train_loss",
        "val_loss",
        "recursion_depth",
        "binary_decision_ratio",
        "code_generation_success",
        "test_pass_rate",
        "grad_norm",
        "learning_rate"
    ])

    # Monitoring frequency
    log_frequency: int = 100
    eval_frequency: int = 1000
    checkpoint_frequency: int = 5000


@dataclass
class EnvironmentConfig:
    """Configuration for the coding environment."""

    # Environment settings
    max_episode_length: int = 100
    reward_shaping: bool = True

    # Reward structure
    syntax_correct_reward: float = 1.0
    test_pass_reward: float = 10.0
    syntax_error_penalty: float = -1.0
    runtime_error_penalty: float = -0.5
    timeout_penalty: float = -0.1

    # State representation
    include_prompt_history: bool = True
    include_execution_history: bool = True
    include_error_messages: bool = True
    max_history_length: int = 10


# Main configuration class
@dataclass
class Config:
    """Main configuration class combining all configurations."""

    # Sub-configurations
    trm: TRMConfig = field(default_factory=TRMConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tool: ToolConfig = field(default_factory=ToolConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    # Project settings
    project_name: str = "trm_coding_agent"
    project_version: str = "1.0.0"
    author: str = "TRM Coding Agent Team"

    # Paths
    base_dir: str = os.getcwd()
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    temp_dir: str = "temp"

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Debug settings
    debug_mode: bool = False
    verbose: bool = True
    profile_performance: bool = False

    def __post_init__(self):
        """Post-initialization setup."""
        # Create directories if they don't exist
        for dir_path in [
            self.output_dir,
            self.cache_dir,
            self.temp_dir,
            self.trm.checkpoint_dir,
            self.logging.tensorboard_log_dir,
            os.path.dirname(self.logging.log_file)
        ]:
            os.makedirs(dir_path, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create configuration from dictionary."""
        return cls(
            trm=TRMConfig(**config_dict.get('trm', {})),
            dataset=DatasetConfig(**config_dict.get('dataset', {})),
            tool=ToolConfig(**config_dict.get('tool', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            environment=EnvironmentConfig(**config_dict.get('environment', {})),
            **{k: v for k, v in config_dict.items()
               if k not in ['trm', 'dataset', 'tool', 'logging', 'environment']}
        )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'trm': self.trm.__dict__,
            'dataset': self.dataset.__dict__,
            'tool': self.tool.__dict__,
            'logging': self.logging.__dict__,
            'environment': self.environment.__dict__,
            'project_name': self.project_name,
            'project_version': self.project_version,
            'author': self.author,
            'base_dir': self.base_dir,
            'output_dir': self.output_dir,
            'cache_dir': self.cache_dir,
            'temp_dir': self.temp_dir,
            'seed': self.seed,
            'deterministic': self.deterministic,
            'debug_mode': self.debug_mode,
            'verbose': self.verbose,
            'profile_performance': self.profile_performance
        }


# Default configuration instance
DEFAULT_CONFIG = Config()


# Configuration factory functions
def create_debug_config() -> Config:
    """Create configuration for debugging."""
    config = Config()

    # Reduce sizes for debugging
    config.trm.batch_size = 8
    config.trm.hidden_size = 128
    config.trm.recursion_depth = 4
    config.dataset.max_samples_per_dataset = 100
    config.trm.num_epochs = 2

    # Enable debug features
    config.debug_mode = True
    config.verbose = True
    config.logging.log_level = "DEBUG"
    config.trm.use_tpu = False  # Use CPU for debugging

    return config


def create_tpu_config() -> Config:
    """Create configuration optimized for TPU training."""
    config = Config()

    # TPU optimizations
    config.trm.use_tpu = True
    config.trm.num_tpu_chips = 8
    config.trm.dtype = jnp.bfloat16  # Better for TPUs
    config.trm.batch_size = 4096  # Larger batch for TPU

    # Memory optimizations
    config.trm.use_binary_activations = True
    config.trm.binary_binarization = True
    config.trm.gradient_free_steps = 15

    # Larger training scale
    config.dataset.max_samples_per_dataset = 50000
    config.trm.num_epochs = 100

    return config


def create_cpu_config() -> Config:
    """Create configuration for CPU training."""
    config = Config()

    # CPU optimizations
    config.trm.use_tpu = False
    config.trm.batch_size = 32  # Smaller batch for CPU
    config.trm.dtype = jnp.float32

    # Reduce model size for CPU
    config.trm.hidden_size = 256
    config.trm.recursion_depth = 8

    return config


def get_config(config_type: str = "default") -> Config:
    """Get configuration by type."""
    if config_type == "debug":
        return create_debug_config()
    elif config_type == "tpu":
        return create_tpu_config()
    elif config_type == "cpu":
        return create_cpu_config()
    else:
        return DEFAULT_CONFIG


if __name__ == "__main__":
    # Example usage
    config = get_config("tpu")
    print(f"Configuration loaded: {config.project_name} v{config.project_version}")
    print(f"TRM hidden size: {config.trm.hidden_size}")
    print(f"Recursion depth: {config.trm.recursion_depth}")
    print(f"Batch size: {config.trm.batch_size}")