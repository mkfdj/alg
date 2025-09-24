"""
Configuration module for NCA Trading Bot.

This module provides centralized configuration management for all components
of the Neural Cellular Automata trading system, including hyperparameters,
API settings, trading parameters, and system configurations.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


@dataclass
class NCAConfig:
    """Neural Cellular Automata configuration parameters."""

    # Architecture parameters
    state_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 4
    kernel_size: int = 3

    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout_rate: float = 0.1

    # NCA-specific parameters
    alpha: float = 0.1  # Update rate
    beta: float = 0.1   # Growth rate
    gamma: float = 0.1  # Decay rate

    # Self-adaptation parameters
    adaptation_rate: float = 0.01
    mutation_rate: float = 0.001
    selection_pressure: float = 0.1


@dataclass
class DataConfig:
    """Data handling configuration parameters."""

    # Data sources
    tickers: List[str] = None  # Will be set to default if None
    timeframes: List[str] = None  # Will be set to default if None

    # New dataset sources
    use_sp500_yahoo: bool = True  # Fetch S&P500 from Yahoo Finance up to 2021
    use_kaggle_nasdaq: bool = True  # Use Kaggle NASDAQ dataset
    use_kaggle_huge_stock: bool = True  # Huge US stock market dataset
    use_kaggle_sp500: bool = True  # S&P500 stock data
    use_kaggle_world_stocks: bool = True  # World stock prices
    use_kaggle_exchanges: bool = True  # Stock exchange data
    use_quantopian_data: bool = False  # Quantopian data (limited availability)
    use_global_financial_data: bool = True  # Global financial datasets
    backtest_end_year: int = 2021  # Filter data <= this year for backtesting

    # Kaggle API settings
    kaggle_username: str = ""
    kaggle_key: str = ""

    # Technical indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0

    # Data preprocessing
    sequence_length: int = 60
    prediction_horizon: int = 5
    train_test_split: float = 0.8
    validation_split: float = 0.1

    # Normalization
    normalize_features: bool = True
    feature_range: Tuple[float, float] = (-1.0, 1.0)

    # Memory management for large datasets
    max_ram_gb: float = 320.0  # Maximum RAM to use for data loading
    chunk_size_mb: int = 100  # Chunk size for processing large files


@dataclass
class TradingConfig:
    """Trading configuration parameters."""

    # Risk management
    max_position_size: float = 0.1  # Max 10% of portfolio
    max_daily_loss: float = 0.05    # Max 5% daily loss
    max_drawdown: float = 0.15      # Max 15% drawdown
    risk_per_trade: float = 0.01    # 1% risk per trade

    # Position sizing
    kelly_fraction: float = 0.5     # Use 50% of Kelly criterion
    volatility_target: float = 0.15 # Target 15% annual volatility

    # Entry/Exit signals
    confidence_threshold: float = 0.7
    stop_loss_pct: float = 0.05     # 5% stop loss
    take_profit_pct: float = 0.10   # 10% take profit

    # Trading costs
    commission_per_share: float = 0.005
    slippage_pct: float = 0.001     # 0.1% slippage


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    # RL parameters
    gamma: float = 0.99             # Discount factor
    gae_lambda: float = 0.95        # GAE parameter
    clip_ratio: float = 0.2         # PPO clip ratio
    entropy_coeff: float = 0.01     # Entropy coefficient
    value_coeff: float = 0.5         # Value function coefficient

    # Training hyperparameters
    batch_size: int = 64
    num_epochs: int = 10
    max_grad_norm: float = 0.5
    target_kl: float = 0.01

    # DDP parameters (deprecated - use TPU config instead)
    num_gpus: int = 2
    local_rank: int = -1

    # AMP parameters
    use_amp: bool = True
    amp_dtype: str = "float16"

    # Checkpointing
    save_freq: int = 1000
    eval_freq: int = 100
    log_freq: int = 10


@dataclass
class APIConfig:
    """API configuration parameters."""

    # Alpaca API
    alpaca_api_key: str = "PKJ346E2YWMT7HCFZX09"
    alpaca_secret_key: str = "w3LaDFeYjy3CJM9S37Ox0YQbeQIgEyfmlhFO7Y3m"
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # Alpha Vantage API (fallback)
    alpha_vantage_key: str = ""

    # Polygon API (alternative)
    polygon_key: str = ""

    # Rate limiting
    requests_per_minute: int = 200
    requests_per_second: int = 5


@dataclass
class TPUConfig:
    """TPU-specific configuration parameters."""

    # TPU hardware
    tpu_cores: int = 8  # TPU v5e-8 has 8 cores
    tpu_chips: int = 1  # Single host by default
    tpu_topology: str = "2x4"  # TPU v5e-8 topology

    # XLA compilation
    xla_compile: bool = True
    xla_memory_fraction: float = 0.8
    xla_precompile: bool = False

    # SPMD configuration
    spmd_sharding: bool = True
    sharding_strategy: str = "2d"  # 2d, 1d, replicated
    mesh_shape: Tuple[int, int] = (1, 8)  # (chips, cores_per_chip)

    # TPU-specific optimizations
    bf16_precision: bool = True  # TPU v5e-8 supports bfloat16
    matmul_precision: str = "high"  # high, medium, low
    enable_fusion: bool = True

    # Performance monitoring
    tpu_metrics_enabled: bool = True
    xla_metrics_enabled: bool = True
    memory_stats_enabled: bool = True


@dataclass
class SystemConfig:
    """System configuration parameters."""

    # Paths
    project_root: Path = None
    data_dir: Path = None
    model_dir: Path = None
    log_dir: Path = None

    # Logging
    log_level: str = "INFO"
    log_file: str = "nca_trading_bot.log"

    # Performance
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # Hardware
    device: str = "auto"  # auto, cpu, cuda, tpu
    seed: int = 42

    # Caching
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour

    # Monitoring
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    wandb_project: str = "nca-trading-bot"


class ConfigManager:
    """
    Centralized configuration manager for the NCA Trading Bot.

    Handles loading configuration from environment variables, config files,
    and provides validation and type checking.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path or os.getenv("NCA_CONFIG_PATH")
        self._configs = {}

        # Initialize all config sections
        self.nca = NCAConfig()
        self.data = DataConfig()
        self.trading = TradingConfig()
        self.training = TrainingConfig()
        self.api = APIConfig()
        self.system = SystemConfig()
        self.tpu = TPUConfig()

        # Set default tickers and timeframes if not provided
        if self.data.tickers is None:
            self.data.tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
                "NVDA", "META", "NFLX", "SPY", "QQQ"
            ]

        if self.data.timeframes is None:
            self.data.timeframes = ["1m", "5m", "15m", "1h", "1d"]

        # Set system paths
        self._setup_paths()

        # Load configuration from file if provided
        if self.config_path:
            self._load_from_file()

        # Load from environment variables
        self._load_from_env()

        # Validate configuration
        self._validate_config()

    def _setup_paths(self) -> None:
        """Set up system paths."""
        self.system.project_root = Path(__file__).parent
        self.system.data_dir = self.system.project_root / "data"
        self.system.model_dir = self.system.project_root / "models"
        self.system.log_dir = self.system.project_root / "logs"

        # Create directories if they don't exist
        for path in [self.system.data_dir, self.system.model_dir, self.system.log_dir]:
            path.mkdir(exist_ok=True)

    def _load_from_file(self) -> None:
        """Load configuration from file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)

            self._update_from_dict(config_dict)
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # API keys
        if api_key := os.getenv("ALPACA_API_KEY"):
            self.api.alpaca_api_key = api_key
        if secret_key := os.getenv("ALPACA_SECRET_KEY"):
            self.api.alpaca_secret_key = secret_key

        # Alpha Vantage
        if av_key := os.getenv("ALPHA_VANTAGE_KEY"):
            self.api.alpha_vantage_key = av_key

        # Polygon
        if poly_key := os.getenv("POLYGON_API_KEY"):
            self.api.polygon_key = poly_key

        # Kaggle API
        if kaggle_user := os.getenv("KAGGLE_USERNAME"):
            self.data.kaggle_username = kaggle_user
        if kaggle_key := os.getenv("KAGGLE_KEY"):
            self.data.kaggle_key = kaggle_key

        # System settings
        if device := os.getenv("NCA_DEVICE"):
            self.system.device = device
        if log_level := os.getenv("NCA_LOG_LEVEL"):
            self.system.log_level = log_level.upper()

        # TPU settings
        if tpu_cores := os.getenv("NCA_TPU_CORES"):
            self.tpu.tpu_cores = int(tpu_cores)
        if tpu_chips := os.getenv("NCA_TPU_CHIPS"):
            self.tpu.tpu_chips = int(tpu_chips)
        if xla_memory := os.getenv("NCA_XLA_MEMORY_FRACTION"):
            self.tpu.xla_memory_fraction = float(xla_memory)
        if sharding_strategy := os.getenv("NCA_SHARDING_STRATEGY"):
            self.tpu.sharding_strategy = sharding_strategy

        # Trading parameters
        if max_pos := os.getenv("NCA_MAX_POSITION"):
            self.trading.max_position_size = float(max_pos)
        if risk_per_trade := os.getenv("NCA_RISK_PER_TRADE"):
            self.trading.risk_per_trade = float(risk_per_trade)

        # Data parameters
        if backtest_year := os.getenv("NCA_BACKTEST_END_YEAR"):
            self.data.backtest_end_year = int(backtest_year)
        if max_ram := os.getenv("NCA_MAX_RAM_GB"):
            self.data.max_ram_gb = float(max_ram)

    def _update_from_dict(self, config_dict: Dict) -> None:
        """Update configuration from dictionary."""
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate API keys
        if not self.api.alpaca_api_key or not self.api.alpaca_secret_key:
            print("Warning: Alpaca API keys not configured. Paper trading will not work.")

        # Validate trading parameters
        assert 0 < self.trading.max_position_size <= 1, "max_position_size must be between 0 and 1"
        assert 0 < self.trading.risk_per_trade <= 0.1, "risk_per_trade should be reasonable"
        assert 0 < self.trading.confidence_threshold <= 1, "confidence_threshold must be between 0 and 1"

        # Validate training parameters
        assert self.training.batch_size > 0, "batch_size must be positive"
        assert 0 < self.training.clip_ratio <= 1, "clip_ratio must be between 0 and 1"

        # Validate NCA parameters
        assert self.nca.state_dim > 0, "state_dim must be positive"
        assert self.nca.num_layers > 0, "num_layers must be positive"

        # Validate TPU parameters
        assert self.tpu.tpu_cores > 0, "tpu_cores must be positive"
        assert self.tpu.tpu_chips > 0, "tpu_chips must be positive"
        assert 0 < self.tpu.xla_memory_fraction <= 1, "xla_memory_fraction must be between 0 and 1"
        assert self.tpu.sharding_strategy in ["2d", "1d", "replicated"], "Invalid sharding strategy"

    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file.

        Args:
            path: Path to save configuration (optional)
        """
        save_path = path or self.config_path or "config.yaml"

        try:
            import yaml

            config_dict = {}
            for attr_name in dir(self):
                if not attr_name.startswith('_') and isinstance(getattr(self, attr_name), (NCAConfig, DataConfig, TradingConfig, TrainingConfig, APIConfig, SystemConfig, TPUConfig)):
                    config_dict[attr_name] = {}
                    for field in dir(getattr(self, attr_name)):
                        if not field.startswith('_') and not callable(getattr(getattr(self, attr_name), field)):
                            config_dict[attr_name][field] = getattr(getattr(self, attr_name), field)

            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)

            print(f"Configuration saved to {save_path}")

        except Exception as e:
            print(f"Error saving configuration: {e}")

    def get_config_summary(self) -> Dict:
        """Get a summary of current configuration.

        Returns:
            Dictionary containing configuration summary
        """
        return {
            "nca": {
                "state_dim": self.nca.state_dim,
                "hidden_dim": self.nca.hidden_dim,
                "num_layers": self.nca.num_layers,
                "learning_rate": self.nca.learning_rate
            },
            "data": {
                "num_tickers": len(self.data.tickers),
                "sequence_length": self.data.sequence_length,
                "prediction_horizon": self.data.prediction_horizon
            },
            "trading": {
                "max_position_size": self.trading.max_position_size,
                "risk_per_trade": self.trading.risk_per_trade,
                "confidence_threshold": self.trading.confidence_threshold
            },
            "training": {
                "batch_size": self.training.batch_size,
                "num_gpus": self.training.num_gpus,
                "use_amp": self.training.use_amp
            },
            "system": {
                "device": self.system.device,
                "log_level": self.system.log_level,
                "num_workers": self.system.num_workers
            },
            "tpu": {
                "tpu_cores": self.tpu.tpu_cores,
                "tpu_chips": self.tpu.tpu_chips,
                "xla_compile": self.tpu.xla_compile,
                "sharding_strategy": self.tpu.sharding_strategy,
                "bf16_precision": self.tpu.bf16_precision
            }
        }


# Global configuration instance
config = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration instance.

    Returns:
        Global configuration manager instance
    """
    return config


def detect_tpu_availability() -> bool:
    """Detect if TPU is available.

    Returns:
        True if TPU is available, False otherwise
    """
    try:
        import torch_xla.core.xla_model as xm
        return xm.is_master_ordinal()
    except ImportError:
        return False


def get_tpu_device_count() -> int:
    """Get the number of available TPU devices.

    Returns:
        Number of TPU devices available
    """
    try:
        import torch_xla.core.xla_model as xm
        return xm.xrt_world_size()
    except ImportError:
        return 0


def reload_config() -> ConfigManager:
    """Reload configuration from file and environment.

    Returns:
        Updated configuration manager instance
    """
    global config
    config = ConfigManager(config.config_path)
    return config


if __name__ == "__main__":
    # Example usage and validation
    print("NCA Trading Bot Configuration")
    print("=" * 40)

    config_summary = config.get_config_summary()
    for section, params in config_summary.items():
        print(f"\n{section.upper()} CONFIGURATION:")
        for key, value in params.items():
            print(f"  {key}: {value}")

    print(f"\nAPI Keys Configured: {'Yes' if config.api.alpaca_api_key else 'No'}")
    print(f"Data Directory: {config.system.data_dir}")
    print(f"Model Directory: {config.system.model_dir}")