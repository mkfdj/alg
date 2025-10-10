"""
Configuration settings for NCA Trading Bot
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import os


@dataclass
class Config:
    """Configuration settings for the NCA Trading Bot"""

    # === NCA Model Configuration ===
    nca_grid_size: Tuple[int, int] = (64, 64)
    nca_channels: int = 16  # RGB(3) + Alpha(1) + Hidden(12)
    nca_hidden_dim: int = 128
    nca_layers: int = 2
    nca_evolution_steps: int = 96  # Number of NCA steps per prediction
    nca_learning_rate: float = 1e-3
    nca_growth_threshold: float = 0.1  # Error threshold for grid expansion
    nca_max_grid_size: Tuple[int, int] = (256, 256)

    # === Reinforcement Learning Configuration ===
    rl_batch_size: int = 4096
    rl_learning_rate: float = 1e-4
    rl_gamma: float = 0.99  # Discount factor
    rl_lambda: float = 0.95  # GAE parameter
    rl_clip_eps: float = 0.2  # PPO clipping parameter
    rl_value_coef: float = 0.5
    rl_entropy_coef: float = 0.01
    rl_max_grad_norm: float = 0.5
    rl_update_epochs: int = 10
    rl_ensemble_size: int = 5  # Number of parallel NCAs

    # === Trading Configuration ===
    trading_initial_balance: float = 10000.0
    trading_max_position_size: float = 0.25  # Max 25% of portfolio per trade
    trading_max_risk_per_trade: float = 0.02  # Max 2% risk per trade
    trading_max_portfolio_heat: float = 0.20  # Max 20% total exposure
    trading_stop_loss_atr_multiplier: float = 2.0
    trading_commission: float = 0.001  # 0.1% commission
    trading_slippage: float = 0.0005  # 0.05% slippage

    # === Data Configuration ===
    data_start_date: str = "1990-01-01"
    data_end_date: str = "2021-12-31"
    data_validation_split: float = 0.2
    data_test_split: float = 0.1
    data_sequence_length: int = 200  # Number of time steps for NCA input
    data_prediction_horizon: int = 20  # Number of steps to predict ahead

    # === Top Trading Tickers (2025) ===
    top_tickers: List[str] = None

    def __post_init__(self):
        if self.top_tickers is None:
            self.top_tickers = [
                "NVDA",  # NVIDIA - AI/semiconductors
                "MSFT",  # Microsoft - Cloud/software
                "AAPL",  # Apple - Consumer tech
                "AMZN",  # Amazon - E-commerce/cloud
                "GOOGL", # Alphabet - Search/AI
                "META",  # Meta - Social/metaverse
                "BRK-B", # Berkshire Hathaway - Diversified
                "TSLA",  # Tesla - EV/Energy
                "UNH",   # UnitedHealth - Healthcare
                "JNJ"    # Johnson & Johnson - Healthcare
            ]

        # Initialize datasets
        if self.datasets is None:
            self.datasets = {
                "kaggle_stock_market": {
                    "path": "/kaggle/input/stock-market-dataset",
                    "format": "csv",
                    "description": "NASDAQ stocks with OHLCV data"
                },
                "yahoo_finance": {
                    "tickers": self.top_tickers,
                    "format": "yfinance",
                    "description": "Real-time and historical data via yfinance"
                },
                "sp500_components": {
                    "path": "/kaggle/input/s-and-p-500",
                    "format": "csv",
                    "description": "S&P 500 historical data"
                }
            }

        # Initialize technical indicators
        if self.technical_indicators is None:
            self.technical_indicators = {
                "rsi": {"period": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bollinger": {"period": 20, "std": 2},
                "atr": {"period": 14},
                "sma": {"periods": [5, 20, 50]},
                "ema": {"periods": [12, 26]},
                "stochastic": {"k": 14, "d": 3},
                "volume_sma": {"period": 20},
                "vwap": {"periods": [5, 20]}
            }

        # Initialize risk management
        if self.risk_management is None:
            self.risk_management = {
                "max_drawdown": 0.20,  # 20% max drawdown
                "max_consecutive_losses": 5,
                "min_sharpe_ratio": 0.5,
                "max_volatility": 0.30,  # 30% annual volatility
                "min_liquidity": 50000000,  # $50M daily volume minimum
                "max_position_age_days": 30,  # Close positions after 30 days
                "portfolio_rebalance_frequency": "daily"
            }

    # === JAX/TPU Configuration ===
    jax_platform: str = "tpu"
    jax_enable_x64: bool = False  # Use float32 for speed
    jax_debug_nans: bool = True
    jax_memory_fraction: float = 0.9
    tpu_device_count: int = 8

    # === Dataset Configuration ===
    datasets: Dict[str, Dict] = None

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = {
                "kaggle_stock_market": {
                    "path": "/kaggle/input/stock-market-dataset",
                    "format": "csv",
                    "description": "NASDAQ stocks with OHLCV data"
                },
                "yahoo_finance": {
                    "tickers": self.top_tickers,
                    "format": "yfinance",
                    "description": "Real-time and historical data via yfinance"
                },
                "sp500_components": {
                    "path": "/kaggle/input/s-and-p-500",
                    "format": "csv",
                    "description": "S&P 500 historical data"
                }
            }

    # === Alpaca API Configuration ===
    alpaca_paper_api_key: Optional[str] = None
    alpaca_paper_secret_key: Optional[str] = None
    alpaca_paper_base_url: str = "https://paper-api.alpaca.markets/v2"
    alpaca_live_api_key: Optional[str] = None
    alpaca_live_secret_key: Optional[str] = None
    alpaca_live_base_url: str = "https://api.alpaca.markets"

    def get_alpaca_config(self, paper_mode: bool = True) -> Dict[str, str]:
        """Get Alpaca API configuration for paper or live trading"""
        if paper_mode:
            # For paper trading, use separate API key and secret key
            api_key = self.alpaca_paper_api_key or os.getenv("ALPACA_PAPER_API_KEY")
            secret_key = self.alpaca_paper_secret_key or os.getenv("ALPACA_PAPER_SECRET_KEY")
            return {
                "key_id": api_key,
                "secret_key": secret_key,
                "base_url": self.alpaca_paper_base_url
            }
        else:
            # For live trading, use separate API key and secret key
            return {
                "key_id": self.alpaca_live_api_key or os.getenv("ALPACA_LIVE_API_KEY"),
                "secret_key": self.alpaca_live_secret_key or os.getenv("ALPACA_LIVE_SECRET_KEY"),
                "base_url": self.alpaca_live_base_url
            }

    # === Technical Indicators Configuration ===
    technical_indicators: Dict[str, Dict] = None

    def __post_init__(self):
        if self.technical_indicators is None:
            self.technical_indicators = {
                "rsi": {"period": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bollinger": {"period": 20, "std": 2},
                "atr": {"period": 14},
                "sma": {"periods": [5, 20, 50]},
                "ema": {"periods": [12, 26]},
                "stochastic": {"k": 14, "d": 3},
                "volume_sma": {"period": 20},
                "vwap": {"periods": [5, 20]}
            }

    # === Logging and Monitoring ===
    log_level: str = "INFO"
    wandb_project: Optional[str] = "nca-trading-bot"
    wandb_entity: Optional[str] = None
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    tensorboard_log_dir: str = "./logs"

    # === Optimization Configuration ===
    use_float16: bool = True  # Use float16 for memory efficiency
    gradient_checkpointing: bool = True  # Enable gradient checkpointing
    mixed_precision: bool = True
    compile_nca_evolution: bool = True  # JIT compile NCA evolution
    compile_rl_training: bool = True  # JIT compile RL training

    # === Risk Management Configuration ===
    risk_management: Dict[str, float] = None

    def __post_init__(self):
        if self.risk_management is None:
            self.risk_management = {
                "max_drawdown": 0.20,  # 20% max drawdown
                "max_consecutive_losses": 5,
                "min_sharpe_ratio": 0.5,
                "max_volatility": 0.30,  # 30% annual volatility
                "min_liquidity": 50000000,  # $50M daily volume minimum
                "max_position_age_days": 30,  # Close positions after 30 days
                "portfolio_rebalance_frequency": "daily"
            }

    @classmethod
    def from_args(cls, args) -> "Config":
        """Create configuration from command line arguments"""
        config = cls()

        # Override with command line arguments
        if hasattr(args, 'mode') and args.mode:
            config.trading_mode = args.mode
        if hasattr(args, 'tickers') and args.tickers:
            config.top_tickers = args.tickers
        if hasattr(args, 'datasets') and args.datasets:
            config.selected_datasets = args.datasets
        if hasattr(args, 'paper_mode') and args.paper_mode:
            config.paper_trading = args.paper_mode

        return config

    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []

        # Validate trading parameters
        if self.trading_max_position_size > 1.0:
            errors.append("Max position size cannot exceed 100%")
        if self.trading_max_risk_per_trade > 0.05:
            errors.append("Max risk per trade should not exceed 5%")

        # Validate NCA parameters
        if self.nca_grid_size[0] > self.nca_max_grid_size[0]:
            errors.append("Grid size exceeds maximum")
        if self.nca_learning_rate <= 0:
            errors.append("Learning rate must be positive")

        # Validate Alpaca configuration
        if not self.alpaca_paper_api_key:
            errors.append("Alpaca paper API key not configured")

        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True


# Global configuration instance
config = Config()