"""
Test cases for configuration module.

Tests the configuration management system, including loading from files,
environment variables, and validation.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ConfigManager,
    NCAConfig,
    DataConfig,
    TradingConfig,
    TrainingConfig,
    APIConfig,
    SystemConfig,
    TPUConfig,
    get_config,
    reload_config,
    detect_tpu_availability,
    get_tpu_device_count
)


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()

    def test_initialization(self):
        """Test configuration initialization."""
        # Check that all config sections are initialized
        self.assertIsInstance(self.config.nca, NCAConfig)
        self.assertIsInstance(self.config.data, DataConfig)
        self.assertIsInstance(self.config.trading, TradingConfig)
        self.assertIsInstance(self.config.training, TrainingConfig)
        self.assertIsInstance(self.config.api, APIConfig)
        self.assertIsInstance(self.config.system, SystemConfig)
        self.assertIsInstance(self.config.tpu, TPUConfig)

        # Check default values
        self.assertEqual(self.config.nca.state_dim, 64)
        self.assertEqual(self.config.nca.hidden_dim, 128)
        self.assertEqual(self.config.nca.num_layers, 4)
        self.assertEqual(self.config.nca.kernel_size, 3)

        # Check default tickers
        self.assertIn("AAPL", self.config.data.tickers)
        self.assertIn("MSFT", self.config.data.tickers)
        self.assertIn("GOOGL", self.config.data.tickers)

        # Check default timeframes
        self.assertIn("1m", self.config.data.timeframes)
        self.assertIn("5m", self.config.data.timeframes)
        self.assertIn("15m", self.config.data.timeframes)

    def test_load_from_file(self):
        """Test loading configuration from file."""
        # Create temporary config file
        config_data = {
            'nca': {
                'state_dim': 128,
                'hidden_dim': 256
            },
            'training': {
                'batch_size': 1024,
                'learning_rate': 0.001
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Load config from file
            config = ConfigManager(config_path)

            # Check that values were loaded
            self.assertEqual(config.nca.state_dim, 128)
            self.assertEqual(config.nca.hidden_dim, 256)
            self.assertEqual(config.training.batch_size, 1024)
            self.assertEqual(config.training.learning_rate, 0.001)
        finally:
            # Clean up
            os.unlink(config_path)

    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        os.environ['NCA_DEVICE'] = 'cpu'
        os.environ['NCA_LOG_LEVEL'] = 'debug'
        os.environ['ALPACA_API_KEY'] = 'test_api_key'
        os.environ['ALPACA_SECRET_KEY'] = 'test_secret_key'
        os.environ['NCA_MAX_POSITION'] = '0.2'
        os.environ['NCA_RISK_PER_TRADE'] = '0.02'

        try:
            # Create config
            config = ConfigManager()

            # Check that values were loaded
            self.assertEqual(config.system.device, 'cpu')
            self.assertEqual(config.system.log_level, 'DEBUG')
            self.assertEqual(config.api.alpaca_api_key, 'test_api_key')
            self.assertEqual(config.api.alpaca_secret_key, 'test_secret_key')
            self.assertEqual(config.trading.max_position_size, 0.2)
            self.assertEqual(config.trading.risk_per_trade, 0.02)
        finally:
            # Clean up environment variables
            for key in ['NCA_DEVICE', 'NCA_LOG_LEVEL', 'ALPACA_API_KEY', 
                       'ALPACA_SECRET_KEY', 'NCA_MAX_POSITION', 'NCA_RISK_PER_TRADE']:
                if key in os.environ:
                    del os.environ[key]

    def test_validate_config(self):
        """Test configuration validation."""
        # Valid config should not raise exception
        config = ConfigManager()

        # Invalid trading parameters should raise exception
        with self.assertRaises(AssertionError):
            config.trading.max_position_size = 1.5
            config._validate_config()

        with self.assertRaises(AssertionError):
            config.trading.risk_per_trade = 0.2
            config._validate_config()

        with self.assertRaises(AssertionError):
            config.trading.confidence_threshold = 1.5
            config._validate_config()

        # Invalid training parameters should raise exception
        with self.assertRaises(AssertionError):
            config.training.batch_size = -1
            config._validate_config()

        with self.assertRaises(AssertionError):
            config.training.clip_ratio = 1.5
            config._validate_config()

        # Invalid NCA parameters should raise exception
        with self.assertRaises(AssertionError):
            config.nca.state_dim = -1
            config._validate_config()

        with self.assertRaises(AssertionError):
            config.nca.num_layers = -1
            config._validate_config()

    def test_save_config(self):
        """Test saving configuration to file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name

        try:
            # Save config
            self.config.save_config(config_path)

            # Load and verify
            with open(config_path, 'r') as f:
                saved_data = yaml.safe_load(f)

            # Check that values were saved
            self.assertEqual(saved_data['nca']['state_dim'], 64)
            self.assertEqual(saved_data['training']['batch_size'], 2048)
        finally:
            # Clean up
            os.unlink(config_path)

    def test_get_config_summary(self):
        """Test getting configuration summary."""
        summary = self.config.get_config_summary()

        # Check that summary contains expected sections
        self.assertIn('nca', summary)
        self.assertIn('data', summary)
        self.assertIn('trading', summary)
        self.assertIn('training', summary)
        self.assertIn('system', summary)
        self.assertIn('tpu', summary)

        # Check that summary contains expected values
        self.assertEqual(summary['nca']['state_dim'], 64)
        self.assertEqual(summary['training']['batch_size'], 2048)
        self.assertEqual(summary['training']['use_mixed_precision'], True)


class TestNCAConfig(unittest.TestCase):
    """Test cases for NCAConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NCAConfig()

        # Architecture parameters
        self.assertEqual(config.state_dim, 64)
        self.assertEqual(config.hidden_dim, 128)
        self.assertEqual(config.num_layers, 4)
        self.assertEqual(config.kernel_size, 3)

        # Training parameters
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.weight_decay, 1e-4)
        self.assertEqual(config.dropout_rate, 0.1)

        # NCA-specific parameters
        self.assertEqual(config.alpha, 0.1)
        self.assertEqual(config.beta, 0.1)
        self.assertEqual(config.gamma, 0.1)

        # Self-adaptation parameters
        self.assertEqual(config.adaptation_rate, 0.01)
        self.assertEqual(config.mutation_rate, 0.001)
        self.assertEqual(config.selection_pressure, 0.1)


class TestDataConfig(unittest.TestCase):
    """Test cases for DataConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()

        # Data sources
        self.assertTrue(config.use_sp500_yahoo)
        self.assertTrue(config.use_kaggle_nasdaq)
        self.assertTrue(config.use_kaggle_huge_stock)
        self.assertTrue(config.use_kaggle_sp500)
        self.assertTrue(config.use_kaggle_world_stocks)
        self.assertTrue(config.use_kaggle_exchanges)
        self.assertFalse(config.use_quantopian_data)
        self.assertTrue(config.use_global_financial_data)
        self.assertEqual(config.backtest_end_year, 2021)

        # Technical indicators
        self.assertEqual(config.rsi_period, 14)
        self.assertEqual(config.macd_fast, 12)
        self.assertEqual(config.macd_slow, 26)
        self.assertEqual(config.macd_signal, 9)
        self.assertEqual(config.bb_period, 20)
        self.assertEqual(config.bb_std, 2.0)

        # Data preprocessing
        self.assertEqual(config.sequence_length, 60)
        self.assertEqual(config.prediction_horizon, 5)
        self.assertEqual(config.train_test_split, 0.8)
        self.assertEqual(config.validation_split, 0.1)

        # Normalization
        self.assertTrue(config.normalize_features)
        self.assertEqual(config.feature_range, (-1.0, 1.0))

        # Memory management
        self.assertEqual(config.max_ram_gb, 320.0)
        self.assertEqual(config.chunk_size_mb, 100)


class TestTradingConfig(unittest.TestCase):
    """Test cases for TradingConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TradingConfig()

        # Risk management
        self.assertEqual(config.max_position_size, 0.1)
        self.assertEqual(config.max_daily_loss, 0.05)
        self.assertEqual(config.max_drawdown, 0.15)
        self.assertEqual(config.risk_per_trade, 0.01)

        # Position sizing
        self.assertEqual(config.kelly_fraction, 0.5)
        self.assertEqual(config.volatility_target, 0.15)

        # Entry/Exit signals
        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertEqual(config.stop_loss_pct, 0.05)
        self.assertEqual(config.take_profit_pct, 0.10)

        # Trading costs
        self.assertEqual(config.commission_per_share, 0.005)
        self.assertEqual(config.slippage_pct, 0.001)


class TestTrainingConfig(unittest.TestCase):
    """Test cases for TrainingConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()

        # RL parameters
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.gae_lambda, 0.95)
        self.assertEqual(config.clip_ratio, 0.2)
        self.assertEqual(config.entropy_coeff, 0.01)
        self.assertEqual(config.value_coeff, 0.5)

        # Training hyperparameters
        self.assertEqual(config.batch_size, 2048)
        self.assertEqual(config.micro_batch_size, 256)
        self.assertEqual(config.gradient_accumulation_steps, 8)
        self.assertEqual(config.num_epochs, 10)
        self.assertEqual(config.max_grad_norm, 0.5)
        self.assertEqual(config.target_kl, 0.01)

        # JAX-specific parameters
        self.assertTrue(config.use_jax)
        self.assertEqual(config.jax_optimizer, "adamw")
        self.assertEqual(config.jax_lr_schedule, "cosine")

        # DDP parameters
        self.assertEqual(config.num_gpus, 2)
        self.assertEqual(config.local_rank, -1)

        # Mixed precision parameters
        self.assertTrue(config.use_mixed_precision)
        self.assertEqual(config.precision_dtype, "bf16")
        self.assertEqual(config.compute_dtype, "f32")

        # Checkpointing
        self.assertEqual(config.save_freq, 1000)
        self.assertEqual(config.eval_freq, 100)
        self.assertEqual(config.log_freq, 10)


class TestAPIConfig(unittest.TestCase):
    """Test cases for APIConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = APIConfig()

        # Alpaca API
        self.assertEqual(config.alpaca_api_key, "PKJ346E2YWMT7HCFZX09")
        self.assertEqual(config.alpaca_secret_key, "w3LaDFeYjy3CJM9S37Ox0YQbeQIgEyfmlhFO7Y3m")
        self.assertEqual(config.alpaca_base_url, "https://paper-api.alpaca.markets")

        # Alpha Vantage API
        self.assertEqual(config.alpha_vantage_key, "")

        # Polygon API
        self.assertEqual(config.polygon_key, "")

        # Rate limiting
        self.assertEqual(config.requests_per_minute, 200)
        self.assertEqual(config.requests_per_second, 5)


class TestSystemConfig(unittest.TestCase):
    """Test cases for SystemConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SystemConfig()

        # Logging
        self.assertEqual(config.log_level, "INFO")
        self.assertEqual(config.log_file, "nca_trading_bot.log")

        # Performance
        self.assertEqual(config.num_workers, 4)
        self.assertTrue(config.pin_memory)
        self.assertTrue(config.persistent_workers)

        # Hardware
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.seed, 42)

        # Caching
        self.assertEqual(config.cache_size, 1000)
        self.assertEqual(config.cache_ttl, 3600)

        # Monitoring
        self.assertTrue(config.enable_tensorboard)
        self.assertFalse(config.enable_wandb)
        self.assertEqual(config.wandb_project, "nca-trading-bot")


class TestTPUConfig(unittest.TestCase):
    """Test cases for TPUConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TPUConfig()

        # TPU hardware
        self.assertEqual(config.tpu_cores, 8)
        self.assertEqual(config.tpu_chips, 1)
        self.assertEqual(config.tpu_topology, "2x4")

        # JAX/TPU optimizations
        self.assertTrue(config.use_jax)
        self.assertEqual(config.jax_backend, "tpu")

        # Mixed precision
        self.assertTrue(config.mixed_precision)
        self.assertEqual(config.precision_dtype, "bf16")
        self.assertEqual(config.compute_dtype, "f32")

        # Sharding configuration
        self.assertTrue(config.enable_sharding)
        self.assertEqual(config.sharding_strategy, "2d")
        self.assertEqual(config.mesh_shape, (1, 8))
        self.assertEqual(config.axis_names, ("data", "model"))

        # Large batch optimization
        self.assertEqual(config.batch_size, 2048)
        self.assertEqual(config.micro_batch_size, 256)
        self.assertEqual(config.gradient_accumulation_steps, 8)

        # Memory management
        self.assertEqual(config.max_memory_gb, 35.0)
        self.assertEqual(config.memory_fraction, 0.875)
        self.assertTrue(config.enable_memory_optimization)

        # XLA compilation and optimization
        self.assertTrue(config.xla_compile)
        self.assertTrue(config.xla_persistent_cache)
        self.assertEqual(config.xla_memory_fraction, 0.8)
        self.assertFalse(config.xla_precompile)

        # TPU-specific optimizations
        self.assertEqual(config.matmul_precision, "high")
        self.assertTrue(config.enable_fusion)
        self.assertTrue(config.enable_remat)
        self.assertEqual(config.remat_policy, "dots_with_no_batch_dims")

        # Performance monitoring
        self.assertTrue(config.tpu_metrics_enabled)
        self.assertTrue(config.xla_metrics_enabled)
        self.assertTrue(config.memory_stats_enabled)
        self.assertFalse(config.profiling_enabled)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_get_config(self):
        """Test getting global configuration instance."""
        config = get_config()
        self.assertIsInstance(config, ConfigManager)

        # Multiple calls should return the same instance
        config2 = get_config()
        self.assertIs(config, config2)

    def test_reload_config(self):
        """Test reloading configuration."""
        # Get original config
        original_config = get_config()
        original_state_dim = original_config.nca.state_dim

        # Modify config
        original_config.nca.state_dim = 128

        # Reload config
        reloaded_config = reload_config()

        # Check that config was reloaded
        self.assertEqual(reloaded_config.nca.state_dim, 64)

        # Check that global config was updated
        current_config = get_config()
        self.assertEqual(current_config.nca.state_dim, 64)

    def test_detect_tpu_availability(self):
        """Test TPU availability detection."""
        # Should return False in most environments
        is_available = detect_tpu_availability()
        self.assertIsInstance(is_available, bool)

    def test_get_tpu_device_count(self):
        """Test getting TPU device count."""
        # Should return 0 in most environments
        count = get_tpu_device_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)


if __name__ == '__main__':
    unittest.main()