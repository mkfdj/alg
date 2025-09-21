"""
Unit tests for configuration module.

Tests configuration loading, validation, and management functionality.
"""

import unittest
import tempfile
import os
from pathlib import Path
import yaml

from ..config import ConfigManager, NCAConfig, DataConfig, TradingConfig, TrainingConfig, APIConfig, SystemConfig


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temp directory and files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_initialization(self):
        """Test configuration initialization with default values."""
        config = ConfigManager()

        # Test that all config sections are initialized
        self.assertIsInstance(config.nca, NCAConfig)
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.trading, TradingConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertIsInstance(config.api, APIConfig)
        self.assertIsInstance(config.system, SystemConfig)

        # Test default values
        self.assertEqual(config.nca.state_dim, 64)
        self.assertEqual(config.data.sequence_length, 60)
        self.assertEqual(config.trading.max_position_size, 0.1)

    def test_config_from_file(self):
        """Test loading configuration from file."""
        # Create test config file
        test_config = {
            'nca': {
                'state_dim': 128,
                'hidden_dim': 256,
                'learning_rate': 0.0005
            },
            'trading': {
                'max_position_size': 0.05,
                'risk_per_trade': 0.005
            }
        }

        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)

        # Load configuration
        config = ConfigManager(self.config_path)

        # Test loaded values
        self.assertEqual(config.nca.state_dim, 128)
        self.assertEqual(config.nca.hidden_dim, 256)
        self.assertEqual(config.nca.learning_rate, 0.0005)
        self.assertEqual(config.trading.max_position_size, 0.05)
        self.assertEqual(config.trading.risk_per_trade, 0.005)

    def test_config_from_environment(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        os.environ['ALPACA_API_KEY'] = 'test_key'
        os.environ['ALPACA_SECRET_KEY'] = 'test_secret'
        os.environ['NCA_DEVICE'] = 'cpu'
        os.environ['NCA_LOG_LEVEL'] = 'DEBUG'

        config = ConfigManager()

        # Test environment variable loading
        self.assertEqual(config.api.alpaca_api_key, 'test_key')
        self.assertEqual(config.api.alpaca_secret_key, 'test_secret')
        self.assertEqual(config.system.device, 'cpu')
        self.assertEqual(config.system.log_level, 'DEBUG')

    def test_config_validation(self):
        """Test configuration validation."""
        config = ConfigManager()

        # Test valid configuration
        config._validate_config()  # Should not raise exception

        # Test invalid configuration
        config.trading.max_position_size = 1.5  # Invalid value
        with self.assertRaises(AssertionError):
            config._validate_config()

    def test_config_save_load(self):
        """Test saving and loading configuration."""
        config = ConfigManager()

        # Modify some values
        config.nca.state_dim = 128
        config.trading.max_position_size = 0.05

        # Save configuration
        config.save_config(self.config_path)

        # Verify file exists
        self.assertTrue(os.path.exists(self.config_path))

        # Load configuration from file
        new_config = ConfigManager(self.config_path)

        # Test loaded values
        self.assertEqual(new_config.nca.state_dim, 128)
        self.assertEqual(new_config.trading.max_position_size, 0.05)

    def test_config_summary(self):
        """Test configuration summary generation."""
        config = ConfigManager()
        summary = config.get_config_summary()

        # Test summary structure
        self.assertIn('nca', summary)
        self.assertIn('data', summary)
        self.assertIn('trading', summary)
        self.assertIn('training', summary)
        self.assertIn('system', summary)

        # Test summary content
        self.assertIn('state_dim', summary['nca'])
        self.assertIn('num_tickers', summary['data'])
        self.assertIn('max_position_size', summary['trading'])

    def test_path_setup(self):
        """Test automatic path setup."""
        config = ConfigManager()

        # Test that paths are set up correctly
        self.assertTrue(config.system.project_root.exists())
        self.assertTrue(config.system.data_dir.exists())
        self.assertTrue(config.system.model_dir.exists())
        self.assertTrue(config.system.log_dir.exists())

        # Test path types
        self.assertIsInstance(config.system.project_root, Path)
        self.assertIsInstance(config.system.data_dir, Path)


class TestNCAConfig(unittest.TestCase):
    """Test cases for NCAConfig class."""

    def test_nca_config_defaults(self):
        """Test NCA configuration default values."""
        config = NCAConfig()

        self.assertEqual(config.state_dim, 64)
        self.assertEqual(config.hidden_dim, 128)
        self.assertEqual(config.num_layers, 4)
        self.assertEqual(config.kernel_size, 3)
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.dropout_rate, 0.1)
        self.assertEqual(config.adaptation_rate, 0.01)


class TestDataConfig(unittest.TestCase):
    """Test cases for DataConfig class."""

    def test_data_config_defaults(self):
        """Test data configuration default values."""
        config = DataConfig()

        self.assertIsNotNone(config.tickers)
        self.assertIsNotNone(config.timeframes)
        self.assertEqual(config.sequence_length, 60)
        self.assertEqual(config.prediction_horizon, 5)
        self.assertEqual(config.rsi_period, 14)
        self.assertEqual(config.macd_fast, 12)


class TestTradingConfig(unittest.TestCase):
    """Test cases for TradingConfig class."""

    def test_trading_config_defaults(self):
        """Test trading configuration default values."""
        config = TradingConfig()

        self.assertEqual(config.max_position_size, 0.1)
        self.assertEqual(config.max_daily_loss, 0.05)
        self.assertEqual(config.risk_per_trade, 0.01)
        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertEqual(config.stop_loss_pct, 0.05)


class TestTrainingConfig(unittest.TestCase):
    """Test cases for TrainingConfig class."""

    def test_training_config_defaults(self):
        """Test training configuration default values."""
        config = TrainingConfig()

        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.num_epochs, 10)
        self.assertEqual(config.num_gpus, 2)
        self.assertTrue(config.use_amp)
        self.assertEqual(config.amp_dtype, "float16")


class TestAPIConfig(unittest.TestCase):
    """Test cases for APIConfig class."""

    def test_api_config_defaults(self):
        """Test API configuration default values."""
        config = APIConfig()

        self.assertEqual(config.alpaca_base_url, "https://paper-api.alpaca.markets")
        self.assertEqual(config.requests_per_minute, 200)
        self.assertEqual(config.requests_per_second, 5)


class TestSystemConfig(unittest.TestCase):
    """Test cases for SystemConfig class."""

    def test_system_config_defaults(self):
        """Test system configuration default values."""
        config = SystemConfig()

        self.assertEqual(config.log_level, "INFO")
        self.assertEqual(config.num_workers, 4)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.cache_size, 1000)


if __name__ == '__main__':
    unittest.main()