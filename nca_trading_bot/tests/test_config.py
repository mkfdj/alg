"""
Test configuration module
"""

import pytest
import os
from nca_trading_bot.config import Config


class TestConfig:
    """Test configuration class"""

    def test_config_initialization(self):
        """Test that config initializes with default values"""
        config = Config()

        assert config.nca_grid_size == (64, 64)
        assert config.nca_channels == 16
        assert config.trading_initial_balance == 10000.0
        assert len(config.top_tickers) == 10
        assert "NVDA" in config.top_tickers

    def test_config_validation(self):
        """Test configuration validation"""
        config = Config()

        # Valid config should pass
        assert config.validate() is True

        # Invalid position size should fail
        config.trading_max_position_size = 1.5
        assert config.validate() is False

        # Reset and test invalid risk
        config = Config()
        config.trading_max_risk_per_trade = 0.10
        assert config.validate() is False

    def test_alpaca_config(self):
        """Test Alpaca API configuration"""
        config = Config()

        # Test paper trading config
        paper_config = config.get_alpaca_config(paper_mode=True)
        assert "key_id" in paper_config
        assert "secret_key" in paper_config
        assert "base_url" in paper_config
        assert "paper" in paper_config["base_url"]

        # Test live trading config
        live_config = config.get_alpaca_config(paper_mode=False)
        assert "api.alpaca.markets" in live_config["base_url"]

    def test_environment_variable_override(self):
        """Test environment variable override"""
        # Set environment variables
        os.environ["ALPACA_PAPER_API_KEY"] = "test_key"
        os.environ["ALPACA_PAPER_SECRET_KEY"] = "test_secret"

        config = Config()
        paper_config = config.get_alpaca_config(paper_mode=True)

        assert paper_config["key_id"] == "test_key"
        assert paper_config["secret_key"] == "test_secret"

        # Clean up
        del os.environ["ALPACA_PAPER_API_KEY"]
        del os.environ["ALPACA_PAPER_SECRET_KEY"]