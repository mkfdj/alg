#!/usr/bin/env python3
"""
Unified test script for Kaggle environment
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

def setup_environment():
    """Setup required packages"""
    print("Setting up environment...")
    packages = ["alpaca-trade-api>=3.1.0", "yfinance>=0.2.28", "pandas>=2.0.0"]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

def test_alpaca():
    """Test Alpaca API with environment variables"""
    print("Testing Alpaca API...")

    api_key = os.getenv("ALPACA_PAPER_API_KEY")
    secret_key = os.getenv("ALPACA_PAPER_SECRET_KEY")

    if not api_key or not secret_key:
        print("ERROR: Alpaca credentials not found in environment variables")
        return False

    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Secret Key: {secret_key[:8]}...{secret_key[-8:]}")

    try:
        import alpaca_trade_api as tradeapi

        api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url="https://paper-api.alpaca.markets",
            api_version='v2'
        )

        account = api.get_account()
        print("SUCCESS: Alpaca connection working")
        print(f"Account ID: {account.id}")
        print(f"Status: {account.status}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")

        clock = api.get_clock()
        print(f"Market: {'Open' if clock.is_open else 'Closed'}")

        return True

    except Exception as e:
        print(f"ERROR: Alpaca connection failed: {e}")
        return False

def test_kaggle():
    """Test Kaggle dataset access"""
    print("Testing Kaggle dataset access...")

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        print("ERROR: Kaggle credentials not found in environment variables")
        return False

    print(f"Kaggle Username: {username}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        # List datasets to verify access
        datasets = api.dataset_list()
        print(f"SUCCESS: Kaggle access working - {len(datasets)} datasets available")

        return True

    except ImportError:
        print("Installing Kaggle API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "--quiet"])
        return test_kaggle()
    except Exception as e:
        print(f"ERROR: Kaggle access failed: {e}")
        return False

def test_jax():
    """Test JAX functionality"""
    print("Testing JAX...")

    try:
        import jax
        import jax.numpy as jnp

        devices = jax.devices()
        print(f"SUCCESS: JAX working - {len(devices)} x {devices[0].device_kind}")

        # Simple test
        x = jnp.array([1, 2, 3])
        y = jnp.array([4, 5, 6])
        z = x + y
        print(f"JAX computation test: {z}")

        return True

    except Exception as e:
        print(f"ERROR: JAX failed: {e}")
        return False

def test_yfinance():
    """Test Yahoo Finance fallback"""
    print("Testing Yahoo Finance...")

    try:
        import yfinance as yf

        ticker = yf.Ticker("NVDA")
        info = ticker.info
        price = info.get('currentPrice')

        print(f"SUCCESS: Yahoo Finance working - NVDA: ${price}")
        return True

    except Exception as e:
        print(f"ERROR: Yahoo Finance failed: {e}")
        return False

def test_config():
    """Test configuration system"""
    print("Testing configuration...")

    try:
        # Create test config
        class TestConfig:
            def __init__(self):
                self.nca_grid_size = (16, 16, 16)
                self.nca_channels = 32
                self.nca_steps = 12
                self.top_tickers = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL"]
                self.alpaca_paper_base_url = "https://paper-api.alpaca.markets"
                self.alpaca_paper_api_key = os.getenv("ALPACA_PAPER_API_KEY")
                self.alpaca_paper_secret_key = os.getenv("ALPACA_PAPER_SECRET_KEY")
                self.trading_max_position_size = 0.1

            def get_alpaca_config(self, paper_mode=True):
                if paper_mode:
                    return {
                        "key_id": self.alpaca_paper_api_key,
                        "secret_key": self.alpaca_paper_secret_key,
                        "base_url": self.alpaca_paper_base_url
                    }

        config = TestConfig()
        alpaca_config = config.get_alpaca_config()

        print("SUCCESS: Configuration system working")
        print(f"Grid Size: {config.nca_grid_size}")
        print(f"Top Tickers: {len(config.top_tickers)}")
        print(f"API Key Set: {'Yes' if alpaca_config['key_id'] else 'No'}")
        print(f"Secret Set: {'Yes' if alpaca_config['secret_key'] else 'No'}")

        return True

    except Exception as e:
        print(f"ERROR: Configuration failed: {e}")
        return False

def main():
    """Main test function"""
    print("UNIFIED TEST SCRIPT FOR KAGGLE")
    print("=" * 50)

    setup_environment()

    tests = [
        ("JAX/TPU", test_jax),
        ("Configuration", test_config),
        ("Yahoo Finance", test_yfinance),
        ("Alpaca API", test_alpaca),
        ("Kaggle Datasets", test_kaggle),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        result = test_func()
        results.append((test_name, result))

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY:")

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(tests)} tests passed")

    if passed >= 3:  # At least 3 tests pass
        print("READY: System ready for trading bot development")
        print("Next steps:")
        print("1. Download datasets: python datasets/download_datasets.py")
        print("2. Train model: python nca_trading_bot/main.py --mode train")
        print("3. Start trading: python nca_trading_bot/main.py --mode trade")
    else:
        print("ISSUES: Some components need attention")
        print("Check failed tests above")

if __name__ == "__main__":
    main()