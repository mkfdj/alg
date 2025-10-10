#!/usr/bin/env python3

import sys
sys.path.append('.')

from nca_trading_bot.config import Config

def test_config():
    """Test that the config initializes properly"""
    print("Testing configuration...")

    # Create config instance
    config = Config()

    print(f"✅ Top tickers: {len(config.top_tickers)} tickers")
    print(f"✅ Datasets: {list(config.datasets.keys())}")
    print(f"✅ Technical indicators: {list(config.technical_indicators.keys())}")
    print(f"✅ Risk management: {list(config.risk_management.keys())}")

    # Test Kaggle dataset configuration
    kaggle_config = config.datasets.get("kaggle_stock_market")
    if kaggle_config:
        print(f"✅ Kaggle dataset path: {kaggle_config.get('path')}")
        print(f"✅ Kaggle dataset structure: {kaggle_config.get('structure')}")
    else:
        print("❌ Kaggle dataset configuration not found!")

    print("Configuration test completed successfully!")

if __name__ == "__main__":
    test_config()