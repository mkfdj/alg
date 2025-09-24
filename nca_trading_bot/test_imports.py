#!/usr/bin/env python3
"""
Test script to verify imports work correctly.
"""

def test_imports():
    """Test all required imports."""
    try:
        # Test alpaca-py imports
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        # from alpaca.stream import Stream  # Not available in current version
        print("✓ Alpaca-py imports successful")

        # Test gymnasium
        import gymnasium as gym
        print("✓ Gymnasium import successful")

        # Test other imports
        import numpy as np
        import pandas as pd
        import torch
        import yfinance as yf
        print("✓ Core dependencies imports successful")

        # Test local imports
        from config import get_config
        from nca_model import NCATradingModel
        print("✓ Local module imports successful")

        print("\n🎉 All imports successful!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports()