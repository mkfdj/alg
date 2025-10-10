"""
Simple test script for Kaggle environment
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path.cwd().parent))

def test_imports():
    """Test if we can import the modules"""
    print("🔧 Testing Imports...")

    try:
        # Test basic imports
        import jax
        import jax.numpy as jnp
        import flax
        import optax
        print("✅ JAX/Flax imports successful")

        # Test Alpaca imports
        from alpaca.trading.client import TradingClient
        print("✅ Alpaca imports successful")

        # Test our module imports
        from nca_trading_bot.config import Config
        from nca_trading_bot.nca_model import AdaptiveNCA
        from nca_trading_bot.data_handler import DataHandler
        print("✅ NCA Trading Bot imports successful")

        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_jax():
    """Test JAX functionality"""
    print("\n🔧 Testing JAX...")

    try:
        import jax
        import jax.numpy as jnp

        # Check devices
        devices = jax.devices()
        print(f"✅ Available devices: {len(devices)} x {devices[0].device_kind}")

        # Simple computation
        x = jnp.array([1, 2, 3])
        y = jnp.array([4, 5, 6])
        z = x + y
        print(f"✅ JAX computation: {z}")

        return True

    except Exception as e:
        print(f"❌ JAX error: {e}")
        return False

def test_alpaca():
    """Test Alpaca connection"""
    print("\n🔧 Testing Alpaca...")

    try:
        # Set environment variable
        os.environ["ALPACA_PAPER_API_KEY"] = "PKJ346E2YWMT7HCFZX09"

        from alpaca.trading.client import TradingClient

        trading_client = TradingClient(
            api_key="PKJ346E2YWMT7HCFZX09",
            secret_key="PKJ346E2YWMT7HCFZX09",  # Paper trading uses same key
            paper=True
        )

        account = trading_client.get_account()
        print(f"✅ Alpaca connection successful!")
        print(f"   Account ID: {account.id}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")

        return True

    except Exception as e:
        print(f"❌ Alpaca error: {e}")
        return False

def test_nca_config():
    """Test NCA configuration"""
    print("\n🔧 Testing NCA Configuration...")

    try:
        from nca_trading_bot.config import Config

        config = Config()
        print(f"✅ Config created successfully")
        print(f"   NCA Grid Size: {config.nca_grid_size}")
        print(f"   Top Tickers: {config.top_tickers[:5]}")
        print(f"   Paper Base URL: {config.alpaca_paper_base_url}")

        return True

    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Kaggle Environment Test")
    print("=" * 50)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_jax()
    all_passed &= test_nca_config()
    all_passed &= test_alpaca()

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Ready to start training!")
        print("\nNext steps:")
        print("1. Download datasets: python datasets/download_datasets.py")
        print("2. Analyze data: python nca_trading_bot/main.py --mode analyze")
        print("3. Train model: python nca_trading_bot/main.py --mode train")
    else:
        print("❌ Some tests failed. Check the errors above.")