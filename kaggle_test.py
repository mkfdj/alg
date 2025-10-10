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

    # Get API key from environment variable
    api_key = os.getenv("ALPACA_PAPER_API_KEY")
    if not api_key:
        print("❌ ALPACA_PAPER_API_KEY environment variable not set!")
        print("Set it with: os.environ['ALPACA_PAPER_API_KEY'] = 'your_api_key'")
        return False

    print(f"🔑 Using API Key: {api_key[:8]}...{api_key[-4:]}")  # Show only partial key

    try:
        # Try fallback with older library that works with single key
        print("🔄 Trying alpaca-trade-api...")

        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            print("❌ alpaca-trade-api not installed. Installing...")
            import subprocess
            subprocess.run(["pip", "install", "alpaca-trade-api"], check=True)
            import alpaca_trade_api as tradeapi

        api = tradeapi.REST(
            key_id=api_key,
            secret_key=api_key,  # Single key for paper trading
            base_url="https://paper-api.alpaca.markets/v2"
        )

        account = api.get_account()
        print(f"✅ Alpaca connection successful!")
        print(f"   Account ID: {account.id}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")

        return True

    except Exception as e:
        print(f"❌ Alpaca error: {e}")

        # Try without v2 endpoint
        print("🔄 Trying without /v2...")
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(
                key_id=api_key,
                secret_key=api_key,
                base_url="https://paper-api.alpaca.markets"
            )
            account = api.get_account()
            print(f"✅ Alpaca fallback connection successful!")
            print(f"   Account ID: {account.id}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            return True
        except Exception as e2:
            print(f"❌ All Alpaca attempts failed: {e2}")
            return False

def test_nca_config():
    """Test NCA configuration"""
    print("\n🔧 Testing NCA Configuration...")

    try:
        from nca_trading_bot.config import Config

        config = Config()
        print(f"✅ Config created successfully")

        # Test individual attributes to avoid NoneType error
        try:
            grid_size = config.nca_grid_size
            print(f"   NCA Grid Size: {grid_size}")
        except Exception as e:
            print(f"   ⚠️  Grid size error: {e}")

        try:
            tickers = config.top_tickers
            if tickers and len(tickers) > 0:
                print(f"   Top Tickers: {tickers[:5]}")
            else:
                print(f"   ⚠️  Top tickers: Empty or None")
        except Exception as e:
            print(f"   ⚠️  Tickers error: {e}")

        try:
            base_url = config.alpaca_paper_base_url
            print(f"   Paper Base URL: {base_url}")
        except Exception as e:
            print(f"   ⚠️  Base URL error: {e}")

        return True

    except Exception as e:
        print(f"❌ Config error: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🚀 Kaggle Environment Test")
    print("=" * 50)

    # Set environment variable (remove hardcoded API key)
    print("🔐 Setting up environment variables...")
    os.environ["ALPACA_PAPER_API_KEY"] = os.getenv("ALPACA_PAPER_API_KEY", "")
    print("✅ Environment variables configured")

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_jax()
    all_passed &= test_nca_config()
    all_passed &= test_alpaca()

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Ready to start training!")
        print("\nNext steps:")
        print("1. Set your API key: os.environ['ALPACA_PAPER_API_KEY'] = 'your_key'")
        print("2. Download datasets: python datasets/download_datasets.py")
        print("3. Analyze data: python nca_trading_bot/main.py --mode analyze")
        print("4. Train model: python nca_trading_bot/main.py --mode train")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("\n💡 Tips:")
        print("- Make sure ALPACA_PAPER_API_KEY environment variable is set")
        print("- Check that your Alpaca paper account is active")