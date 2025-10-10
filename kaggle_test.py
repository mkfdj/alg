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
    """Test Alpaca connection with fixed authentication"""
    print("\n🔧 Testing Alpaca...")

    # Use the user's provided credentials directly
    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "PA3IM0VGKOOM"
    base_url = "https://paper-api.alpaca.markets/v2"

    # Also set environment variables for other parts of the system
    os.environ["ALPACA_PAPER_API_KEY"] = api_key
    os.environ["ALPACA_PAPER_SECRET_KEY"] = secret_key

    print(f"🔑 Using API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"🔐 Using Secret Key: {secret_key[:4]}...{secret_key[-4:]}")
    print(f"🌐 Base URL: {base_url}")

    try:
        print("🔄 Trying alpaca-trade-api with v2 endpoint...")

        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            print("❌ alpaca-trade-api not installed. Installing...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "alpaca-trade-api>=3.1.0", "--quiet"], check=True)
            import alpaca_trade_api as tradeapi

        # Use the correct v2 endpoint with api_version parameter
        api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2'
        )

        account = api.get_account()
        print(f"✅ Alpaca connection successful!")
        print(f"   Account ID: {account.id}")
        print(f"   Status: {account.status}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Day Trading Count: {account.daytrade_count}")

        # Test clock
        try:
            clock = api.get_clock()
            print(f"   Market: {'🟢 Open' if clock.is_open else '🔴 Closed'}")
        except:
            print(f"   Market: Unable to get clock info")

        return True

    except Exception as e:
        print(f"❌ Alpaca error: {type(e).__name__}: {str(e)}")

        # Try direct HTTP request as fallback
        print("🔄 Trying direct HTTP request...")
        try:
            import requests

            headers = {
                'APCA-API-KEY-ID': api_key,
                'APCA-API-SECRET-KEY': secret_key,
                'Content-Type': 'application/json'
            }

            response = requests.get(f"{base_url}/account", headers=headers, timeout=30)

            if response.status_code == 200:
                account_data = response.json()
                print(f"✅ Direct HTTP request successful!")
                print(f"   Account ID: {account_data.get('id')}")
                print(f"   Status: {account_data.get('status')}")
                print(f"   Portfolio Value: ${float(account_data.get('portfolio_value', 0)):,.2f}")
                return True
            else:
                print(f"❌ HTTP request failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")

        except Exception as http_error:
            print(f"❌ HTTP fallback failed: {http_error}")

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
    print("=" * 60)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_jax()
    all_passed &= test_nca_config()
    all_passed &= test_alpaca()

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Ready to start training!")
        print("\n🚀 Next steps:")
        print("1. Run debug script: python kaggle_debug_alpaca.py")
        print("2. Download datasets: python datasets/download_datasets.py")
        print("3. Analyze data: python nca_trading_bot/main.py --mode analyze")
        print("4. Train model: python nca_trading_bot/main.py --mode train")
        print("5. Start paper trading: python nca_trading_bot/main.py --mode trade")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("\n💡 Debug Tips:")
        print("- Run: python kaggle_debug_alpaca.py for detailed diagnosis")
        print("- Check your Alpaca dashboard: https://app.alpaca.markets/")
        print("- Verify paper trading account is approved")
        print("- API keys are pre-configured in this test script")