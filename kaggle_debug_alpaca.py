#!/usr/bin/env python3
"""
Kaggle-compatible Alpaca authentication debug
"""

import os
import sys
import subprocess
import logging

# Kaggle-specific setup
def setup_kaggle_environment():
    """Setup environment for Kaggle"""
    print("🔧 Setting up Kaggle environment...")

    # Install required packages
    packages = [
        "alpaca-trade-api>=3.1.0",
        "yfinance>=0.2.28",
        "pandas>=2.0.0",
        "requests>=2.28.0"
    ]

    for package in packages:
        try:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to install {package}: {e}")

def test_alpaca_auth():
    """Test Alpaca authentication on Kaggle"""

    # API credentials from user
    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "PA3IM0VGKOOM"
    base_url = "https://paper-api.alpaca.markets/v2"

    print(f"🔍 Testing Alpaca Authentication")
    print(f"   API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"   Secret: {secret_key[:4]}...{secret_key[-4:]}")
    print(f"   Base URL: {base_url}")
    print()

    # Set environment variables for security
    os.environ["ALPACA_PAPER_API_KEY"] = api_key
    os.environ["ALPACA_PAPER_SECRET_KEY"] = secret_key

    # Test 1: Using alpaca-trade-api
    print("📋 Test 1: alpaca-trade-api REST")
    try:
        from alpaca_trade_api import REST

        api = REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2'
        )

        # Test account info
        account = api.get_account()
        print(f"✅ Test 1 SUCCESS!")
        print(f"   Account ID: {account.id}")
        print(f"   Account Status: {account.status}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Day Trading Count: {account.daytrade_count}")

        # Test clock
        clock = api.get_clock()
        print(f"   Market: {'🟢 Open' if clock.is_open else '🔴 Closed'}")
        print(f"   Next Open: {clock.next_open}")
        print(f"   Next Close: {clock.next_close}")

        # Test positions
        try:
            positions = api.list_positions()
            print(f"   Open Positions: {len(positions)}")
            for pos in positions[:3]:  # Show first 3
                print(f"     - {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")
        except:
            print(f"   Open Positions: None or Error")

    except Exception as e:
        print(f"❌ Test 1 FAILED: {type(e).__name__}: {str(e)}")

    print()

    # Test 2: Direct HTTP request
    print("📋 Test 2: Direct HTTP Request")
    try:
        import requests

        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': secret_key,
            'Content-Type': 'application/json'
        }

        response = requests.get(f"{base_url}/account", headers=headers, timeout=30)

        print(f"   Status Code: {response.status_code}")

        if response.status_code == 200:
            account_data = response.json()
            print(f"✅ Test 2 SUCCESS!")
            print(f"   Account ID: {account_data.get('id')}")
            print(f"   Account Status: {account_data.get('status')}")
            print(f"   Currency: {account_data.get('currency')}")
        else:
            print(f"❌ Test 2 FAILED: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}...")

    except Exception as e:
        print(f"❌ Test 2 FAILED: {type(e).__name__}: {str(e)}")

    print()

def test_market_data():
    """Test market data access"""
    print("📊 Test 3: Market Data Access")

    try:
        from alpaca_trade_api import REST

        api_key = os.getenv("ALPACA_PAPER_API_KEY")
        secret_key = os.getenv("ALPACA_PAPER_SECRET_KEY")
        base_url = "https://paper-api.alpaca.markets/v2"

        api = REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2'
        )

        # Test getting assets
        assets = api.list_assets(status='active', asset_class='equity')
        print(f"✅ Found {len(assets)} active assets")

        # Test getting bars for popular stocks
        symbols = ["AAPL", "MSFT", "NVDA"]

        print("📈 Recent price data:")
        for symbol in symbols:
            try:
                bars = api.get_latest_bar(symbol)
                print(f"   {symbol}: ${float(bars.close):.2f} (Volume: {int(bars.volume):,})")
            except Exception as e:
                print(f"   {symbol}: Error - {e}")

    except Exception as e:
        print(f"❌ Market data test failed: {e}")

def test_yfinance_fallback():
    """Test yfinance as fallback data source"""
    print("📊 Test 4: Yahoo Finance Fallback")

    try:
        import yfinance as yf

        symbols = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA"]

        print("💹 Current market data from Yahoo Finance:")
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                market_cap = info.get('marketCap')

                if current_price:
                    print(f"   {symbol}: ${current_price:.2f}")
                    if market_cap:
                        print(f"      Market Cap: ${market_cap/1e9:.1f}B")
                else:
                    print(f"   {symbol}: Price data unavailable")

            except Exception as e:
                print(f"   {symbol}: Error - {e}")

    except ImportError:
        print("❌ yfinance not available - installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "--quiet"])
        print("✅ yfinance installed successfully")
    except Exception as e:
        print(f"❌ Yahoo Finance test failed: {e}")

def test_config_loading():
    """Test configuration loading"""
    print("⚙️  Test 5: Configuration Loading")

    try:
        # Add the nca_trading_bot directory to Python path
        bot_dir = "/kaggle/working/nca_trading_bot"
        if bot_dir not in sys.path:
            sys.path.insert(0, bot_dir)

        from config import Config

        # Create config instance
        config = Config()

        print(f"✅ Configuration loaded successfully!")
        print(f"   Top Tickers: {len(config.top_tickers)} stocks")
        print(f"   First 5: {config.top_tickers[:5]}")
        print(f"   NCA Grid Size: {config.nca_grid_size}")
        print(f"   Trading Max Position: {config.trading_max_position_size*100:.1f}%")

        # Test Alpaca config
        alpaca_config = config.get_alpaca_config(paper_mode=True)
        print(f"   Alpaca Base URL: {alpaca_config['base_url']}")
        print(f"   API Key Set: {'Yes' if alpaca_config['key_id'] else 'No'}")
        print(f"   Secret Set: {'Yes' if alpaca_config['secret_key'] else 'No'}")

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

def main():
    """Main debug function"""
    print("🚀 Kaggle Alpaca Authentication Debug")
    print("=" * 60)

    # Setup environment
    setup_kaggle_environment()
    print()

    # Run tests
    test_alpaca_auth()
    test_market_data()
    test_config_loading()
    test_yfinance_fallback()

    print("\n" + "=" * 60)
    print("🏁 Debug Complete")

    # Summary
    print("\n📋 Summary:")
    print("✅ If Test 1 succeeded: Alpaca authentication is working!")
    print("✅ If Test 2 succeeded: Basic HTTP access works")
    print("✅ If Test 4 succeeded: Yahoo Finance is available as fallback")
    print("✅ If Test 5 succeeded: Configuration system is working")

if __name__ == "__main__":
    main()