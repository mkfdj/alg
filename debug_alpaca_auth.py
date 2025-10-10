#!/usr/bin/env python3
"""
Debug Alpaca authentication issues
Testing different approaches with alpaca-trade-api
"""

import os
import sys
from alpaca_trade_api import REST, TimeFrame
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_alpaca_auth_approaches():
    """Test different Alpaca authentication approaches"""

    # API credentials from user
    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "PA3IM0VGKOOM"
    base_url = "https://paper-api.alpaca.markets/v2"

    print(f"🔍 Testing Alpaca Authentication")
    print(f"   API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"   Secret: {secret_key[:4]}...{secret_key[-4:]}")
    print(f"   Base URL: {base_url}")
    print()

    # Test 1: Using REST with direct parameters
    print("📋 Test 1: REST with direct parameters")
    try:
        api = REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2'
        )

        # Test account info
        account = api.get_account()
        print(f"✅ Test 1 SUCCESS: Account ID: {account.id}")
        print(f"   Account Status: {account.status}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")

        # Test clock
        clock = api.get_clock()
        print(f"   Market: {'Open' if clock.is_open else 'Closed'}")

    except Exception as e:
        print(f"❌ Test 1 FAILED: {type(e).__name__}: {str(e)}")

    print()

    # Test 2: Using REST with headers (alternative approach)
    print("📋 Test 2: REST with custom headers")
    try:
        import requests

        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': secret_key,
            'Content-Type': 'application/json'
        }

        response = requests.get(f"{base_url}/account", headers=headers)

        if response.status_code == 200:
            account_data = response.json()
            print(f"✅ Test 2 SUCCESS: Account ID: {account_data.get('id')}")
            print(f"   Account Status: {account_data.get('status')}")
        else:
            print(f"❌ Test 2 FAILED: Status {response.status_code}")
            print(f"   Response: {response.text}")

    except Exception as e:
        print(f"❌ Test 2 FAILED: {type(e).__name__}: {str(e)}")

    print()

    # Test 3: Using alpaca-py if available (for comparison)
    print("📋 Test 3: Testing with alpaca-py (if available)")
    try:
        from alpaca import TradingClient
        from alpaca.trading.requests import GetAccountRequest

        trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True
        )

        account = trading_client.get_account(GetAccountRequest())
        print(f"✅ Test 3 SUCCESS: Account ID: {account.id}")
        print(f"   Account Status: {account.account_status}")

    except ImportError:
        print("⚠️  Test 3: alpaca-py not available")
    except Exception as e:
        print(f"❌ Test 3 FAILED: {type(e).__name__}: {str(e)}")

def test_environment_variables():
    """Test environment variable approach"""
    print("\n🌍 Testing Environment Variable Approach")

    # Set environment variables
    os.environ["ALPACA_PAPER_API_KEY"] = "PKJ346E2YWMT7HCFZX09"
    os.environ["ALPACA_PAPER_SECRET_KEY"] = "PA3IM0VGKOOM"

    # Test config loading
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'nca_trading_bot'))
        from config import config

        alpaca_config = config.get_alpaca_config(paper_mode=True)
        print(f"✅ Environment variables loaded:")
        print(f"   API Key: {alpaca_config['key_id'][:8]}...{alpaca_config['key_id'][-4:]}")
        print(f"   Secret: {alpaca_config['secret_key'][:4]}...{alpaca_config['secret_key'][-4:]}")
        print(f"   Base URL: {alpaca_config['base_url']}")

    except Exception as e:
        print(f"❌ Environment variable test failed: {e}")

def test_data_sources():
    """Test alternative data sources"""
    print("\n📊 Testing Alternative Data Sources")

    try:
        import yfinance as yf
        import pandas as pd

        # Test getting data for top tickers
        tickers = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL"]

        for ticker in tickers[:3]:  # Test first 3
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                print(f"✅ {ticker}: {info.get('shortName', 'N/A')} - ${info.get('currentPrice', 'N/A')}")
            except Exception as e:
                print(f"❌ {ticker}: Error - {e}")

    except ImportError:
        print("❌ yfinance not available")
    except Exception as e:
        print(f"❌ Data source test failed: {e}")

if __name__ == "__main__":
    print("🚀 Alpaca Authentication Debug")
    print("=" * 50)

    test_alpaca_auth_approaches()
    test_environment_variables()
    test_data_sources()

    print("\n" + "=" * 50)
    print("🏁 Debug Complete")