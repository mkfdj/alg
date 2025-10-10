#!/usr/bin/env python3
"""
Comprehensive Alpaca Authentication Diagnostic for Kaggle
"""

import os
import sys
import subprocess
import requests
import json
from datetime import datetime

def setup_environment():
    """Setup environment with all required packages"""
    print("🔧 Setting up Kaggle environment...")

    packages = [
        "alpaca-py>=0.30.0",
        "alpaca-trade-api>=3.1.0",
        "yfinance>=0.2.28",
        "pandas>=2.0.0",
        "requests>=2.28.0",
        "websocket-client>=1.6.0"
    ]

    for package in packages:
        try:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def test_alpaca_py():
    """Test alpaca-py (newer package)"""
    print("\n🔍 Testing alpaca-py (newer package)...")

    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "PA3IM0VGKOOM"

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAccountRequest
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        print("✅ alpaca-py imports successful")

        # Test trading client
        print("🔄 Testing TradingClient...")
        trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True
        )

        account = trading_client.get_account(GetAccountRequest())
        print(f"✅ TradingClient SUCCESS!")
        print(f"   Account ID: {account.id}")
        print(f"   Account Status: {account.account_status}")
        print(f"   Account Type: {account.account_type}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Trading Blocked: {'Yes' if account.trading_blocked else 'No'}")
        print(f"   Account Created: {account.created_at}")

        # Test data client
        print("🔄 Testing StockHistoricalDataClient...")
        data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

        request_params = StockBarsRequest(
            symbol_or_symbols=["AAPL"],
            timeframe=TimeFrame.Day,
            start=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            limit=1
        )

        bars = data_client.get_stock_bars(request_params)
        if bars.data:
            print(f"✅ DataClient SUCCESS!")
            for symbol, bar_list in bars.data.items():
                for bar in bar_list:
                    print(f"   {symbol}: ${float(bar.close):.2f}")
        else:
            print("⚠️  DataClient: No data returned (might be market hours)")

        return True

    except ImportError as e:
        print(f"❌ alpaca-py import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ alpaca-py test failed: {type(e).__name__}: {str(e)}")
        return False

def test_alpaca_trade_api():
    """Test alpaca-trade-api (older package)"""
    print("\n🔍 Testing alpaca-trade-api (older package)...")

    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "PA3IM0VGKOOM"

    try:
        import alpaca_trade_api as tradeapi
        print("✅ alpaca-trade-api imports successful")

        # Test different endpoints and configurations
        configs = [
            {
                "name": "v2 endpoint with api_version",
                "base_url": "https://paper-api.alpaca.markets/v2",
                "api_version": "v2"
            },
            {
                "name": "v2 endpoint without api_version",
                "base_url": "https://paper-api.alpaca.markets/v2",
                "api_version": None
            },
            {
                "name": "base endpoint",
                "base_url": "https://paper-api.alpaca.markets",
                "api_version": None
            }
        ]

        for config in configs:
            print(f"🔄 Testing {config['name']}...")
            try:
                kwargs = {
                    "key_id": api_key,
                    "secret_key": secret_key,
                    "base_url": config["base_url"]
                }
                if config["api_version"]:
                    kwargs["api_version"] = config["api_version"]

                api = tradeapi.REST(**kwargs)
                account = api.get_account()

                print(f"✅ {config['name']} SUCCESS!")
                print(f"   Account ID: {account.id}")
                print(f"   Status: {account.status}")
                print(f"   Buying Power: ${float(account.buying_power):,.2f}")
                print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")

                # Test clock
                try:
                    clock = api.get_clock()
                    print(f"   Market: {'Open' if clock.is_open else 'Closed'}")
                except:
                    print(f"   Market: Unable to get clock")

                return True

            except Exception as e:
                print(f"❌ {config['name']} failed: {type(e).__name__}: {str(e)}")

        return False

    except ImportError as e:
        print(f"❌ alpaca-trade-api import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ alpaca-trade-api test failed: {type(e).__name__}: {str(e)}")
        return False

def test_direct_http():
    """Test direct HTTP requests to Alpaca API"""
    print("\n🔍 Testing Direct HTTP Requests...")

    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "PA3IM0VGKOOM"

    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key,
        'Content-Type': 'application/json'
    }

    # Test different endpoints
    endpoints = [
        "https://paper-api.alpaca.markets/v2/account",
        "https://paper-api.alpaca.markets/account",
        "https://api.alpaca.markets/v2/account"  # Live endpoint for comparison
    ]

    for endpoint in endpoints:
        print(f"🔄 Testing {endpoint}...")
        try:
            response = requests.get(endpoint, headers=headers, timeout=30)

            print(f"   Status Code: {response.status_code}")

            if response.status_code == 200:
                account_data = response.json()
                print(f"✅ {endpoint} SUCCESS!")
                print(f"   Account ID: {account_data.get('id')}")
                print(f"   Status: {account_data.get('status', 'N/A')}")
                print(f"   Portfolio Value: ${float(account_data.get('portfolio_value', 0)):,.2f}")
                return True
            else:
                print(f"❌ {endpoint} failed")
                print(f"   Response: {response.text[:200]}...")

        except Exception as e:
            print(f"❌ {endpoint} error: {type(e).__name__}: {str(e)}")

    return False

def test_api_key_format():
    """Test API key format and validity"""
    print("\n🔍 Analyzing API Key Format...")

    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "PA3IM0VGKOOM"

    print(f"API Key: {api_key}")
    print(f"  Length: {len(api_key)}")
    print(f"  Starts with 'PK': {api_key.startswith('PK')}")
    print(f"  Format: {'Valid' if len(api_key) == 20 and api_key.startswith('PK') else 'Invalid'}")

    print(f"Secret Key: {secret_key}")
    print(f"  Length: {len(secret_key)}")
    print(f"  Format: {'Valid' if len(secret_key) >= 16 else 'Invalid'}")

    # Check if these look like valid Alpaca credentials
    if len(api_key) == 20 and api_key.startswith('PK') and len(secret_key) >= 16:
        print("✅ API credentials appear to be in correct format")
        return True
    else:
        print("❌ API credentials may be in incorrect format")
        return False

def test_account_permissions():
    """Check what permissions the account might have"""
    print("\n🔍 Checking Account Permissions...")

    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "PA3IM0VGKOOM"

    # Test various endpoints to check permissions
    test_endpoints = [
        ("/v2/account", "Account Info"),
        ("/v2/positions", "Positions"),
        ("/v2/orders", "Orders"),
        ("/v2/assets", "Assets"),
        ("/v2/clock", "Clock"),
        ("/v2/calendar", "Calendar")
    ]

    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key,
        'Content-Type': 'application/json'
    }

    base_url = "https://paper-api.alpaca.markets"
    working_endpoints = []

    for endpoint, description in test_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", headers=headers, timeout=30)
            if response.status_code == 200:
                print(f"✅ {description}: Access Granted")
                working_endpoints.append(endpoint)
            elif response.status_code == 401:
                print(f"❌ {description}: Unauthorized")
            elif response.status_code == 403:
                print(f"⚠️  {description}: Forbidden (permission issue)")
            else:
                print(f"⚠️  {description}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {description}: Error - {e}")

    if working_endpoints:
        print(f"\n✅ {len(working_endpoints)} endpoints accessible")
        print("   Account has some permissions, authentication may be working")
        return True
    else:
        print("\n❌ No endpoints accessible - authentication or permissions issue")
        return False

def main():
    """Main diagnostic function"""
    print("🚀 Comprehensive Alpaca Authentication Diagnostic")
    print("=" * 70)

    # Setup
    setup_environment()

    # Run all tests
    results = []

    results.append(("API Key Format", test_api_key_format()))
    results.append(("Direct HTTP", test_direct_http()))
    results.append(("alpaca-py", test_alpaca_py()))
    results.append(("alpaca-trade-api", test_alpaca_trade_api()))
    results.append(("Account Permissions", test_account_permissions()))

    # Summary
    print("\n" + "=" * 70)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 70)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} : {status}")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")

    if any(result for _, result in results):
        print("✅ Some tests passed - the system is working!")
        print("\n🚀 Next steps:")
        print("1. Use the working authentication method")
        print("2. Run: python nca_trading_bot/main.py --mode analyze")
        print("3. Download training datasets")
        print("4. Train the NCA model")
    else:
        print("❌ All authentication tests failed")
        print("\n🔧 Troubleshooting:")
        print("1. Check Alpaca dashboard: https://app.alpaca.markets/")
        print("2. Verify paper trading account is approved")
        print("3. Generate new API keys if needed")
        print("4. Check if account needs to be funded for paper trading")
        print("5. Contact Alpaca support if issues persist")
        print("6. Use Yahoo Finance fallback for now")

if __name__ == "__main__":
    main()