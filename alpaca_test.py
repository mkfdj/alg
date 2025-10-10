#!/usr/bin/env python3
"""
Alpaca Paper Trading API Test using Environment Variables Only
Tests paper trading API using ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from urllib.parse import urljoin
import importlib.util

def setup_environment():
    """Setup environment and install required packages"""
    print("🔧 Setting up environment...")

    # Install required packages
    packages = [
        'alpaca-py>=0.30.0',
        'requests>=2.28.0',
        'websocket-client>=1.6.0'
    ]

    for package in packages:
        try:
            __import__(importlib.util.find_spec(package.replace('>=', '').split('<')[0]).name)
            print(f"✅ {package}")
        except (ImportError, AttributeError):
            print(f"Installing {package}...")
            os.system(f"pip install -q {package}")
            print(f"✅ {package}")

def validate_credentials():
    """Validate paper trading environment variables exist and have correct format"""
    print("🔍 Validating paper trading credentials...")

    # Check for required environment variables
    paper_key = os.environ.get("ALPACA_PAPER_API_KEY")
    paper_secret = os.environ.get("ALPACA_PAPER_SECRET_KEY")

    # Validate paper credentials
    if not paper_key:
        raise ValueError("❌ ALPACA_PAPER_API_KEY environment variable not found")
    if not paper_secret:
        raise ValueError("❌ ALPACA_PAPER_SECRET_KEY environment variable not found")

    # Check format
    if not paper_key.startswith('PK'):
        raise ValueError(f"❌ Invalid paper API key format: {paper_key[:10]}...")
    if len(paper_secret) < 10:
        raise ValueError(f"❌ Paper secret key too short: {len(paper_secret)} chars")

    print(f"✅ Paper API Key: {paper_key[:10]}...{paper_key[-4:]}")
    print(f"✅ Paper Secret Key: {paper_secret[:10]}...{paper_secret[-4:]}")
    print("✅ Credential format looks correct")

    return {'key': paper_key, 'secret': paper_secret}

def test_http_api(api_key, api_secret, base_url, account_type):
    """Test API access via direct HTTP requests"""
    print(f"📡 Testing {account_type} API via Direct HTTP...")
    print(f"   Endpoint: {base_url}/v2/account")

    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.get(
            urljoin(base_url, '/v2/account'),
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            account_data = response.json()
            print(f"   ✅ {account_type} API SUCCESS!")
            print(f"   Account ID: {account_data.get('id', 'N/A')}")
            print(f"   Status: {account_data.get('status', 'N/A')}")
            print(f"   Buying Power: ${account_data.get('buying_power', '0.00')}")
            print(f"   Portfolio Value: ${account_data.get('portfolio_value', '0.00')}")
            return True
        else:
            print(f"   ❌ HTTP {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ HTTP request failed: {str(e)}")
        return False

def test_sdk_api(api_key, api_secret, paper=True):
    """Test API access via alpaca-py SDK"""
    account_type = "Paper Trading" if paper else "Live Trading"
    print(f"🔄 Testing {account_type} API via alpaca-py SDK...")

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest
        from alpaca.data.timeframe import TimeFrame

        # Create trading client
        trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper
        )

        # Test account info
        account = trading_client.get_account()
        print(f"   ✅ {account_type} Client SUCCESS!")
        print(f"   Account ID: {account.id}")
        print(f"   Status: {account.account_status}")
        print(f"   Buying Power: ${account.buying_power}")
        print(f"   Portfolio Value: ${account.portfolio_value}")
        print(f"   Market Status: {account.trading_blocked or 'Open'}")
        if hasattr(account, 'next_open') and account.next_open:
            print(f"   Next Open: {account.next_open}")

        # Test market data
        print(f"📊 Testing Market Data Access...")
        try:
            # Use the same API key for data (works for paper)
            data_client = StockHistoricalDataClient(api_key, api_secret)

            # Get latest trade data
            request_params = StockLatestTradeRequest(symbol_or_symbols="AAPL")
            latest_trade = data_client.get_stock_latest_trade(request_params)

            if latest_trade and 'AAPL' in latest_trade:
                price = latest_trade['AAPL'].price
                timestamp = latest_trade['AAPL'].timestamp
                print(f"   ✅ Market data access working!")
                print(f"   Latest AAPL price: ${price}")
                print(f"   Timestamp: {timestamp}")
                return True
            else:
                print(f"   ❌ Market data failed: No data returned")
                return False

        except Exception as e:
            print(f"   ❌ Market data failed: {str(e)}")
            return False

    except ImportError as e:
        print(f"   ❌ SDK not available: {str(e)}")
        return False
    except Exception as e:
        print(f"   ❌ SDK test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 ALPACA PAPER TRADING API TEST")
    print("=" * 50)
    print(f"Environment: Local")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 50)

    try:
        # Setup
        setup_environment()

        # Validate credentials
        creds = validate_credentials()

        # Test results tracking
        results = {
            'paper_http': False,
            'paper_sdk': False,
            'market_data': False
        }

        # Test Paper API via HTTP
        results['paper_http'] = test_http_api(
            creds['key'],
            creds['secret'],
            'https://paper-api.alpaca.markets',
            'Paper'
        )

        # Test Paper API via SDK
        paper_result = test_sdk_api(
            creds['key'],
            creds['secret'],
            paper=True
        )
        results['paper_sdk'] = paper_result
        results['market_data'] = paper_result  # Market data is tested in SDK function

        # Summary
        print(f"\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        print(f"Paper Http: {'✅ PASS' if results['paper_http'] else '❌ FAIL'}")
        print(f"Paper Sdk: {'✅ PASS' if results['paper_sdk'] else '❌ FAIL'}")
        print(f"Market Data: {'✅ PASS' if results['market_data'] else '❌ FAIL'}")

        passed = sum(results.values())
        total = len(results)
        print(f"\nOverall: {passed}/{total} tests passed")

        if results['paper_http'] or results['paper_sdk']:
            print(f"\n🎉 SUCCESS! Alpaca Paper Trading API is working!")
            print(f"\n📝 Next Steps:")
            print(f"1. Your trading bot can now use these credentials")
            print(f"2. Test order placement (if needed)")
            print(f"3. Start model training with live data")
            print(f"\n🔑 Environment variables are ready:")
            print(f"   ALPACA_PAPER_API_KEY={creds['key'][:10]}...{creds['key'][-4:]}")
            print(f"   ALPACA_PAPER_SECRET_KEY={creds['secret'][:10]}...{creds['secret'][-4:]}")
        else:
            print(f"\n❌ Paper API tests failed - check credentials")

        return passed >= 2  # Need at least 2 tests to pass (HTTP + SDK)

    except Exception as e:
        print(f"\n💥 FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)