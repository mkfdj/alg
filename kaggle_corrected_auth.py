#!/usr/bin/env python3
"""
Corrected Alpaca Authentication based on Official Documentation
Based on https://docs.alpaca.markets/docs/authentication-1
"""

import os
import sys
import subprocess
import requests
import json
from datetime import datetime

def setup_environment():
    """Setup environment with required packages"""
    print("🔧 Setting up environment...")

    packages = [
        "alpaca-py>=0.30.0",
        "alpaca-trade-api>=3.1.0",
        "requests>=2.28.0"
    ]

    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️  {package} already installed or failed")

def test_direct_http_correct():
    """Test direct HTTP with correct headers based on official docs"""
    print("\n🔍 Testing Direct HTTP (Official Documentation Method)...")

    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "PA3IM0VGKOOM"

    # Correct headers from official documentation
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key,
        'Content-Type': 'application/json'
    }

    # Test the exact endpoint from docs
    endpoints = [
        "https://paper-api.alpaca.markets/v2/account",
        "https://paper-api.alpaca.markets/v2/clock",
        "https://paper-api.alpaca.markets/v2/positions"
    ]

    for endpoint in endpoints:
        print(f"🔄 Testing {endpoint}...")
        try:
            response = requests.get(endpoint, headers=headers, timeout=30)

            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS: {endpoint}")

                if "account" in endpoint:
                    print(f"   Account ID: {data.get('id')}")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Buying Power: ${float(data.get('buying_power', 0)):,.2f}")
                    print(f"   Portfolio Value: ${float(data.get('portfolio_value', 0)):,.2f}")
                elif "clock" in endpoint:
                    print(f"   Market: {'Open' if data.get('is_open') else 'Closed'}")
                    print(f"   Timestamp: {data.get('timestamp')}")
                elif "positions" in endpoint:
                    print(f"   Positions: {len(data)} open positions")

                return True
            else:
                print(f"❌ FAILED: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('message', 'Unknown error')}")
                except:
                    print(f"   Response: {response.text[:200]}...")

        except Exception as e:
            print(f"❌ ERROR: {type(e).__name__}: {str(e)}")

    return False

def test_alpaca_trade_api_correct():
    """Test alpaca-trade-api with correct configuration"""
    print("\n🔍 Testing alpaca-trade-api (Correct Method)...")

    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "PA3IM0VGKOOM"

    try:
        import alpaca_trade_api as tradeapi
        print("✅ alpaca-trade-api imported")

        # Use the correct base URL from documentation
        api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url="https://paper-api.alpaca.markets",  # NO /v2 here
            api_version='v2'  # But specify v2 version
        )

        # Test account
        account = api.get_account()
        print(f"✅ Account SUCCESS!")
        print(f"   ID: {account.id}")
        print(f"   Status: {account.status}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Pattern Day Trader: {account.pattern_day_trader}")
        print(f"   Trading Blocked: {account.trading_blocked}")

        # Test clock
        clock = api.get_clock()
        print(f"   Market: {'Open' if clock.is_open else 'Closed'}")
        print(f"   Next Open: {clock.next_open}")
        print(f"   Next Close: {clock.next_close}")

        # Test positions
        try:
            positions = api.list_positions()
            print(f"   Positions: {len(positions)} open")
            for pos in positions[:3]:
                print(f"     - {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f}")
        except:
            print(f"   Positions: Unable to retrieve")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        return False

def test_alpaca_py_correct():
    """Test alpaca-py with correct usage"""
    print("\n🔍 Testing alpaca-py (Correct Method)...")

    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "PA3IM0VGKOOM"

    try:
        from alpaca.trading.client import TradingClient
        print("✅ TradingClient imported")

        # Create trading client for paper trading
        trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True  # This should handle the paper endpoint automatically
        )

        # Get account - try different methods
        try:
            # Method 1: Direct call
            account = trading_client.get_account()
            print(f"✅ TradingClient SUCCESS!")
            print(f"   Account ID: {account.id}")
            print(f"   Account Status: {account.account_status}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")

            return True

        except Exception as e:
            print(f"❌ Direct account call failed: {e}")

            # Method 2: Try with request object
            try:
                from alpaca.trading.requests import GetAccountRequest
                account = trading_client.get_account(GetAccountRequest())
                print(f"✅ TradingClient with Request SUCCESS!")
                print(f"   Account ID: {account.id}")
                print(f"   Account Status: {account.account_status}")
                return True

            except ImportError as ie:
                print(f"❌ Import error for GetAccountRequest: {ie}")
            except Exception as e2:
                print(f"❌ Request method also failed: {e2}")

        return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        return False

def test_basic_connectivity():
    """Test basic connectivity to Alpaca servers"""
    print("\n🔍 Testing Basic Connectivity...")

    endpoints = [
        "https://paper-api.alpaca.markets",
        "https://api.alpaca.markets"
    ]

    for endpoint in endpoints:
        print(f"🔄 Pinging {endpoint}...")
        try:
            response = requests.get(endpoint, timeout=10)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print(f"✅ Server reachable")
            else:
                print(f"⚠️  Server responded with {response.status_code}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")

def analyze_api_key_format():
    """Analyze the API key format"""
    print("\n🔍 Analyzing API Key Format...")

    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "PA3IM0VGKOOM"

    print(f"API Key: {api_key}")
    print(f"  Length: {len(api_key)}")
    print(f"  Format: {'Valid (starts with PK)' if api_key.startswith('PK') and len(api_key) >= 16 else 'Invalid'}")

    print(f"Secret Key: {secret_key}")
    print(f"  Length: {len(secret_key)}")
    print(f"  Format: {'Valid (length >= 16)' if len(secret_key) >= 16 else 'Invalid'}")

    # Check if these match typical Alpaca patterns
    if api_key.startswith('PK') and len(api_key) == 20:
        print("✅ API Key appears to be valid Alpaca format")
    else:
        print("⚠️  API Key may not be in standard Alpaca format")

    if len(secret_key) >= 16:
        print("✅ Secret Key appears to be valid length")
    else:
        print("⚠️  Secret Key may be too short")

def main():
    """Main test function"""
    print("🚀 Corrected Alpaca Authentication Test")
    print("=" * 60)
    print("Based on official Alpaca documentation")
    print("https://docs.alpaca.markets/docs/authentication-1")
    print("=" * 60)

    # Setup
    setup_environment()

    # Analyze credentials
    analyze_api_key_format()

    # Test connectivity
    test_basic_connectivity()

    # Run tests in order of preference
    results = []

    print("\n" + "=" * 60)
    print("🧪 AUTHENTICATION TESTS")
    print("=" * 60)

    results.append(("Direct HTTP (Official)", test_direct_http_correct()))
    results.append(("alpaca-trade-api", test_alpaca_trade_api_correct()))
    results.append(("alpaca-py", test_alpaca_py_correct()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)

    success_count = 0
    for test_name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"{test_name:25} : {status}")
        if result:
            success_count += 1

    if success_count > 0:
        print(f"\n🎉 {success_count} authentication method(s) working!")
        print("\n🚀 You can proceed with the working method!")
        print("   Use this configuration in your trading bot.")
    else:
        print(f"\n❌ All authentication methods failed")
        print("\n🔧 Troubleshooting:")
        print("   1. Check Alpaca dashboard: https://app.alpaca.markets/")
        print("   2. Verify paper trading account is active")
        print("   3. Generate new API keys if needed")
        print("   4. Ensure account is approved for paper trading")
        print("   5. Use Yahoo Finance fallback for now")

if __name__ == "__main__":
    main()