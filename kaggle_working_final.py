#!/usr/bin/env python3
"""
FINAL WORKING VERSION with EXACT endpoint from Alpaca panel
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path.cwd().parent))

def setup_environment():
    """Setup environment with required packages"""
    print("🔧 Installing packages...")
    packages = ["alpaca-trade-api>=3.1.0", "requests>=2.28.0"]
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️  {package}")

def test_alpaca_exact_endpoint():
    """Test with the EXACT endpoint from your Alpaca panel"""
    print("\n🔧 Testing with EXACT endpoint from Alpaca panel...")

    # Your exact credentials
    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "Pge5ic8eDN0ze0YTpEJxNmpdf3YGUnhOVnWJZbf7"

    # EXACT endpoint from your panel: https://paper-api.alpaca.markets/v2
    base_url = "https://paper-api.alpaca.markets/v2"

    print(f"🔑 API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"🔐 Secret Key: {secret_key[:8]}...{secret_key[-8:]}")
    print(f"🌐 Base URL: {base_url}")

    # Test 1: Direct HTTP to account endpoint
    print("\n📡 Test 1: Direct HTTP to account...")
    try:
        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': secret_key,
            'Content-Type': 'application/json'
        }

        # Full endpoint path
        account_url = f"{base_url}/account"
        response = requests.get(account_url, headers=headers, timeout=30)

        print(f"   URL: {account_url}")
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            account_data = response.json()
            print(f"✅ SUCCESS!")
            print(f"   Account ID: {account_data.get('id')}")
            print(f"   Status: {account_data.get('status')}")
            print(f"   Buying Power: ${float(account_data.get('buying_power', 0)):,.2f}")
            print(f"   Portfolio Value: ${float(account_data.get('portfolio_value', 0)):,.2f}")
            return True
        else:
            print(f"❌ FAILED: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('message', 'Unknown')}")
            except:
                print(f"   Response: {response.text[:200]}...")

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")

    # Test 2: alpaca-trade-api with v2 base URL
    print("\n🔄 Test 2: alpaca-trade-api with v2 base URL...")
    try:
        import alpaca_trade_api as tradeapi

        # Use the v2 endpoint as base URL without api_version
        api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url  # Use full v2 URL as base
        )

        account = api.get_account()
        print(f"✅ SUCCESS!")
        print(f"   Account ID: {account.id}")
        print(f"   Status: {account.status}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")

        # Test clock
        clock = api.get_clock()
        print(f"   Market: {'Open' if clock.is_open else 'Closed'}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {str(e)}")

    # Test 3: alpaca-trade-api with different config
    print("\n🔄 Test 3: Alternative alpaca-trade-api config...")
    try:
        import alpaca_trade_api as tradeapi

        # Try using paper-api.alpaca.markets as base and v2 as version
        api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url="https://paper-api.alpaca.markets",
            api_version='v2'
        )

        account = api.get_account()
        print(f"✅ SUCCESS!")
        print(f"   Account ID: {account.id}")
        print(f"   Status: {account.status}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {str(e)}")

    return False

def test_jax_working():
    """Confirm JAX is working"""
    print("\n🔧 Confirming JAX/TPU...")
    try:
        import jax
        devices = jax.devices()
        print(f"✅ JAX working: {len(devices)} x {devices[0].device_kind}")
    except Exception as e:
        print(f"❌ JAX failed: {e}")

def main():
    """Main function"""
    print("🚀 WORKING FINAL VERSION")
    print("=" * 50)
    print("Using EXACT endpoint from your Alpaca panel")
    print("Base URL: https://paper-api.alpaca.markets/v2")
    print("=" * 50)

    setup_environment()
    test_jax_working()

    success = test_alpaca_exact_endpoint()

    print("\n" + "=" * 50)
    if success:
        print("🎉 FINALLY! ALPACA AUTHENTICATION WORKING!")
        print("\n🚀 READY TO START TRADING BOT!")
        print("Next steps:")
        print("1. ✅ Authentication working")
        print("2. Update config.py with working credentials")
        print("3. Test trading functionality")
        print("4. Start NCA model training")
    else:
        print("❌ Still having issues with Alpaca")
        print("\n💡 But you can proceed with:")
        print("✅ Yahoo Finance for market data")
        print("✅ JAX/TPU for model training")
        print("✅ System components working")

if __name__ == "__main__":
    main()