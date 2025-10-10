#!/usr/bin/env python3
"""
Final Alpaca Fix Script - Addressing Common Issues
Based on StackOverflow and Alpaca Forum solutions
"""

import os
import sys
import subprocess
import requests
import base64
from pathlib import Path

def setup_environment():
    """Setup environment"""
    print("🔧 Installing packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "alpaca-trade-api>=3.1.0", "--quiet"])

def test_basic_auth_method():
    """Test HTTP Basic Auth method (StackOverflow solution)"""
    print("\n🔧 Testing HTTP Basic Auth Method...")

    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "Pge5ic8eDN0ze0YTpEJxNmpdf3YGUnhOVnWJZbf7"

    print(f"🔑 API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"🔐 Secret Key: {secret_key[:8]}...{secret_key[-8:]}")

    # Create Basic Auth header (StackOverflow solution)
    credentials = f"{api_key}:{secret_key}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {
        'Authorization': f'Basic {encoded_credentials}',
        'Content-Type': 'application/json'
    }

    endpoints = [
        "https://paper-api.alpaca.markets/v2/account",
        "https://paper-api.alpaca.markets/account"
    ]

    for endpoint in endpoints:
        print(f"🔄 Testing {endpoint}...")
        try:
            response = requests.get(endpoint, headers=headers, timeout=30)
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                account_data = response.json()
                print(f"✅ SUCCESS with Basic Auth!")
                print(f"   Account ID: {account_data.get('id')}")
                print(f"   Status: {account_data.get('status')}")
                print(f"   Buying Power: ${float(account_data.get('buying_power', 0)):,.2f}")
                return True
            else:
                print(f"   ❌ Response: {response.text[:100]}...")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    return False

def test_alpaca_trade_api_methods():
    """Test multiple alpaca-trade-api configurations"""
    print("\n🔧 Testing Alpaca Trade API Methods...")

    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "Pge5ic8eDN0ze0YTpEJxNmpdf3YGUnhOVnWJZbf7"

    try:
        import alpaca_trade_api as tradeapi

        # Method 1: Original from Alpaca docs
        print("🔄 Method 1: Original docs configuration...")
        try:
            api = tradeapi.REST(
                key_id=api_key,
                secret_key=secret_key,
                base_url="https://paper-api.alpaca.markets",
                api_version='v2'
            )
            account = api.get_account()
            print(f"✅ Method 1 SUCCESS!")
            print(f"   Account ID: {account.id}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            return True
        except Exception as e:
            print(f"   ❌ Method 1 failed: {e}")

        # Method 2: Full v2 URL as base
        print("🔄 Method 2: Full v2 URL as base...")
        try:
            api = tradeapi.REST(
                key_id=api_key,
                secret_key=secret_key,
                base_url="https://paper-api.alpaca.markets/v2"
            )
            account = api.get_account()
            print(f"✅ Method 2 SUCCESS!")
            print(f"   Account ID: {account.id}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            return True
        except Exception as e:
            print(f"   ❌ Method 2 failed: {e}")

        # Method 3: Without API version
        print("🔄 Method 3: Without API version...")
        try:
            api = tradeapi.REST(
                key_id=api_key,
                secret_key=secret_key,
                base_url="https://paper-api.alpaca.markets/v2"
            )
            account = api.get_account()
            print(f"✅ Method 3 SUCCESS!")
            print(f"   Account ID: {account.id}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            return True
        except Exception as e:
            print(f"   ❌ Method 3 failed: {e}")

    except ImportError as e:
        print(f"❌ Import failed: {e}")

    return False

def diagnose_key_issues():
    """Diagnose potential key issues"""
    print("\n🔧 Diagnosing Key Issues...")

    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "Pge5ic8eDN0ze0YTpEJxNmpdf3YGUnhOVnWJZbf7"

    print("📋 Key Analysis:")
    print(f"   API Key: {api_key}")
    print(f"   Format: {'Valid' if api_key.startswith('PK') and len(api_key) == 20 else 'Invalid'}")
    print(f"   Secret Key Length: {len(secret_key)}")

    print("\n🔍 Common Issues & Solutions:")
    print("   1. Using App Credentials instead of Trade Credentials")
    print("      → Go to: https://app.alpaca.markets/ → API Keys")
    print("      → Make sure you're on the 'Trading API' tab, not 'App API'")
    print("      → Generate new keys under Trading API section")

    print("   2. Paper account was reset")
    print("      → Resetting paper account invalidates old keys")
    print("      → Generate new keys after reset")

    print("   3. Account not approved")
    print("      → Paper accounts may need approval")
    print("      → Check dashboard for account status")

    print("   4. Using live keys for paper trading")
    print("      → Ensure keys are from paper account section")
    print("      → Live keys won't work on paper endpoint")

def test_alternative_endpoints():
    """Test alternative endpoints that might work"""
    print("\n🔧 Testing Alternative Endpoints...")

    api_key = "PKJ346E2YWMT7HCFZX09"
    secret_key = "Pge5ic8eDN0ze0YTpEJxNmpdf3YGUnhOVnWJZbf7"

    # Different endpoint variations
    test_endpoints = [
        ("https://paper-api.alpaca.markets/v2/account", "v2 direct"),
        ("https://paper-api.alpaca.markets/account", "no version"),
        ("https://api.alpaca.markets/v2/account", "live endpoint test"),
    ]

    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key,
        'Content-Type': 'application/json'
    }

    for endpoint, description in test_endpoints:
        print(f"🔄 Testing {description}: {endpoint}")
        try:
            response = requests.get(endpoint, headers=headers, timeout=30)
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                print(f"   ✅ SUCCESS with {description}!")
                return True
            elif response.status_code == 401:
                print(f"   ❌ 401 Unauthorized")
            else:
                print(f"   ❌ {response.status_code}: {response.text[:50]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return False

def main():
    """Main function"""
    print("🚀 FINAL ALPACA FIX SCRIPT")
    print("=" * 60)
    print("Based on StackOverflow & Alpaca Forum solutions")
    print("=" * 60)

    setup_environment()

    # Run all diagnostic tests
    diagnose_key_issues()

    success = False
    success |= test_basic_auth_method()
    success |= test_alpaca_trade_api_methods()
    success |= test_alternative_endpoints()

    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS! Alpaca authentication working!")
        print("✅ Ready to proceed with trading bot development")
    else:
        print("❌ All authentication methods failed")
        print("\n🔧 FINAL RECOMMENDATIONS:")
        print("1. Go to: https://app.alpaca.markets/")
        print("2. Navigate to API Keys section")
        print("3. Make sure you're on 'Trading API' tab (not 'App API')")
        print("4. Generate new Trading API keys")
        print("5. Replace the keys in the script")
        print("6. Try again")

        print("\n📞 Support:")
        print("- Alpaca Support: https://alpaca.markets/support")
        print("- Community Forum: https://forum.alpaca.markets/")

if __name__ == "__main__":
    main()