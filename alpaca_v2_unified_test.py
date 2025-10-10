#!/usr/bin/env python3
"""
UNIFIED ALPACA V2 API TEST FOR KAGGLE
====================================
Clean, working solution for Alpaca v2 API authentication and testing.
Works in Kaggle notebooks and local environments.
"""

import os
import sys
import subprocess
import requests
import json
from datetime import datetime
from pathlib import Path

# Environment detection
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
IS_COLAB = 'COLAB_GPU' in os.environ

class AlpacaV2Tester:
    """Unified Alpaca v2 API tester with comprehensive diagnostics"""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        # Use environment variables or provided keys
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        # Alpaca v2 endpoints
        self.paper_base_url = "https://paper-api.alpaca.markets/v2"
        self.live_base_url = "https://api.alpaca.markets/v2"
        
    def setup_environment(self):
        """Install required packages"""
        print("🔧 Setting up environment...")
        
        packages = [
            "alpaca-py>=0.30.0",  # Modern Alpaca Python SDK
            "requests>=2.28.0",
            "websocket-client>=1.6.0"
        ]
        
        for package in packages:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    package, "--quiet", "--upgrade"
                ])
                print(f"✅ {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                
    def validate_credentials(self):
        """Comprehensive credential validation and troubleshooting"""
        print("\n🔍 Validating credentials...")
        
        if not self.api_key or not self.secret_key:
            print("❌ Missing API credentials!")
            print("   Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
            print("   Or pass them to AlpacaV2Tester(api_key='...', secret_key='...')")
            return False
            
        print(f"✅ API Key: {self.api_key[:8]}...{self.api_key[-4:]}")
        print(f"✅ Secret Key: {self.secret_key[:8]}...{self.secret_key[-8:]}")
        
        # Validate format
        issues = []
        if len(self.api_key) != 20:
            issues.append(f"API Key length: {len(self.api_key)} (should be 20)")
        if len(self.secret_key) != 40:
            issues.append(f"Secret Key length: {len(self.secret_key)} (should be 40)")
            
        # Check for common issues
        if not self.api_key.startswith('PK'):
            issues.append("API Key should start with 'PK'")
            
        if any(char in self.api_key for char in [' ', '\n', '\t']):
            issues.append("API Key contains whitespace")
            
        if any(char in self.secret_key for char in [' ', '\n', '\t']):
            issues.append("Secret Key contains whitespace")
            
        if issues:
            print("⚠️  Credential Issues Detected:")
            for issue in issues:
                print(f"   - {issue}")
            print("\n🔧 Common Solutions:")
            print("   1. Copy keys directly from Alpaca dashboard")
            print("   2. Check for extra spaces or line breaks")
            print("   3. Ensure you're using TRADING API keys (not App API)")
            print("   4. Generate new keys if needed")
        else:
            print("✅ Credential format looks correct")
            
        return True
        
    def diagnose_auth_failure(self):
        """Provide detailed troubleshooting for authentication failures"""
        print("\n🩺 AUTHENTICATION FAILURE DIAGNOSIS")
        print("=" * 50)
        
        print("📋 Most Common Causes of 401 Unauthorized:")
        print("1. Using App API keys instead of Trading API keys")
        print("   → Solution: Go to https://app.alpaca.markets/ → API Keys → Trading API tab")
        print("   → Generate new keys under 'Trading API' section")
        
        print("\n2. Paper account was reset, invalidating keys")
        print("   → Solution: Generate new keys after account reset")
        
        print("\n3. Account not approved for trading")
        print("   → Solution: Check account status in dashboard")
        
        print("\n4. Using live keys for paper trading (or vice versa)")
        print("   → Solution: Ensure keys match the endpoint being used")
        
        print("\n5. Keys copied incorrectly with extra characters")
        print("   → Solution: Copy keys carefully, check for spaces/newlines")
        
        print("\n6. Rate limiting or temporary server issues")
        print("   → Solution: Wait a few minutes and try again")
        
        print("\n🔍 Key Verification Steps:")
        print("1. Log into https://app.alpaca.markets/")
        print("2. Navigate to 'API Keys' section")
        print("3. Verify you're on 'Trading API' tab (NOT 'App API')")
        print("4. Check if your current keys are listed and active")
        print("5. If not listed, generate new Trading API keys")
        print("6. Copy the EXACT keys (no extra spaces)")
        
        print(f"\n🔧 Your Current Keys Analysis:")
        print(f"   API Key Format: {'✅ Valid' if len(self.api_key) == 20 and self.api_key.startswith('PK') else '❌ Invalid'}")
        print(f"   Secret Key Format: {'✅ Valid' if len(self.secret_key) == 40 else '❌ Invalid'}")
        
        # Test key character validation
        api_clean = ''.join(c for c in self.api_key if c.isalnum())
        secret_clean = ''.join(c for c in self.secret_key if c.isalnum())
        
        if len(api_clean) != len(self.api_key):
            print(f"   ⚠️  API Key contains non-alphanumeric characters")
        if len(secret_clean) != len(self.secret_key):
            print(f"   ⚠️  Secret Key contains non-alphanumeric characters")
            
        # Test for App API vs Trading API
        self._test_app_api_endpoint()
        
    def _test_app_api_endpoint(self):
        """Test if these are App API keys by trying App API endpoint"""
        print(f"\n🔍 Testing if these are App API keys instead of Trading API...")
        
        # Try the Apps API endpoint which uses different authentication
        try:
            headers = {
                'APCA-API-KEY-ID': self.api_key,
                'APCA-API-SECRET-KEY': self.secret_key,
                'Content-Type': 'application/json'
            }
            
            # App API uses different endpoints
            app_urls = [
                "https://broker-api.alpaca.markets/v1/accounts",
                "https://broker-api.alpaca.markets/v1/journals",
            ]
            
            for url in app_urls:
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        print(f"   🎯 FOUND THE ISSUE! These are App API keys, not Trading API keys!")
                        print(f"   ✅ App API endpoint works: {url}")
                        print(f"\n   🔧 SOLUTION:")
                        print(f"   1. Go to https://app.alpaca.markets/")
                        print(f"   2. Click on 'API Keys' section")
                        print(f"   3. Switch to 'Trading API' tab (you're currently using App API)")
                        print(f"   4. Generate new Trading API keys")
                        print(f"   5. Replace your current keys with the Trading API keys")
                        return True
                    elif response.status_code != 404:  # 404 is expected if not app keys
                        print(f"   App API test: {response.status_code} for {url}")
                except:
                    continue
                    
            print(f"   ❌ App API endpoints also failed - keys may be invalid")
            
        except Exception as e:
            print(f"   ❌ App API test failed: {e}")
            
        return False
        
    def test_direct_http(self, use_paper=True):
        """Test direct HTTP requests to Alpaca v2 API with multiple authentication methods"""
        base_url = self.paper_base_url if use_paper else self.live_base_url
        env_type = "Paper" if use_paper else "Live"
        
        print(f"\n📡 Testing {env_type} API via Direct HTTP...")
        print(f"   Endpoint: {base_url}/account")
        
        # Method 1: Standard headers (most common)
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json',
            'User-Agent': 'AlpacaV2-Kaggle-Test/1.0'
        }
        
        print(f"   Method 1: Standard headers...")
        success = self._test_http_request(f"{base_url}/account", headers, env_type)
        if success:
            return True
            
        # Method 2: Try with Basic Auth (some users report this works)
        print(f"   Method 2: Basic Authentication...")
        import base64
        auth_string = f"{self.api_key}:{self.secret_key}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        
        headers_basic = {
            'Authorization': f'Basic {auth_b64}',
            'Content-Type': 'application/json',
            'User-Agent': 'AlpacaV2-Kaggle-Test/1.0'
        }
        
        success = self._test_http_request(f"{base_url}/account", headers_basic, env_type)
        if success:
            return True
            
        # Method 3: Try without Content-Type header (sometimes helps)
        print(f"   Method 3: Minimal headers...")
        headers_minimal = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        }
        
        success = self._test_http_request(f"{base_url}/account", headers_minimal, env_type)
        if success:
            return True
            
        # Method 4: Try older v1 endpoint (fallback)
        if use_paper:
            print(f"   Method 4: Trying v1 endpoint...")
            v1_url = "https://paper-api.alpaca.markets/v1/account"
            success = self._test_http_request(v1_url, headers, env_type)
            if success:
                return True
            
        return False
        
    def _test_http_request(self, url, headers, env_type):
        """Helper method to test HTTP request"""
        try:
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                account_data = response.json()
                print(f"   ✅ {env_type} API SUCCESS!")
                self._print_account_info(account_data)
                return True
            elif response.status_code == 401:
                print(f"   ❌ 401 Unauthorized")
                try:
                    error_data = response.json()
                    print(f"     Error: {error_data.get('message', 'Unauthorized')}")
                except:
                    print(f"     Response: {response.text[:100]}")
            else:
                print(f"   ❌ HTTP {response.status_code}")
                print(f"     Response: {response.text[:100]}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Network error: {e}")
            
        return False
        
    def test_alpaca_py_sdk(self, use_paper=True):
        """Test using modern alpaca-py SDK"""
        env_type = "Paper" if use_paper else "Live"
        
        print(f"\n🔄 Testing {env_type} API via alpaca-py SDK...")
        
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            
            # Create trading client
            trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=use_paper
            )
            
            # Test account
            account = trading_client.get_account()
            print(f"✅ {env_type} Trading Client SUCCESS!")
            self._print_account_info(account.__dict__)
            
            # Test market data
            data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            # Test clock
            clock = trading_client.get_clock()
            print(f"   Market Status: {'Open' if clock.is_open else 'Closed'}")
            print(f"   Next Open: {clock.next_open}")
            
            return True
            
        except ImportError:
            print("❌ alpaca-py not available, trying alpaca-trade-api...")
            return self._test_legacy_api(use_paper)
        except Exception as e:
            print(f"❌ alpaca-py failed: {type(e).__name__}: {e}")
            return self._test_legacy_api(use_paper)
            
    def _test_legacy_api(self, use_paper=True):
        """Fallback to legacy alpaca-trade-api"""
        env_type = "Paper" if use_paper else "Live"
        
        try:
            import alpaca_trade_api as tradeapi
            
            base_url = "https://paper-api.alpaca.markets" if use_paper else "https://api.alpaca.markets"
            
            api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=base_url,
                api_version='v2'
            )
            
            account = api.get_account()
            print(f"✅ {env_type} Legacy API SUCCESS!")
            print(f"   Account ID: {account.id}")
            print(f"   Status: {account.status}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Legacy API failed: {type(e).__name__}: {e}")
            return False
            
    def _print_account_info(self, account_data):
        """Print formatted account information"""
        if isinstance(account_data, dict):
            account_id = account_data.get('id')
            status = account_data.get('status')
            buying_power = account_data.get('buying_power', 0)
            portfolio_value = account_data.get('portfolio_value', 0)
        else:
            # Handle object attributes
            account_id = getattr(account_data, 'id', 'N/A')
            status = getattr(account_data, 'status', 'N/A')
            buying_power = getattr(account_data, 'buying_power', 0)
            portfolio_value = getattr(account_data, 'portfolio_value', 0)
            
        print(f"   Account ID: {account_id}")
        print(f"   Status: {status}")
        print(f"   Buying Power: ${float(buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(portfolio_value):,.2f}")
        
    def test_market_data(self):
        """Test market data access"""
        print("\n📊 Testing Market Data Access...")
        
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from datetime import datetime, timedelta
            
            data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            # Test with popular stock
            request_params = StockBarsRequest(
                symbol_or_symbols=["AAPL"],
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=5)
            )
            
            bars = data_client.get_stock_bars(request_params)
            print("✅ Market data access working!")
            print(f"   Retrieved {len(bars.data)} bars for AAPL")
            
            if bars.data:
                latest = list(bars.data.values())[0][-1]
                print(f"   Latest close: ${latest.close:.2f}")
                
            return True
            
        except Exception as e:
            print(f"❌ Market data failed: {type(e).__name__}: {e}")
            return False
            
    def run_comprehensive_test(self):
        """Run all tests"""
        print("🚀 UNIFIED ALPACA V2 API TEST")
        print("=" * 50)
        print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Colab' if IS_COLAB else 'Local'}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 50)
        
        # Setup
        self.setup_environment()
        
        if not self.validate_credentials():
            return False
            
        # Test sequence
        results = {
            'paper_http': False,
            'paper_sdk': False,
            'live_http': False,
            'market_data': False
        }
        
        # Test paper trading (most common)
        results['paper_http'] = self.test_direct_http(use_paper=True)
        results['paper_sdk'] = self.test_alpaca_py_sdk(use_paper=True)
        
        # Test live (if paper fails)
        if not results['paper_http']:
            print("\n⚠️  Paper trading failed, testing live API...")
            results['live_http'] = self.test_direct_http(use_paper=False)
            
        # Test market data
        if any(results.values()):
            results['market_data'] = self.test_market_data()
            
        # Summary
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        
        success_count = sum(results.values())
        total_tests = len(results)
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
            
        print(f"\nOverall: {success_count}/{total_tests} tests passed")
        
        if success_count > 0:
            print("\n🎉 SUCCESS! Alpaca API is working!")
            print("\n📝 Next Steps:")
            print("1. Update your trading bot config with these credentials")
            print("2. Test order placement (if needed)")
            print("3. Start model training with live data")
            
            if IS_KAGGLE:
                print("\n🏆 KAGGLE INTEGRATION READY!")
                print("Copy these credentials to your Kaggle secrets:")
                print("- Add 'ALPACA_API_KEY' secret")
                print("- Add 'ALPACA_SECRET_KEY' secret")
        else:
            print("\n❌ All tests failed!")
            self.diagnose_auth_failure()
            
        return success_count > 0


def main():
    """Main function for direct script execution"""
    # Example usage with hardcoded keys (replace with your own)
    api_key = "PK8OP9HMO3QICJMAFMKG"  # Replace with your key
    secret_key = "Pge5ic8eDN0ze0YTpEJxNmpdf3YGUnhOVnWJZbf7"  # Replace with your secret
    
    # Or use environment variables (recommended)
    # tester = AlpacaV2Tester()
    
    tester = AlpacaV2Tester(api_key=api_key, secret_key=secret_key)
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()