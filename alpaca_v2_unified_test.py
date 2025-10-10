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
        """Validate credential format"""
        print("\n🔍 Validating credentials...")
        
        if not self.api_key or not self.secret_key:
            print("❌ Missing API credentials!")
            print("   Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
            print("   Or pass them to AlpacaV2Tester(api_key='...', secret_key='...')")
            return False
            
        print(f"✅ API Key: {self.api_key[:8]}...{self.api_key[-4:]}")
        print(f"✅ Secret Key: {self.secret_key[:8]}...{self.secret_key[-8:]}")
        
        # Validate format
        if len(self.api_key) != 20:
            print("⚠️  API Key should be 20 characters long")
        if len(self.secret_key) != 40:
            print("⚠️  Secret Key should be 40 characters long")
            
        return True
        
    def test_direct_http(self, use_paper=True):
        """Test direct HTTP requests to Alpaca v2 API"""
        base_url = self.paper_base_url if use_paper else self.live_base_url
        env_type = "Paper" if use_paper else "Live"
        
        print(f"\n📡 Testing {env_type} API via Direct HTTP...")
        print(f"   Endpoint: {base_url}/account")
        
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json',
            'User-Agent': 'AlpacaV2-Kaggle-Test/1.0'
        }
        
        try:
            response = requests.get(
                f"{base_url}/account",
                headers=headers,
                timeout=30
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                account_data = response.json()
                print(f"✅ {env_type} API SUCCESS!")
                self._print_account_info(account_data)
                return True
            elif response.status_code == 401:
                print(f"❌ Authentication failed")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('message', 'Unauthorized')}")
                except:
                    print(f"   Response: {response.text}")
            else:
                print(f"❌ HTTP {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            
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
            print("\n🔧 Troubleshooting:")
            print("1. Verify credentials at https://app.alpaca.markets/")
            print("2. Ensure you're using 'Trading API' keys (not 'App API')")
            print("3. Check if your account needs approval")
            print("4. Try generating new API keys")
            
        return success_count > 0


def main():
    """Main function for direct script execution"""
    # Example usage with hardcoded keys (replace with your own)
    api_key = "PKJ346E2YWMT7HCFZX09"  # Replace with your key
    secret_key = "Pge5ic8eDN0ze0YTpEJxNmpdf3YGUnhOVnWJZbf7"  # Replace with your secret
    
    # Or use environment variables (recommended)
    # tester = AlpacaV2Tester()
    
    tester = AlpacaV2Tester(api_key=api_key, secret_key=secret_key)
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()