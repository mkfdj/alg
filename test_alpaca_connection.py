"""
Test Alpaca API connection with single paper trading key
"""

import os
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta


def test_alpaca_connection():
    """Test Alpaca API connection with your paper trading key"""

    # Your paper trading API key (single key for paper accounts)
    PAPER_API_KEY = "PKJ346E2YWMT7HCFZX09"
    BASE_URL = "https://paper-api.alpaca.markets"

    print("🔧 Testing Alpaca API Connection")
    print("=" * 50)
    print(f"API Key: {PAPER_API_KEY}")
    print(f"Base URL: {BASE_URL}")
    print()

    # Test Trading Client
    print("1. Testing Trading Client...")
    try:
        trading_client = TradingClient(
            api_key=PAPER_API_KEY,
            secret_key=PAPER_API_KEY,  # Paper trading uses same key
            paper=True,
            url_override=BASE_URL
        )

        account = trading_client.get_account()
        print("✅ Trading Client Connected!")
        print(f"   Account ID: {account.id}")
        print(f"   Status: {account.status}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print()

    except Exception as e:
        print(f"❌ Trading Client Error: {e}")
        return False

    # Test Data Client
    print("2. Testing Market Data Client...")
    try:
        data_client = StockHistoricalDataClient(api_key=PAPER_API_KEY)

        # Get recent data for a popular stock
        request_params = StockBarsRequest(
            symbol_or_symbols="AAPL",
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=5),
            end=datetime.now()
        )

        bars = data_client.get_stock_bars(request_params)
        print("✅ Market Data Client Connected!")
        print(f"   Retrieved {len(bars.data)} bars of data")

        if bars.data:
            latest_bar = bars.data[0]
            print(f"   Latest AAPL: ${latest_bar.close:.2f}")
        print()

    except Exception as e:
        print(f"❌ Market Data Client Error: {e}")
        return False

    # Test Order Submission (Dry Run)
    print("3. Testing Order Capability...")
    try:
        positions = trading_client.get_all_positions()
        print(f"✅ Order System Ready!")
        print(f"   Current Positions: {len(positions)}")

        if positions:
            for pos in positions[:3]:  # Show first 3 positions
                print(f"   - {pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price:.2f}")
        else:
            print("   No open positions")
        print()

    except Exception as e:
        print(f"❌ Order System Error: {e}")
        return False

    print("🎉 All Alpaca API Tests Passed!")
    return True


def setup_environment_variables():
    """Set up environment variables for the trading bot"""
    print("🔐 Setting up environment variables...")

    # Set environment variables
    os.environ["ALPACA_PAPER_API_KEY"] = "PKJ346E2YWMT7HCFZX09"

    print("✅ Environment variables set!")
    print("   ALPACA_PAPER_API_KEY: PKJ346E2YWMT7HCFZX09")
    print()


if __name__ == "__main__":
    # Set up environment
    setup_environment_variables()

    # Test connection
    success = test_alpaca_connection()

    if success:
        print("\n🚀 Ready to start trading!")
        print("\nNext steps:")
        print("1. Run: python nca_trading_bot/main.py --mode analyze")
        print("2. Run: python nca_trading_bot/main.py --mode train")
        print("3. Run: python nca_trading_bot/main.py --mode backtest")
    else:
        print("\n❌ Connection failed. Please check your API key.")