#!/usr/bin/env python3
"""
Test script for new data handling features.
"""

import asyncio
import logging
from data_handler import data_handler

logging.basicConfig(level=logging.INFO)

async def test_sp500():
    """Test S&P500 data loading."""
    try:
        print("Testing S&P500 data loading...")
        df = await data_handler.get_sp500_data()
        print(f"✅ S&P500 data loaded successfully: {len(df)} records")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"❌ S&P500 data loading failed: {e}")
        return False

def test_kaggle_nasdaq():
    """Test Kaggle NASDAQ data loading."""
    try:
        print("Testing Kaggle NASDAQ data loading...")
        # Note: This will fail without Kaggle credentials
        df = data_handler.get_kaggle_nasdaq_data()
        print(f"✅ NASDAQ data loaded successfully: {len(df)} records")
        return True
    except Exception as e:
        print(f"❌ NASDAQ data loading failed: {e}")
        return False

def test_global_financial():
    """Test global financial data loading."""
    try:
        print("Testing global financial data loading...")
        # Note: This will fail without Kaggle credentials
        df = data_handler.get_global_financial_data()
        print(f"✅ Global financial data loaded successfully: {len(df)} records")
        return True
    except Exception as e:
        print(f"❌ Global financial data loading failed: {e}")
        return False

async def test_comprehensive():
    """Test comprehensive data loading."""
    try:
        print("Testing comprehensive data loading...")
        data = await data_handler.get_comprehensive_market_data()
        print(f"✅ Comprehensive data loaded: {len(data)} sources")
        for source, df in data.items():
            print(f"   {source}: {len(df)} records")
        return True
    except Exception as e:
        print(f"❌ Comprehensive data loading failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 Testing new data handling features")
    print("=" * 50)

    results = []
    results.append(await test_sp500())
    results.append(test_kaggle_nasdaq())
    results.append(test_global_financial())
    results.append(await test_comprehensive())

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check logs above.")

if __name__ == "__main__":
    asyncio.run(main())