#!/usr/bin/env python3
"""
KAGGLE ALPACA API TEST - SIMPLE EXAMPLE
=====================================
Copy this code into your Kaggle notebook and run it.
Replace the API keys with your actual Trading API keys from Alpaca.
"""

# Replace these with your actual Trading API keys from https://app.alpaca.markets/
# Make sure you're using Trading API keys, NOT App API keys!
YOUR_API_KEY = "PK8OP9HMO3QICJMAFMKG"  # Replace with your actual key
YOUR_SECRET_KEY = "Pge5ic8eDN0ze0YTpEJxNmpdf3YGUnhOVnWJZbf7"  # Replace with your actual secret

# Method 1: Quick test (copy both files to Kaggle)
if True:  # Set to True to run this method
    print("🚀 Starting Alpaca API Test in Kaggle...")
    
    # Import and run the unified tester
    exec(open('alpaca_v2_unified_test.py').read())
    
    # Create tester instance
    from alpaca_v2_unified_test import AlpacaV2Tester
    tester = AlpacaV2Tester(api_key=YOUR_API_KEY, secret_key=YOUR_SECRET_KEY)
    
    # Run comprehensive test
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n🎉 SUCCESS! Your Alpaca API is working in Kaggle!")
        print("✅ You can now integrate these credentials into your trading bot")
    else:
        print("\n❌ API test failed. Check the diagnosis above for solutions.")

# Method 2: Using environment variables (recommended for production)
if False:  # Set to True to use this method instead
    import os
    
    # Set environment variables (or use Kaggle secrets)
    os.environ['ALPACA_API_KEY'] = YOUR_API_KEY
    os.environ['ALPACA_SECRET_KEY'] = YOUR_SECRET_KEY
    
    # Import and run
    exec(open('alpaca_v2_unified_test.py').read())
    
    from alpaca_v2_unified_test import AlpacaV2Tester
    tester = AlpacaV2Tester()  # Will use environment variables
    tester.run_comprehensive_test()

print("\n📝 Next Steps if Test Passes:")
print("1. Integrate working credentials into your NCA trading bot")
print("2. Update nca_trading_bot/config.py with these keys")
print("3. Start testing your trading strategies!")