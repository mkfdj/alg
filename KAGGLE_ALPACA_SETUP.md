# Kaggle Alpaca v2 API Setup Guide

## Quick Start

1. **Upload the unified test script to Kaggle:**
   - Copy `alpaca_v2_unified_test.py` to your Kaggle notebook
   - Or upload it as a dataset

2. **Add your API credentials as Kaggle secrets:**
   - Go to your Kaggle account settings
   - Add secrets:
     - `ALPACA_API_KEY` = your API key
     - `ALPACA_SECRET_KEY` = your secret key

3. **Super Simple Method - Upload `kaggle_test_example.py`:**
   - Just replace the API keys in the example file and run it!
   - Everything is pre-configured for you

4. **Or run manually in Kaggle:**
   ```python
   # Replace with your actual keys
   exec(open('alpaca_v2_unified_test.py').read())
   from alpaca_v2_unified_test import AlpacaV2Tester
   
   tester = AlpacaV2Tester(
       api_key="YOUR_ACTUAL_API_KEY", 
       secret_key="YOUR_ACTUAL_SECRET_KEY"
   )
   tester.run_comprehensive_test()
   ```

## Integration with NCA Trading Bot

Once the test passes, update your `nca_trading_bot/config.py`:

```python
# In your Kaggle notebook
import os
from nca_trading_bot.config import TradingConfig

# Set environment variables
os.environ['ALPACA_API_KEY'] = 'your_working_key'
os.environ['ALPACA_SECRET_KEY'] = 'your_working_secret'

# Create config
config = TradingConfig()
config.alpaca.api_key = os.getenv('ALPACA_API_KEY')
config.alpaca.secret_key = os.getenv('ALPACA_SECRET_KEY')
config.alpaca.paper_trading = True  # Always use paper in Kaggle
```

## What the Test Does

✅ **Environment Detection** - Automatically detects Kaggle
✅ **Package Installation** - Installs latest alpaca-py SDK
✅ **Credential Validation** - Checks API key format
✅ **Multiple API Tests** - Tests both modern and legacy SDKs
✅ **Market Data Test** - Verifies data access
✅ **Comprehensive Reporting** - Clear success/failure messages

## Expected Output

If working correctly, you'll see:
```
🚀 UNIFIED ALPACA V2 API TEST
==================================================
Environment: Kaggle
✅ alpaca-py>=0.30.0
✅ API Key: PKJ346E2...ZX09
✅ Paper API SUCCESS!
   Account ID: your-account-id
   Status: ACTIVE
   Buying Power: $100,000.00
🎉 SUCCESS! Alpaca API is working!
```

## Troubleshooting

If tests fail:
1. **Go to https://app.alpaca.markets/**
2. **Navigate to API Keys → Trading API tab**
3. **Generate new Trading API keys** (not App API)
4. **Replace credentials in the script**
5. **Re-run the test**

## Next Steps After Success

1. Update your trading bot configuration
2. Test order placement (optional)
3. Start NCA model training with live market data
4. Set up automated trading workflows