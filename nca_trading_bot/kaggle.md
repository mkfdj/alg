# Kaggle Integration for NCA Trading Bot

This document describes how to integrate the NCA Trading Bot with Kaggle datasets and competitions.

## Setting up Kaggle API

1. Install the Kaggle package:
```bash
pip install kaggle
```

2. Create a Kaggle API token:
   - Log in to your Kaggle account
   - Go to Account > API > Create New API Token
   - Download the kaggle.json file

3. Set up API credentials:
```bash
# Option 1: Place kaggle.json in the correct location
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Option 2: Set environment variables
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

## Available Kaggle Datasets

### Financial Datasets

1. **Huge Stock Market Dataset**
   - Dataset: `sdogan08/huge-stock-market-dataset`
   - Description: Historical stock data for thousands of companies
   - Usage: `kaggle datasets download -d sdogan08/huge-stock-market-dataset`

2. **NASDAQ All Stocks**
   - Dataset: `jacksoncrow/stock-market-dataset`
   - Description: NASDAQ stock data from 2010 to 2020
   - Usage: `kaggle datasets download -d jacksoncrow/stock-market-dataset`

3. **S&P 500 Stock Data**
   - Dataset: `camnugent/sandp500`
   - Description: S&P 500 stock data with daily prices
   - Usage: `kaggle datasets download -d camnugent/sandp500`

4. **World Stock Prices**
   - Dataset: `andrewmvd/sp-500-stocks`
   - Description: World stock prices with company information
   - Usage: `kaggle datasets download -d andrewmvd/sp-500-stocks`

5. **Stock Exchange Data**
   - Dataset: `dsaxton/nyse-prices-daily`
   - Description: NYSE daily prices
   - Usage: `kaggle datasets download -d dsaxton/nyse-prices-daily`

## Using Kaggle Datasets with the NCA Trading Bot

### Configuration

Update your configuration to use Kaggle datasets:

```python
from config import ConfigManager

config = ConfigManager()

# Enable Kaggle datasets
config.data.use_kaggle_nasdaq = True
config.data.use_kaggle_huge_stock = True
config.data.use_kaggle_sp500 = True
config.data.use_kaggle_world_stocks = True
config.data.use_kaggle_exchanges = True

# Set Kaggle API credentials
config.data.kaggle_username = "your_username"
config.data.kaggle_key = "your_api_key"

# Set backtest end year to filter data
config.data.backtest_end_year = 2021
```

### Data Loading

The DataHandler class will automatically download and process Kaggle datasets when enabled:

```python
from data_handler import DataHandler

data_handler = DataHandler()

# Load data from multiple Kaggle datasets
data = await data_handler.get_multiple_tickers_data(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    start_date="2020-01-01",
    end_date="2021-12-31"
)
```

## Kaggle Integration Examples

### Example 1: Training with Kaggle Data

```python
import asyncio
from main import NCATradingBot

async def train_with_kaggle_data():
    # Create bot
    bot = NCATradingBot()
    
    # Enable Kaggle datasets
    bot.config.data.use_kaggle_nasdaq = True
    bot.config.data.use_kaggle_huge_stock = True
    
    # Train model
    await bot.train_offline(num_epochs=100)
    
    # Save model
    bot.save_model()

# Run training
asyncio.run(train_with_kaggle_data())
```

### Example 2: Backtesting with Kaggle Data

```python
import asyncio
from main import NCATradingBot

async def backtest_with_kaggle_data():
    # Create bot
    bot = NCATradingBot()
    
    # Load model
    bot.create_model(load_path="path/to/model.pt")
    bot.setup_trading_agent()
    
    # Run backtest
    results = await bot.run_backtest(
        start_date="2020-01-01",
        end_date="2021-12-31",
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    )
    
    # Print results
    print("Backtest Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")

# Run backtest
asyncio.run(backtest_with_kaggle_data())
```

## Kaggle Competitions

The NCA Trading Bot can be adapted for participation in Kaggle trading competitions:

1. **Optiver Realized Volatility Prediction**
   - Competition: `optiver/realized-volatility-prediction`
   - Description: Predict realized volatility for stocks
   - Adaptation: Modify the model to predict volatility instead of price

2. **Jane Street Market Prediction**
   - Competition: `jane-street-market-prediction`
   - Description: Predict trading actions in a market environment
   - Adaptation: Use the NCA model for reinforcement learning

3. **G-Research Crypto Forecasting**
   - Competition: `g-research-crypto-forecasting`
   - Description: Predict returns for cryptocurrencies
   - Adaptation: Apply the model to cryptocurrency data

## Performance Considerations

1. **Data Size**: Kaggle datasets can be large. Use the `max_ram_gb` configuration to limit memory usage.

2. **Download Time**: Initial download of Kaggle datasets may take time. Consider caching the data.

3. **API Limits**: Kaggle API has rate limits. Handle API errors gracefully.

4. **Data Quality**: Kaggle datasets may have missing values or errors. Implement robust data cleaning.

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure your Kaggle API credentials are correct
   - Check that kaggle.json is in the correct location
   - Verify environment variables are set correctly

2. **Download Failures**
   - Check your internet connection
   - Verify the dataset name is correct
   - Handle API rate limits

3. **Memory Issues**
   - Reduce the `max_ram_gb` configuration
   - Use the `chunk_size_mb` configuration to process data in chunks
   - Consider using a subset of the data

### Debug Mode

Enable debug mode to troubleshoot Kaggle integration issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose logging in the bot
bot = NCATradingBot(verbose=True)
```

## Advanced Usage

### Custom Dataset Processing

Implement custom dataset processing for specific Kaggle datasets:

```python
from data_handler import DataHandler

class CustomDataHandler(DataHandler):
    async def process_custom_kaggle_dataset(self, dataset_name):
        # Download dataset
        await self.download_kaggle_dataset(dataset_name)
        
        # Process data
        # Custom processing logic here
        
        return processed_data
```

### Competition Submission

Prepare submission files for Kaggle competitions:

```python
import pandas as pd

def prepare_submission(predictions, test_data):
    submission = pd.DataFrame({
        'id': test_data['id'],
        'target': predictions
    })
    submission.to_csv('submission.csv', index=False)
```

## References

- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Kaggle Competitions](https://www.kaggle.com/competitions)