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

## TPU v5e-8 Integration for Kaggle

The NCA Trading Bot supports TPU v5e-8 (8 cores) using JAX for high-performance training in Kaggle.

### Setting up TPU in Kaggle

1. **Select TPU v5e-8 in Kaggle Notebook**:
   - In your Kaggle notebook, go to Settings > Accelerator
   - Select "TPU v5e-8" (8 cores, 40GB memory)

2. **Install JAX for TPU**:
```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax
```

3. **Run TPU Tests**:
```bash
cd nca_trading_bot
python kaggle_tpu_test_runner.py
```

### TPU Initialization

The `kaggle_tpu_initializer.py` module handles proper TPU initialization:

```python
from kaggle_tpu_initializer import initialize_tpu_for_kaggle

# Initialize TPU v5e-8
results = initialize_tpu_for_kaggle()

if results['initialization_success']:
    print("✅ TPU initialization successful!")
    print(f"Device info: {results['device_info']}")
else:
    print(f"❌ TPU initialization failed: {results['error']}")
```

### TPU Configuration

Configure TPU settings in your configuration:

```python
from config import ConfigManager

config = ConfigManager()

# TPU v5e-8 specific settings
config.tpu.tpu_cores = 8
config.tpu.tpu_topology = "2x4"
config.tpu.use_jax = True
config.tpu.jax_backend = "tpu"

# Mixed precision for TPU efficiency
config.tpu.mixed_precision = True
config.tpu.precision_dtype = "bf16"
config.tpu.compute_dtype = "f32"

# Large batch training
config.tpu.batch_size = 2048
config.tpu.micro_batch_size = 256
config.tpu.gradient_accumulation_steps = 8

# Memory management (35GB out of 40GB available)
config.tpu.max_memory_gb = 35.0
config.tpu.memory_fraction = 0.875

# Sharding configuration
config.tpu.enable_sharding = True
config.tpu.sharding_strategy = "2d"
config.tpu.mesh_shape = (1, 8)
config.tpu.axis_names = ("data", "model")
```

### JAX TPU Training

Train the NCA model on TPU v5e-8:

```python
import jax
from nca_model import create_jax_nca_model, create_jax_train_state
from trainer import JAXPPOTrainer

# Initialize TPU
from kaggle_tpu_initializer import initialize_tpu_for_kaggle
initialize_tpu_for_kaggle()

# Create JAX model
config = get_config()
model = create_jax_nca_model(config)

# Setup sharding for TPU v5e-8
rng = jax.random.PRNGKey(42)
state, mesh = create_jax_train_state(model, config.__dict__, rng)

# Create trainer
trainer = JAXPPOTrainer(observation_dim=60, action_dim=3, config=config)

# Train on TPU
trainer.train(state, mesh)
```

### TPU Testing

Run comprehensive TPU tests:

```bash
# Run all TPU tests
python kaggle_tpu_test_runner.py

# Run individual test components
python -m unittest tests.test_tpu_integration
python -m unittest tests.test_jax_nca_model
```

### TPU Performance Optimization

1. **Large Batch Training**: Use batch sizes of 1024+ for TPU efficiency
2. **Mixed Precision**: Use bfloat16 for computations, float32 for gradients
3. **Sharding**: Distribute data and model across 8 TPU cores
4. **Memory Management**: Limit memory usage to 35GB for stability
5. **XLA Compilation**: Enable XLA compilation for performance

### TPU Troubleshooting

#### Common TPU Issues

1. **"Device or resource busy" Error**:
   - Use the provided `kaggle_tpu_initializer.py` module
   - Ensure proper JAX environment variables are set
   - Restart the Kaggle notebook if needed

2. **TPU Not Detected**:
   - Verify TPU v5e-8 is selected in notebook settings
   - Check JAX installation: `pip install jax[tpu]`
   - Run TPU detection tests

3. **Memory Issues**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision (bfloat16)

4. **Compilation Errors**:
   - Clear JAX cache: `rm -rf /tmp/jax_cache`
   - Restart the notebook
   - Check XLA flags configuration

#### Debug TPU Issues

Enable detailed logging for TPU debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable JAX logging
os.environ["JAX_LOG_COMPILES"] = "1"
os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/xla_dumps"
```

### TPU Memory Management

Monitor and optimize TPU memory usage:

```python
import jax

# Check memory usage
devices = jax.devices()
for device in devices:
    print(f"Device: {device}")
    if hasattr(device, 'memory_stats'):
        stats = device.memory_stats()
        print(f"  Memory: {stats}")

# Memory-efficient training
config.tpu.enable_memory_optimization = True
config.tpu.enable_remat = True
```

### TPU Performance Monitoring

Monitor TPU performance during training:

```python
from utils import PerformanceMonitor

monitor = PerformanceMonitor()

# Get TPU metrics
tpu_metrics = monitor.get_tpu_training_metrics()
print(f"TPU Metrics: {tpu_metrics}")

# Enable XLA profiling
config.tpu.profiling_enabled = True
```

This TPU integration provides significant performance improvements for training the NCA Trading Bot on Kaggle's TPU v5e-8 infrastructure.