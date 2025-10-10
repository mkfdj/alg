# Kaggle TPU Setup Guide for NCA Trading Bot

## 🚀 Kaggle Environment Setup

This guide explains how to set up and run the NCA Trading Bot on Kaggle's TPU v5e-8 environment.

### 📋 Prerequisites

1. **Kaggle Account** with TPU quota
2. **API Token** for dataset downloads
3. **Alpaca Paper Trading Account** (optional for live features)

### ⚙️ Notebook Configuration

#### 1. Create New Notebook
1. Go to Kaggle → Notebooks → Create
2. Set **Accelerator** to **TPU v5e-8**
3. Verify **8 TPU chips** available (128GB HBM total)

#### 2. Environment Variables
```python
import os

# Set up environment
os.environ['JAX_PLATFORM_NAME'] = 'tpu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

# Add your API keys (set in Kaggle Secrets for security)
# ALPACA_PAPER_API_KEY = "your_api_key"
# ALPACA_PAPER_SECRET_KEY = "your_secret_key"
# KAGGLE_USERNAME = "your_username"
# KAGGLE_KEY = "your_kaggle_key"
```

### 📦 Installation

```bash
# Install JAX with TPU support
!pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install core dependencies
!pip install flax optax chex
!pip install pandas numpy scikit-learn ta yfinance
!pip install alpaca-py alpaca-trade-api
!pip install gymnasium stable-baselines3
!pip install wandb tqdm matplotlib seaborn

# Install the NCA Trading Bot
!git clone https://github.com/your-username/nca-trading-bot.git
%cd nca-trading-bot
!pip install -e .
```

### 🔧 JAX/TPU Configuration

```python
import jax
import jax.numpy as jnp
from jax import config, distributed

# Configure JAX for TPU
config.update("jax_enable_x64", False)  # Use float32 for speed
config.update("jax_platform_name", "tpu")
config.update("jax_debug_nans", True)

# Initialize distributed training
try:
    distributed.initialize()
    print(f"Distributed training initialized")
except:
    print("Single device mode")

# Check devices
devices = jax.devices()
print(f"Available devices: {len(devices)} x {devices[0].device_kind}")
print(f"Total memory: {len(devices) * 16} GB HBM")
```

### 📊 Data Download

```python
# Download datasets
!kaggle datasets download -d jacksoncrow/stock-market-dataset
!kaggle datasets download -d jakewright/9000-tickers-of-stock-market-data-full-history

# Extract datasets
!unzip stock-market-dataset.zip -d /kaggle/working/data/
!unzip 9000-tickers-of-stock-market-data-full-history.zip -d /kaggle/working/data/
```

### 🏃‍♂️ Quick Start

```python
from nca_trading_bot import Config, CombinedTrainer

# Create configuration optimized for Kaggle TPU
config = Config()
config.jax_platform = "tpu"
config.tpu_device_count = 8
config.use_float16 = True
config.rl_batch_size = 4096

# Create trainer
trainer = CombinedTrainer(config)

# Train the model
trainer.train(
    nca_iterations=500,  # Reduced for Kaggle time limits
    ppo_iterations=500
)

# Evaluate results
results = trainer.evaluate(num_episodes=100)
print(f"Results: {results}")
```

### 📈 Resource Optimization

#### Memory Management
```python
# Monitor TPU memory usage
import psutil
import jax.profiler

def check_memory():
    # Check system memory
    mem = psutil.virtual_memory()
    print(f"System memory: {mem.percent}% used ({mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB)")

    # Check JAX memory (if available)
    try:
        for device in jax.devices():
            print(f"{device.device_kind}: {device.device_kind}")
    except:
        pass

check_memory()
```

#### Batch Processing
```python
# Optimize batch sizes for TPU
config.rl_batch_size = 4096  # Large batches for TPU efficiency
config.nca_evolution_steps = 48  # Reduced for faster training

# Use mixed precision
config.use_float16 = True
config.mixed_precision = True
```

### 🔍 Monitoring and Debugging

#### Progress Tracking
```python
from tqdm import tqdm
import wandb

# Initialize wandb for tracking
wandb.init(project="nca-trading-bot-kaggle")

# Training with progress bar
for iteration in tqdm(range(num_iterations), desc="Training"):
    # ... training code ...
    wandb.log({"iteration": iteration, "loss": loss})
```

#### Error Handling
```python
import warnings
warnings.filterwarnings('ignore')

# Safe TPU initialization
try:
    distributed.initialize()
    print("✅ TPU initialized successfully")
except Exception as e:
    print(f"⚠️ TPU initialization failed: {e}")
    print("Falling back to CPU/GPU mode")
    config.jax_platform = "cpu"
```

### 📊 Performance Benchmarks

#### Expected Training Times (TPU v5e-8)
- **NCA Training**: ~2-5 minutes for 500 iterations
- **PPO Training**: ~5-10 minutes for 500 iterations
- **Full Pipeline**: ~10-20 minutes total

#### Memory Usage
- **Model Parameters**: ~50MB per ensemble member
- **Data Batches**: ~2-4GB for 4096 batch size
- **TPU Memory**: ~16GB per chip (128GB total)

### ⚠️ Kaggle-Specific Considerations

#### Time Limits
- **Notebook Session**: 12 hours maximum
- **GPU/TPU Quota**: Weekly limits apply
- **Save checkpoints** frequently to avoid losing progress

#### Data Persistence
```python
# Save to Kaggle working directory
import pickle

# Save model checkpoint
checkpoint_path = "/kaggle/working/checkpoint.pkl"
with open(checkpoint_path, 'wb') as f:
    pickle.dump(model_state, f)

# Save results
results_path = "/kaggle/working/results.json"
import json
with open(results_path, 'w') as f:
    json.dump(results, f)
```

#### Internet Access
```python
# Check if internet is available
import urllib.request
try:
    urllib.request.urlopen('https://www.google.com')
    print("✅ Internet access available")
except:
    print("❌ No internet access - using cached data only")
```

### 🧪 Testing

#### Unit Tests
```bash
# Run tests
!python -m pytest nca_trading_bot/tests/ -v
```

#### Integration Tests
```python
# Quick integration test
def test_nca_training():
    config = Config()
    config.nca_evolution_steps = 10  # Quick test
    config.rl_batch_size = 64  # Small batch for testing

    trainer = CombinedTrainer(config)
    trainer.train(nca_iterations=10, ppo_iterations=10)

    print("✅ Integration test passed")

test_nca_training()
```

### 🚀 Production Deployment

#### Export Model
```python
# Save trained model for production
import joblib

model_path = "/kaggle/working/nca_trading_model.joblib"
joblib.dump(trainer, model_path)

# Download model
from IPython.display import FileLink
FileLink(model_path)
```

#### Performance Report
```python
from nca_trading_bot.utils import create_trading_report

# Generate performance report
report = create_trading_report(results, config,
                              save_path="/kaggle/working/performance_report.md")

print(report)
```

### 🔧 Troubleshooting

#### Common Issues

1. **TPU Not Available**
   ```
   Solution: Switch to GPU or CPU mode
   config.jax_platform = "gpu"  # or "cpu"
   ```

2. **Out of Memory**
   ```
   Solution: Reduce batch size
   config.rl_batch_size = 1024  # Reduce from 4096
   ```

3. **Dataset Download Failed**
   ```
   Solution: Check Kaggle API credentials
   !kaggle datasets list  # Test connection
   ```

4. **Training Too Slow**
   ```
   Solution: Enable all optimizations
   config.use_float16 = True
   config.compile_nca_evolution = True
   config.compile_rl_training = True
   ```

#### Performance Tips

1. **Use larger batches** for TPU efficiency
2. **Enable float16** for memory and speed
3. **Compile functions** with @jax.jit
4. **Monitor memory usage** regularly
5. **Save checkpoints** every 100 iterations

### 📚 Additional Resources

- [JAX on TPU Documentation](https://docs.jax.dev/en/latest/tpu.html)
- [Kaggle TPU Guide](https://www.kaggle.com/docs/tpu)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Optax Optimization](https://optax.readthedocs.io/)

### ⚠️ Important Notes

1. **Wait for User Approval**: Never enable live trading without explicit confirmation
2. **Paper Trading Default**: Always use paper trading mode for testing
3. **API Security**: Store API keys in Kaggle Secrets, not in code
4. **Resource Limits**: Monitor TPU quota and session time limits
5. **Data Validation**: Always validate data quality before training

---

**🚀 Ready to start! Run the cells above to begin your NCA Trading Bot journey on Kaggle TPU!**