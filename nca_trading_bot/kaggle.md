# NCA Trading Bot - Kaggle Setup Guide

This guide provides step-by-step instructions for setting up and running the NCA Trading Bot on Kaggle with GPU and TPU acceleration.

## 🚀 Kaggle Environment Setup

### 1. Create Kaggle Account

1. Go to [kaggle.com](https://kaggle.com)
2. Sign up for a free account
3. Verify your email address

### 2. Enable GPU/TPU Access

1. Go to your Kaggle profile
2. Navigate to "Settings" → "Account"
3. Scroll down to "Accelerator" section
4. Choose your preferred accelerator:
   - **GPU T4 x2** (recommended for most users)
   - **GPU P100** (alternative GPU option)
   - **TPU v3-8** (for TPU-optimized training)

### 3. Create New Notebook

1. Click "Code" → "New Notebook"
2. Choose your preferred accelerator:
   - **GPU T4 x2** (recommended for most users)
   - **TPU v3-8** (for TPU-optimized training)
3. Select "Python" as language
4. Give your notebook a name (e.g., "NCA-Trading-Bot")

## 📦 Installation & Setup

### 1. Install Dependencies

Run this in a Kaggle notebook cell:

```python
# Install core dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install yfinance alpaca-trade-api pandas numpy matplotlib seaborn
!pip install gymnasium scikit-learn
!pip install tensorboard wandb

# Install TPU support (optional)
!pip install torch-xla -f https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.3.0+libtpu-cp310-cp310-linux_x86_64.whl

# Install TA-Lib for technical indicators
!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
!tar -xzf ta-lib-0.4.0-src.tar.gz
!cd ta-lib && ./configure --prefix=/usr/local && make && make install
!pip install talib

# Install additional dependencies
!pip install python-dotenv rich psutil
```

### 2. Clone/Download Project Files

Since Kaggle doesn't support git clone in free tier, you'll need to upload files manually:

1. **Option A: Upload Files**
   - Download all Python files from the repository
   - Click "Data" → "Upload" in Kaggle sidebar
   - Upload each `.py` file

2. **Option B: Create Files in Notebook**
   - Create new text files in Kaggle
   - Copy and paste code from each module

### 3. Set Up Environment Variables

Create a cell with environment variables:

```python
import os

# Set API keys (replace with your actual keys)
os.environ['ALPACA_API_KEY'] = 'your_alpaca_api_key'
os.environ['ALPACA_SECRET_KEY'] = 'your_alpaca_secret_key'

# Optional: Set other API keys
os.environ['ALPHA_VANTAGE_KEY'] = 'your_alpha_vantage_key'
os.environ['WANDB_API_KEY'] = 'your_wandb_key'

# Set Kaggle-specific paths
os.environ['NCA_DATA_DIR'] = '/kaggle/working/data'
os.environ['NCA_MODEL_DIR'] = '/kaggle/working/models'
os.environ['NCA_LOG_DIR'] = '/kaggle/working/logs'
```

## 🏃‍♂️ Running the Bot

### 1. Import Modules

```python
# Import all modules
from config import get_config, ConfigManager
from data_handler import DataHandler
from nca_model import create_nca_model, NCATradingModel
from trader import create_trading_environment, TradingAgent
from trainer import TrainingManager
from utils import initialize_utils, PerformanceMonitor
from main import NCACommandLineInterface

# Initialize configuration
config = get_config()
initialize_utils(config)
```

### 2. Quick Start Example

```python
# Create sample data for testing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample market data
dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
sample_data = pd.DataFrame({
    'timestamp': dates,
    'Open': np.random.randn(1000).cumsum() + 100,
    'High': np.random.randn(1000).cumsum() + 101,
    'Low': np.random.randn(1000).cumsum() + 99,
    'Close': np.random.randn(1000).cumsum() + 100,
    'Volume': np.random.randint(1000, 10000, 1000)
})

print(f"Sample data shape: {sample_data.shape}")
```

### 3. Model Training

```python
# Create training manager
trainer = TrainingManager(config)

# Create model
model = trainer.create_model()
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Train model
await trainer.train_offline(sample_data, num_epochs=10)
```

### 4. Backtesting

```python
# Create trading environment
env = create_trading_environment(sample_data, config)

# Create trading agent
trading_agent = create_trading_agent(model, config)

# Run backtest
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    # Get trading decision
    decision = trading_agent.make_decision(sample_data.iloc[:env.current_step + 60])

    # Convert to environment action
    if decision['action'] == 'buy':
        action = [1, 5]  # Buy with 50% position
    elif decision['action'] == 'sell':
        action = [2, 5]  # Sell with 50% position
    else:
        action = [0, 0]  # Hold

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"Backtest completed: Total reward = {total_reward".2f"}")
print(f"Final portfolio value = ${info['portfolio_value']".2f"}")
```

## 🔧 Kaggle-Specific Optimizations

### 1. GPU Memory Management

```python
import torch
import gc

# Check GPU memory
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3".2f"} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3".2f"} GB")

# Check TPU memory (if using TPU)
try:
    import torch_xla.core.xla_model as xm
    memory_info = xm.get_memory_info(xm.xla_device())
    print(f"TPU memory used: {memory_info.get('used_bytes', 0)/1024**3".2f"} GB")
    print(f"TPU memory total: {memory_info.get('total_bytes', 0)/1024**3".2f"} GB")
except ImportError:
    print("TPU not available")

# Clear memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
```

### 2. TPU Memory Management

```python
# Check TPU availability and memory
try:
    import torch_xla.core.xla_model as xm

    print(f"TPU available: {xm.is_master_ordinal()}")
    print(f"TPU devices: {xm.xrt_world_size()}")

    # Get TPU memory info
    memory_info = xm.get_memory_info(xm.xla_device())
    print(f"TPU memory used: {memory_info.get('used_bytes', 0)/1024**3".2f"} GB")
    print(f"TPU memory total: {memory_info.get('total_bytes', 0)/1024**3".2f"} GB")

    # Clear TPU memory
    xm.mark_step()

except ImportError:
    print("TPU not available - using GPU/CPU")
```

### 3. TPU-Optimized Training

```python
# Configure for TPU training
config.system.device = "tpu"
config.tpu.xla_compile = True
config.tpu.sharding_strategy = "2d"

# Create TPU-optimized model
trainer = TrainingManager(config)
model = trainer.create_model()

# Train with TPU optimizations
print(f"Training on TPU with {config.tpu.tpu_cores} cores")
await trainer.train_offline(data, num_epochs=50)
```
```

### 2. Data Persistence

```python
# Save data to Kaggle working directory
data_path = '/kaggle/working/market_data.parquet'
sample_data.to_parquet(data_path)

# Load data
loaded_data = pd.read_parquet(data_path)
print(f"Data loaded from {data_path}")
```

### 3. Model Checkpointing

```python
# Save model
model_path = '/kaggle/working/nca_model.pt'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Load model
model.load_state_dict(torch.load(model_path))
print("Model loaded successfully")
```

## 📊 Monitoring & Visualization

### 1. System Metrics

```python
from utils import PerformanceMonitor

monitor = PerformanceMonitor()

# Get system metrics
system_metrics = monitor.get_system_metrics()
print("System Metrics:")
for key, value in system_metrics.items():
    print(f"  {key}: {value}")

# Get trading metrics
trading_metrics = monitor.get_trading_metrics(trades)
print("Trading Metrics:")
for key, value in trading_metrics.items():
    print(f"  {key}: {value}")
```

### 2. TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

# Create TensorBoard writer
writer = SummaryWriter('/kaggle/working/logs')

# Log metrics
for epoch in range(100):
    metrics = {'loss': 0.5, 'accuracy': 0.85}
    for key, value in metrics.items():
        writer.add_scalar(f'train/{key}', value, epoch)

# View TensorBoard
# Click on "Data" → "Output" → "tensorboard" in Kaggle sidebar
```

### 3. Plotting

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot price data
plt.figure(figsize=(12, 6))
plt.plot(sample_data['Close'])
plt.title('Sample Price Data')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

# Plot technical indicators
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes[0,0].plot(sample_data['Close'])
axes[0,0].set_title('Price')
axes[0,1].plot(sample_data['Volume'])
axes[0,1].set_title('Volume')
plt.tight_layout()
plt.show()
```

## 🚨 Common Issues & Solutions

### 1. GPU Memory Issues

```python
# Solution: Use gradient checkpointing
model.gradient_checkpointing_enable()

# Or reduce batch size
config.training.batch_size = 32  # Reduce from 64

# Or use mixed precision
config.training.use_amp = True
```

### 2. Out of Memory Errors

```python
# Clear cache
torch.cuda.empty_cache()

# Use smaller model
config.nca.state_dim = 32  # Reduce from 64
config.nca.hidden_dim = 64  # Reduce from 128

# Process data in chunks
chunk_size = 1000
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    # Process chunk
```

### 3. Installation Issues

```python
# If TA-Lib installation fails
!apt-get update
!apt-get install -y build-essential
!pip install --no-cache-dir talib

# If PyTorch installation fails
!pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 4. API Connection Issues

```python
# Test API connections
import os
from alpaca_trade_api.rest import REST

api = REST(
    key_id=os.environ.get('ALPACA_API_KEY', ''),
    secret_key=os.environ.get('ALPACA_SECRET_KEY', ''),
    base_url='https://paper-api.alpaca.markets'
)

try:
    account = api.get_account()
    print(f"API connected: ${account.equity}")
except Exception as e:
    print(f"API connection failed: {e}")
```

## 💾 Saving Your Work

### 1. Download Files

```python
# Save model
torch.save(model.state_dict(), 'nca_model.pt')

# Save data
sample_data.to_parquet('market_data.parquet')

# Download files from Kaggle
from IPython.display import FileLink
FileLink('nca_model.pt')
FileLink('market_data.parquet')
```

### 2. Commit to Git

```python
# Add files to git
!git add nca_model.pt market_data.parquet

# Commit changes
!git commit -m "Add trained model and data"

# Push to repository
!git push origin main
```

### 3. Kaggle Dataset

```python
# Create dataset from output files
!kaggle datasets init -p /kaggle/working

# Create metadata
import json
metadata = {
    "title": "NCA Trading Bot Models",
    "id": "your-username/nca-trading-bot-models",
    "licenses": [{"name": "CC0-1.0"}]
}

with open('/kaggle/working/dataset-metadata.json', 'w') as f:
    json.dump(metadata, f)

# Create dataset
!kaggle datasets create -p /kaggle/working
```

## 🔄 Best Practices for Kaggle

### 1. Session Management

```python
# Save progress regularly
import pickle
import datetime

# Save state
state = {
    'model': model.state_dict(),
    'config': config,
    'timestamp': datetime.now()
}

with open('/kaggle/working/training_state.pkl', 'wb') as f:
    pickle.dump(state, f)

# Load state
with open('/kaggle/working/training_state.pkl', 'rb') as f:
    state = pickle.load(f)
    model.load_state_dict(state['model'])
```

### 2. Resource Monitoring

```python
# Monitor GPU usage
!nvidia-smi

# Monitor memory usage
import psutil
print(f"CPU: {psutil.cpu_percent()}%")
print(f"Memory: {psutil.virtual_memory().percent}%")

# Monitor disk usage
!df -h
```

### 3. Efficient Training

```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Training loop with AMP
for batch in dataloader:
    with autocast():
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 📈 Example Workflow

### Complete Training Session

```python
# 1. Setup
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# 2. Data preparation
# ... (load or generate data)

# 3. Model training
trainer = TrainingManager(config)
model = trainer.create_model()

# 4. Training loop
for epoch in range(10):
    # ... (training code)
    print(f"Epoch {epoch}: Loss = {loss".4f"}")

# 5. Save results
torch.save(model.state_dict(), '/kaggle/working/nca_model_final.pt')
print("Training completed and model saved!")
```

## 🎯 Next Steps

1. **Experiment Tracking**: Use Weights & Biases for experiment tracking
2. **Hyperparameter Tuning**: Use Kaggle's parameter optimization
3. **Model Comparison**: Compare different architectures
4. **Production Deployment**: Deploy trained models to live trading

## 📞 Support

- **Kaggle Forums**: [kaggle.com/discussions](https://kaggle.com/discussions)
- **GitHub Issues**: [github.com/your-username/nca-trading-bot/issues](https://github.com/your-username/nca-trading-bot/issues)
- **Documentation**: [github.com/your-username/nca-trading-bot/wiki](https://github.com/your-username/nca-trading-bot/wiki)

---

**Happy Kaggle Computing! 🚀💻**