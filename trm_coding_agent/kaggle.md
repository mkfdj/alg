# Kaggle Notebook Setup Guide for TRM Coding Agent

This guide explains how to set up and run the TRM Coding Agent on Kaggle using TPU v5e-8.

## 🚀 Quick Setup

### 1. Create New Notebook

1. Go to [Kaggle](https://kaggle.com)
2. Click **Create** → **New Notebook**
3. Set **Accelerator** → **TPU VM v5e-8** (8 chips, 128GB HBM)
4. Set **Internet** → **Internet on** (required for downloads)

### 2. Install Dependencies

```python
# Install JAX with TPU support
!pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install TRM Coding Agent dependencies
!pip install flax optax chex tokenizers pandas numpy tqdm wandb tensorboard lance pyarrow
```

### 3. Clone Repository

```python
!git clone https://github.com/your-org/trm-coding-agent.git
%cd trm-coding-agent
```

### 4. Setup Kaggle API

```python
# Upload your kaggle.json file (from Kaggle account)
# Or use kaggle secrets
import os
os.environ['KAGGLE_USERNAME'] = 'your-username'
os.environ['KAGGLE_KEY'] = 'your-api-key'

# Install kaggle CLI
!pip install kaggle
```

## 📊 Dataset Setup

### Download Datasets

```python
# Download all datasets
!python -m trm_coding_agent.datasets.download_datasets --all

# Or download specific datasets
!python -m trm_coding_agent.datasets.download_datasets --datasets "OpenAI HumanEval" "MBPP Python Problems"
```

### Verify Downloads

```python
from trm_coding_agent.datasets.download_datasets import DatasetDownloader

downloader = DatasetDownloader("datasets")
results = downloader.download_all_datasets(["OpenAI HumanEval", "MBPP Python Problems"])
print(f"Download results: {results}")
```

## 🧠 Model Training

### Basic Training

```python
import sys
sys.path.append('/kaggle/working/trm-coding-agent')

from trm_coding_agent.main import train_model
import argparse

# Create mock arguments for training
class Args:
    def __init__(self):
        self.config = "tpu"
        self.epochs = 10
        self.datasets = ["OpenAI HumanEval", "MBPP Python Problems"]
        self.use_tpu = True

args = Args()

# Start training
train_model(args)
```

### Advanced Training with Custom Configuration

```python
from trm_coding_agent import get_config, TRMTrainer, create_trm_model
from trm_coding_agent.data_handler import DataHandler

# Setup TPU-optimized configuration
config = get_config("tpu")
config.trm.batch_size = 1024  # Large batch for TPU
config.trm.recursion_depth = 16
config.trm.hidden_size = 512

# Create trainer
model = create_trm_model(config.trm)
trainer = PPOTrainer(model, config)

# Prepare data
trainer.prepare_data(["OpenAI HumanEval", "MBPP Python Problems"])

# Start training
trainer.train(num_epochs=20)
```

## 🎯 Code Generation

### Interactive Mode

```python
from trm_coding_agent.main import interactive_mode
import argparse

class Args:
    def __init__(self):
        self.config = "tpu"
        self.checkpoint = "/kaggle/working/trm_coding_agent/checkpoints/best_model.pkl"

args = Args()
interactive_mode(args)
```

### Batch Generation

```python
from trm_coding_agent import TRMModel, DataHandler, get_config
from trm_coding_agent.utils import load_checkpoint

# Load model
config = get_config("tpu")
model = create_trm_model(config.trm)
data_handler = DataHandler(config)

# Load checkpoint
checkpoint_path = "/kaggle/working/trm_coding_agent/checkpoints/best_model.pkl"
checkpoint = load_checkpoint(checkpoint_path)
train_state = checkpoint['state']

# Prepare tokenizer
data_handler.setup_tokenizer()

# Generate code for multiple prompts
prompts = [
    "Write a function to calculate factorial",
    "Implement binary search algorithm",
    "Create a function to validate email addresses"
]

for prompt in prompts:
    # Tokenize
    from trm_coding_agent.data_handler import DatasetSample
    sample = DatasetSample(prompt, "", "", 0, {})
    tokenized = data_handler.tokenize_samples([sample])

    # Generate
    generated_ids = model.generate(
        train_state.params,
        tokenized['prompt_input_ids'],
        max_length=256,
        deterministic=True
    )

    print(f"Prompt: {prompt}")
    print(f"Generated code length: {generated_ids.shape[1]}")
    print("-" * 50)
```

## 📈 Monitoring

### Setup Weights & Biases

```python
import wandb

# Login to wandb
wandb.login(key='your-wandb-api-key')

# Enable wandb in config
config.logging.use_wandb = True
config.logging.wandb_project = "trm-coding-agent-kaggle"
```

### Setup TensorBoard

```python
# Start tensorboard
%load_ext tensorboard
%tensorboard --logdir logs/tensorboard
```

### Monitor TPU Usage

```python
import jax

print(f"JAX devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")
print(f"Device type: {jax.devices()[0].device_kind}")

# Check memory usage
import psutil
memory = psutil.virtual_memory()
print(f"Available memory: {memory.available / (1024**3):.2f} GB")
```

## 🔧 Optimization Tips

### Memory Optimization

```python
# Use binary activations for 32x memory reduction
config.trm.use_binary_activations = True
config.trm.binary_binarization = True

# Use bfloat16 for TPU
config.trm.dtype = jnp.bfloat16

# Large batch size for TPU
config.trm.batch_size = 4096
```

### Data Loading Optimization

```python
# Limit samples for faster iteration
config.trm.max_samples_per_dataset = 1000

# Enable data augmentation
config.dataset.use_data_augmentation = True

# Optimize data loading
config.dataset.num_workers = 4
config.dataset.prefetch_factor = 2
```

## 🐛 Common Issues and Solutions

### TPU Initialization Issues

```python
# Force TPU initialization
import jax
jax.config.update('jax_platform_name', 'tpu')

# Check TPU status
try:
    devices = jax.devices()
    print(f"TPU devices found: {len(devices)}")
except Exception as e:
    print(f"TPU initialization failed: {e}")
    print("Falling back to CPU")
    jax.config.update('jax_platform_name', 'cpu')
```

### Memory Issues

```python
# Reduce batch size if OOM
config.trm.batch_size = 256

# Enable gradient checkpointing
config.trm.use_gradient_checkpointing = True

# Use smaller model
config.trm.hidden_size = 256
config.trm.intermediate_size = 1024
```

### Dataset Download Issues

```python
# Check kaggle authentication
!kaggle datasets list

# Download individual datasets
!python -m trm_coding_agent.datasets.download_datasets --datasets "OpenAI HumanEval"

# Clean up corrupted downloads
!python -m trm_coding_agent.datasets.download_datasets --cleanup
```

## 📊 Kaggle-Specific Optimizations

### Use Kaggle Datasets

```python
# Add existing Kaggle datasets to your notebook
# Search for: "python code", "programming challenges", "coding datasets"

# Example: Add HumanEval dataset
!kaggle datasets download -d thedevastator/handcrafted-dataset-for-code-generation-models --unzip
```

### Optimize for Kaggle Environment

```python
# Use Kaggle's high-memory environment
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Optimize JAX for TPU
import jax
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_platform_name', 'tpu')
```

### Save to Kaggle Datasets

```python
# Save your trained model as a Kaggle dataset
!kaggle datasets init -d /kaggle/working/trm-coding-agent-model
!kaggle datasets create -p /kaggle/working/trm-coding-agent-model --dir-mode zip
```

## 🎯 Example Workflow

### Complete Training Pipeline

```python
# 1. Setup environment
!pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
!git clone https://github.com/your-org/trm-coding-agent.git
%cd trm-coding-agent
!pip install -r requirements.txt

# 2. Setup Kaggle API
!mkdir -p ~/.kaggle
# Upload your kaggle.json file here

# 3. Download datasets
!python -m trm_coding_agent.datasets.download_datasets --all

# 4. Configure and train
import sys
sys.path.append('/kaggle/working/trm-coding-agent')

from trm_coding_agent import get_config, TRMTrainer, create_trm_model

# TPU-optimized config
config = get_config("tpu")
config.trm.batch_size = 2048
config.trm.recursion_depth = 16
config.trm.hidden_size = 512

# Create and train
model = create_trm_model(config.trm)
trainer = TRMTrainer(model, config)
trainer.prepare_data(["OpenAI HumanEval", "MBPP Python Problems"])
trainer.train(num_epochs=15)

# 5. Save results
!kaggle datasets init -d /kaggle/working/trm-coding-agent-results
!kaggle datasets create -p /kaggle/working/trm-coding-agent-results --dir-mode zip
```

## 📚 Additional Resources

- [JAX on TPU Guide](https://jax.readthedocs.io/en/latest/jax-101/08-pjnp.html)
- [Kaggle TPU Documentation](https://www.kaggle.com/docs/tpu)
- [TRM Paper](https://arxiv.org/abs/2510.04871)
- [Flax Documentation](https://flax.readthedocs.io/)

## 🤝 Support

If you encounter issues:

1. Check the **Output** tab for error messages
2. Verify **TPU** is properly selected
3. Ensure **Internet** is enabled
4. Check **Dependencies** are correctly installed

For additional support, create an issue on the GitHub repository.