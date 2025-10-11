# Coding Datasets Manager

A comprehensive system for managing 18+ coding datasets from Kaggle, designed for machine learning training and research.

## Features

- **18 Integrated Datasets**: Access to major coding datasets including HumanEval, MBPP, CodeParrot, and more
- **Multiple Format Support**: CSV, JSON, JSONL, Lance, Parquet
- **Automatic Downloading**: Kaggle API integration for seamless dataset acquisition
- **Data Preprocessing**: Built-in cleaning, validation, and formatting utilities
- **Training Data Creation**: Format datasets for various ML training approaches
- **Quality Validation**: Comprehensive data quality assessment and reporting
- **Flexible Configuration**: YAML/JSON-based configuration system

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Setup Kaggle API credentials (choose one method):

# Method 1: Environment Variables (Recommended)
export KAGGLE_USERNAME='mautlej'
export KAGGLE_KEY='your_kaggle_api_key_here'

# Method 2: Python script
import os
os.environ['KAGGLE_USERNAME'] = 'mautlej'
os.environ['KAGGLE_KEY'] = 'your_kaggle_api_key_here'

# Method 3: Command line arguments
python main.py --kaggle-username mautlej --kaggle-key your_key download dataset_name

# Method 4: Configuration file
# Create config.yaml with:
# data:
#   kaggle_username: 'mautlej'
#   kaggle_key: 'your_kaggle_api_key_here'

# Method 5: kaggle.json file (traditional)
# 1. Create account at kaggle.com
# 2. Download kaggle.json from account settings
# 3. Place kaggle.json in ~/.kaggle/ or specify path
```

### Basic Usage

```python
import os
from coding_datasets import DatasetManager

# Set Kaggle credentials
os.environ['KAGGLE_USERNAME'] = 'mautlej'
os.environ['KAGGLE_KEY'] = 'your_kaggle_api_key_here'

# Initialize manager
manager = DatasetManager()

# List all available datasets
datasets = manager.list_datasets()
print(datasets)

# Download a specific dataset
manager.download_dataset("handcrafted_code_gen")

# Extract prompts from dataset
prompts = manager.extract_prompts("handcrafted_code_gen")
print(f"Extracted {len(prompts)} prompts")

# Create training data
training_data = manager.create_training_data(
    dataset_ids=["handcrafted_code_gen", "openai_human_eval"],
    format_type="instruction"
)

# Export training data
manager.export_data(training_data, "training_data.jsonl")
```

### Command Line Usage

```bash
# List all datasets
python main.py list

# Download specific datasets
python main.py download handcrafted_code_gen openai_human_eval

# Download all datasets
python main.py download --all

# Create training data
python main.py process handcrafted_code_gen --format instruction --output training.jsonl

# Validate datasets
python main.py validate handcrafted_code_gen

# Show dataset information
python main.py info --dataset handcrafted_code_gen
```

## Available Datasets

| Dataset ID | Name | Size | Format | Description |
|------------|------|------|--------|-------------|
| `handcrafted_code_gen` | OpenAI HumanEval | 500 KB | CSV | 164 handcrafted Python coding problems |
| `openai_human_eval` | OpenAI HumanEval Mirror | 200 KB | JSONL | Original OpenAI evaluation set |
| `mbpp_python_problems` | MBPP Python Problems | 1 MB | JSONL | 1000 basic Python problems |
| `codeparrot_1m` | CodeParrot 1M | 100 MB | Lance | 1M Python code files from GitHub |
| `python_code_instruction` | Python Code Instructions | 50 MB | CSV | Instruction-based code tasks |
| `glaive_python_code_qa` | Glaive Python QA | 100 MB | CSV | 140K Python Q&A pairs |
| `evol_instruct_code` | Evol Instruct Code | 200 MB | JSON | 80K evolved code instructions |
| ...and 11 more datasets |

## Configuration

The system uses a flexible configuration system. Create a `config.yaml` file:

```yaml
data:
  download_path: "./coding_datasets/data"
  cache_enabled: true
  max_concurrent_downloads: 3

preprocessing:
  normalize_whitespace: true
  remove_duplicates: true
  filter_by_length: true
  min_length: 10
  max_length: 10000

training:
  format: "instruction"
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42

datasets:
  enabled:
    - handcrafted_code_gen
    - openai_human_eval
    - mbpp_python_problems
```

## Data Processing Pipeline

1. **Download**: Fetch datasets from Kaggle
2. **Load**: Parse various formats (CSV, JSON, JSONL, etc.)
3. **Extract**: Get prompts and responses from raw data
4. **Preprocess**: Clean, filter, and normalize text
5. **Validate**: Assess data quality and integrity
6. **Format**: Convert to training-ready formats
7. **Export**: Save in desired format (JSONL, JSON, etc.)

## Training Formats

- **Instruction**: Simple instruction-response pairs
- **Alpaca**: Alpaca-style format with input/output fields
- **Chat**: Chat message format for conversational models
- **Code Completion**: Prefix-completion format for code models

## Quality Validation

The system includes comprehensive data validation:

- **Basic Checks**: Empty content, encoding issues, length bounds
- **Content Quality**: Duplicate detection, suspicious patterns
- **Code Quality**: Syntax validation, security checks
- **Language Consistency**: Format and language analysis

## API Reference

### DatasetManager

Main class for managing datasets:

```python
manager = DatasetManager(config_path="config.yaml")

# Dataset operations
manager.download_dataset(dataset_id)
manager.load_dataset(dataset_id)
manager.extract_prompts(dataset_id)

# Processing
manager.preprocess_dataset(dataset_id)
manager.validate_dataset(dataset_id)
manager.create_training_data(dataset_ids)

# Utilities
manager.get_dataset_info()
manager.get_statistics()
manager.export_data()
```

### Data Components

- **Registry**: Dataset metadata and information
- **Downloader**: Kaggle API integration
- **Loaders**: Format-specific data loading
- **Preprocessor**: Text cleaning and normalization
- **Validator**: Quality assessment
- **Formatter**: Training data formatting

## Examples

### Example 1: Quick Start

```python
from coding_datasets import DatasetManager

# Initialize with default config
manager = DatasetManager()

# Download and process a single dataset
manager.download_dataset("handcrafted_code_gen")
prompts = manager.extract_prompts("handcrafted_code_gen")
training_data = manager.create_training_data(["handcrafted_code_gen"])
manager.export_data(training_data, "quick_start.jsonl")
```

### Example 2: Custom Processing

```python
from coding_datasets import DatasetManager
from coding_datasets.utils import PromptFormatter

# Initialize with custom config
manager = DatasetManager(config_path="my_config.yaml")

# Process multiple datasets
datasets = ["handcrafted_code_gen", "mbpp_python_problems", "glaive_python_code_qa"]

# Custom preprocessing
for dataset_id in datasets:
    prompts = manager.extract_prompts(dataset_id)
    # Apply custom preprocessing
    processed = manager.preprocess_dataset(dataset_id, prompts, {
        "min_length": 50,
        "max_length": 5000,
        "remove_duplicates": True
    })

# Create custom format
formatter = PromptFormatter()
formatted = formatter.format_prompts(
    prompts,
    format_type=PromptFormat.CHAT
)
```

### Example 3: Quality Validation

```python
from coding_datasets import DatasetManager

manager = DatasetManager()

# Validate all datasets
datasets = manager.list_datasets().keys()
for dataset_id in datasets:
    try:
        results = manager.validate_dataset(dataset_id)
        if results['quality_score'] < 70:
            print(f"Low quality dataset: {dataset_id} ({results['quality_score']})")
    except Exception as e:
        print(f"Could not validate {dataset_id}: {e}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

## Changelog

### v1.0.0
- Initial release with 18 integrated datasets
- Full Kaggle API integration
- Comprehensive preprocessing pipeline
- Multiple training format support
- Quality validation system
- Command-line interface
- Flexible configuration management