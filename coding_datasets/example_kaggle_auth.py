"""
Example script demonstrating Kaggle authentication
"""

import os
from coding_datasets import DatasetManager

# Method 1: Using environment variables (recommended)
os.environ['KAGGLE_USERNAME'] = 'mautlej'
os.environ['KAGGLE_KEY'] = 'your_kaggle_api_key_here'  # Replace with your actual key

# Method 2: Via configuration file
# Create a config.yaml with:
# data:
#   kaggle_username: 'mautlej'
#   kaggle_key: 'your_kaggle_api_key_here'

# Method 3: Via command line arguments
# python main.py --kaggle-username mautlej --kaggle-key your_key download handcrafted_code_gen

def main():
    """Example usage with Kaggle authentication"""
    print("🔐 Setting up Kaggle authentication...")

    # Initialize with environment variables already set
    manager = DatasetManager()

    # List available datasets
    print("\n📋 Available datasets:")
    datasets = manager.list_datasets()
    for dataset_id, info in list(datasets.items())[:3]:  # Show first 3
        print(f"  • {dataset_id}: {info['name']}")
    print(f"  ... and {len(datasets)-3} more datasets")

    # Download a small dataset as test
    try:
        print("\n⬇️  Downloading a test dataset...")
        manager.download_dataset("handcrafted_code_gen")
        print("✅ Download successful!")

        # Extract prompts
        print("\n📝 Extracting prompts...")
        prompts = manager.extract_prompts("handcrafted_code_gen")
        print(f"✅ Extracted {len(prompts)} prompts")

        # Show a sample
        print(f"\n💡 Sample prompt:")
        print(f"   {prompts[0][:200]}..." if prompts else "No prompts found")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Make sure your Kaggle credentials are correct:")
        print("   1. Visit https://kaggle.com/your-account")
        print("   2. Click 'Create New API Token'")
        print("   3. Update the KAGGLE_KEY in this script")

if __name__ == "__main__":
    main()