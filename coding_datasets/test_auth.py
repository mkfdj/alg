"""
Test script to verify Kaggle authentication works
"""

import os
import logging

# Set your credentials
os.environ['KAGGLE_USERNAME'] = 'mautlej'
# You need to set your actual Kaggle API key
# os.environ['KAGGLE_KEY'] = 'your_actual_kaggle_api_key_here'

def test_kaggle_auth():
    """Test Kaggle authentication"""
    print("🔐 Testing Kaggle Authentication...")
    print(f"Username: {os.environ.get('KAGGLE_USERNAME', 'Not set')}")
    print(f"Key: {'Set' if os.environ.get('KAGGLE_KEY') else 'Not set'}")

    try:
        from coding_datasets import DatasetManager

        # Initialize manager
        print("\n📦 Initializing DatasetManager...")
        manager = DatasetManager()

        # Test by listing datasets (this requires authentication)
        print("\n📋 Testing dataset listing...")
        datasets = manager.list_datasets()
        print(f"✅ Successfully listed {len(datasets)} datasets!")

        # Show a few datasets
        print("\n🎯 Sample datasets:")
        for i, (dataset_id, info) in enumerate(list(datasets.items())[:3]):
            print(f"  {i+1}. {dataset_id}: {info['name']}")

        return True

    except Exception as e:
        print(f"❌ Authentication failed: {str(e)}")
        print("\n💡 Possible solutions:")
        print("  1. Check your Kaggle username and API key")
        print("  2. Make sure you have a valid Kaggle account")
        print("  3. Generate a new API token from https://kaggle.com/your-account")
        return False

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run test
    success = test_kaggle_auth()

    if success:
        print("\n🎉 Authentication successful! You can now download datasets.")
        print("\nExample commands:")
        print("  python main.py download handcrafted_code_gen")
        print("  python main.py list")
    else:
        print("\n❌ Please fix authentication issues before proceeding.")