"""
Simple test to verify the system works without arguments
"""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic functionality without requiring credentials"""
    print("🧪 Testing basic functionality...")

    try:
        # Test import
        from dataset_manager import DatasetManager
        print("✅ Successfully imported DatasetManager")

        # Test initialization (without Kaggle API for now)
        manager = DatasetManager()
        print("✅ Successfully initialized DatasetManager")

        # Test listing datasets
        datasets = manager.list_datasets()
        print(f"✅ Successfully listed {len(datasets)} datasets")

        # Show first few datasets
        print("\n📋 First 5 datasets:")
        for i, (dataset_id, info) in enumerate(list(datasets.items())[:5]):
            print(f"  {i+1}. {dataset_id}: {info['name']}")

        # Test authentication (without credentials)
        print("\n🔐 Testing authentication system...")
        if manager.downloader.api is None:
            print("⚠️  Kaggle API not authenticated (expected without credentials)")
            print("💡 Set credentials with:")
            print("   export KAGGLE_USERNAME='your_username'")
            print("   export KAGGLE_KEY='your_api_key'")
        else:
            print("✅ Kaggle API authenticated")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_with_credentials():
    """Test with environment variable credentials"""
    print("\n🔐 Testing with environment variables...")

    # Check if credentials are set
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')

    if not username or not key:
        print("⚠️  Kaggle credentials not set in environment variables")
        print("💡 Set them with:")
        print("   export KAGGLE_USERNAME='mautlej'")
        print("   export KAGGLE_KEY='your_api_key'")
        return False

    try:
        from dataset_manager import DatasetManager
        manager = DatasetManager()

        if manager.downloader.api:
            print("✅ Successfully authenticated with environment variables")
            return True
        else:
            print("❌ Failed to authenticate with environment variables")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Coding Datasets Test Suite")
    print("=" * 50)

    # Test basic functionality
    basic_success = test_basic_functionality()

    # Test with credentials if available
    test_with_credentials()

    print("\n" + "=" * 50)
    if basic_success:
        print("🎉 Basic tests passed! System is working correctly.")
        print("\n💡 Next steps:")
        print("1. Set Kaggle credentials:")
        print("   export KAGGLE_USERNAME='mautlej'")
        print("   export KAGGLE_KEY='your_api_key'")
        print("2. Run: python main.py list")
        print("3. Run: python main.py download handcrafted_code_gen")
    else:
        print("❌ Basic tests failed. Check the error messages above.")