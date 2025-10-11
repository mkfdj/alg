#!/usr/bin/env python3
"""
Test one dataset to verify everything works
"""

import os
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_one_dataset():
    """Test downloading one small dataset"""
    print("🧪 Testing One Dataset Download")
    print("=" * 40)

    # Check credentials
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')

    if not username or not key:
        print("❌ Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        return False

    print(f"✅ Kaggle credentials ready for: {username}")

    try:
        # Test imports
        from dataset_manager import DatasetManager
        print("✅ Imports working")

        # Initialize manager
        manager = DatasetManager()
        print("✅ DatasetManager initialized")

        # Test one small dataset
        dataset_id = "openai_humaneval_code_gen"  # 196.17 kB - very small
        print(f"\n📥 Testing download of: {dataset_id}")

        try:
            file_path = manager.download_dataset(dataset_id)
            print(f"✅ Downloaded to: {file_path}")

            # Test extraction
            print(f"\n📝 Testing prompt extraction...")
            prompts = manager.extract_prompts(dataset_id)
            print(f"✅ Extracted {len(prompts)} prompts")

            # Show sample
            if prompts:
                print(f"\n📋 Sample prompt:")
                print(f"   {prompts[0][:200]}...")

            return True

        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_one_dataset()

    if success:
        print("\n✅ SUCCESS! The system is working correctly.")
        print("\nNow run: python quick_start_final.py")
    else:
        print("\n❌ Test failed. Check the error messages above.")