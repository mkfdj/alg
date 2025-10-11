#!/usr/bin/env python3
"""
Test imports directly without relative imports
"""

import sys
import os
from pathlib import Path

# Add the coding_datasets directory to the path
coding_datasets_path = Path(__file__).parent / "coding_datasets"
sys.path.insert(0, str(coding_datasets_path))

def test_imports():
    """Test all imports work correctly"""
    print("🧪 Testing imports...")

    try:
        # Test main components
        print("  Importing registry...")
        import registry
        print("  ✓ registry")

        print("  Importing downloader...")
        import downloader
        print("  ✓ downloader")

        print("  Importing dataset_manager...")
        import dataset_manager
        print("  ✓ dataset_manager")

        print("  Importing configs...")
        import configs
        print("  ✓ configs")

        print("  Importing loaders...")
        import loaders
        print("  ✓ loaders")

        print("  Importing utils...")
        import utils
        print("  ✓ utils")

        # Test creating DatasetManager
        print("  Creating DatasetManager...")
        from dataset_manager import DatasetManager
        manager = DatasetManager()
        print("  ✓ DatasetManager created")

        # Test listing datasets
        print("  Listing datasets...")
        datasets = manager.list_datasets()
        print(f"  ✓ Listed {len(datasets)} datasets")

        return True

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Import Fixes")
    print("=" * 40)

    success = test_imports()

    if success:
        print("\n✅ All imports working!")
        print("\n💡 You can now run:")
        print("  python test_imports.py")
        print("  cd coding_datasets && python main.py list")
    else:
        print("\n❌ Import fixes failed")