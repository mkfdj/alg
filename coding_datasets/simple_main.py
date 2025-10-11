#!/usr/bin/env python3
"""
Simple working version of coding datasets manager
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add current directory to path for all imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
for subdir in ['loaders', 'configs', 'utils']:
    subdir_path = current_dir / subdir
    if subdir_path.exists():
        sys.path.insert(0, str(subdir_path))

def setup_logging(level: str = "INFO"):
    """Setup basic logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    Coding Datasets Manager                    ║
    ║                                                              ║
    ║  Manage 18+ coding datasets from Kaggle for ML training     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def test_imports():
    """Test if all imports work"""
    try:
        print("🧪 Testing imports...")

        # Test individual imports
        import registry
        print("  ✓ registry")

        import downloader
        print("  ✓ downloader")

        import config_manager
        print("  ✓ config_manager")

        import default_configs
        print("  ✓ default_configs")

        import preprocessor
        print("  ✓ preprocessor")

        import base_loader
        print("  ✓ base_loader")

        import csv_loader
        print("  ✓ csv_loader")

        # Test the main manager
        import dataset_manager
        from dataset_manager import DatasetManager
        print("  ✓ DatasetManager")

        return True

    except Exception as e:
        print(f"  ❌ Import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def cmd_list(manager, args):
    """List all available datasets"""
    try:
        datasets = manager.list_datasets()

        print(f"\n{'Dataset ID':<25} {'Name':<40} {'Size':<10} {'Format':<15} {'Downloaded'}")
        print("-" * 110)

        for dataset_id, info in datasets.items():
            downloaded = "✓" if info['downloaded'] else "✗"
            print(f"{dataset_id:<25} {info['name'][:40]:<40} {info['size']:<10} {info['format']:<15} {downloaded}")

        print(f"\nTotal: {len(datasets)} datasets")

    except Exception as e:
        print(f"Error listing datasets: {str(e)}")

def cmd_download(manager, args):
    """Download datasets"""
    if not args.datasets and not args.all:
        print("Please specify dataset IDs or use --all")
        return

    try:
        if args.all:
            print("Downloading all datasets...")
            datasets = list(manager.registry.list_datasets())
        else:
            datasets = args.datasets

        for dataset_id in datasets:
            print(f"Downloading {dataset_id}...")
            try:
                file_path = manager.download_dataset(dataset_id)
                print(f"✓ Downloaded to: {file_path}")
            except Exception as e:
                print(f"✗ Failed to download {dataset_id}: {str(e)}")

    except Exception as e:
        print(f"Error in download command: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Coding Datasets Manager - Manage 18+ coding datasets from Kaggle"
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List datasets command
    list_parser = subparsers.add_parser("list", help="List all available datasets")
    list_parser.set_defaults(func=cmd_list)

    # Download command
    download_parser = subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument("datasets", nargs="*", help="Dataset IDs to download")
    download_parser.add_argument("--all", action="store_true", help="Download all datasets")
    download_parser.set_defaults(func=cmd_download)

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    print_banner()

    # Check Kaggle credentials
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')

    if username:
        print(f"🔐 Kaggle username: {username}")
        if key:
            print("🔐 Kaggle API key: set")
        else:
            print("⚠️  Kaggle API key not set - set KAGGLE_KEY environment variable")
    else:
        print("⚠️  Kaggle credentials not set")
        print("💡 Set with: export KAGGLE_USERNAME='mautlej'")
        print("         export KAGGLE_KEY='your_api_key'")

    # Test imports first
    if not test_imports():
        print("\n❌ Import tests failed. Please fix import issues.")
        return

    # Initialize manager
    try:
        from dataset_manager import DatasetManager
        manager = DatasetManager()
        print("✅ DatasetManager initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize DatasetManager: {str(e)}")
        return

    # Execute command
    if args.command:
        try:
            args.func(manager, args)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()