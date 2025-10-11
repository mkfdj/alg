"""
Main entry point for the coding datasets system
"""

import argparse
import logging
import os
import sys
from pathlib import Path

try:
    from dataset_manager import DatasetManager
    from configs import get_default_config
except ImportError:
    # Handle relative import issues
    sys.path.append(str(Path(__file__).parent))
    from dataset_manager import DatasetManager
    from configs import get_default_config


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


def cmd_list_datasets(manager: DatasetManager, args):
    """List all available datasets"""
    datasets = manager.list_datasets()

    print(f"\n{'Dataset ID':<25} {'Name':<40} {'Size':<10} {'Format':<15} {'Downloaded'}")
    print("-" * 110)

    for dataset_id, info in datasets.items():
        downloaded = "✓" if info['downloaded'] else "✗"
        print(f"{dataset_id:<25} {info['name'][:40]:<40} {info['size']:<10} {info['format']:<15} {downloaded}")

    print(f"\nTotal: {len(datasets)} datasets")


def cmd_download_datasets(manager: DatasetManager, args):
    """Download datasets"""
    if args.all:
        print("Downloading all datasets...")
        results = manager.download_all_datasets(args.force)
        successful = sum(1 for v in results.values() if v is not None)
        print(f"Downloaded {successful}/{len(results)} datasets successfully")
    else:
        if not args.datasets:
            print("Error: Please specify dataset IDs or use --all")
            return

        for dataset_id in args.datasets:
            try:
                print(f"Downloading {dataset_id}...")
                file_path = manager.download_dataset(dataset_id, args.force)
                print(f"✓ Downloaded to: {file_path}")
            except Exception as e:
                print(f"✗ Failed to download {dataset_id}: {str(e)}")


def cmd_info(manager: DatasetManager, args):
    """Show dataset information"""
    if args.dataset:
        try:
            info = manager.get_dataset_info(args.dataset)
            print(f"\nDataset: {info.name}")
            print(f"Description: {info.description}")
            print(f"Size: {info.size}")
            print(f"Format: {info.format}")
            print(f"Kaggle Link: {info.kaggle_link}")
            print(f"Load Function: {info.load_function}")
            print(f"\nStructure Example:\n{info.structure_example}")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        stats = manager.get_statistics()
        print("\n=== System Statistics ===")
        print(f"Total Datasets: {stats['registry']['total_datasets']}")
        print(f"Total Size: {stats['registry']['size_info']['total_size_gb']:.2f} GB")
        print(f"Downloaded Datasets: {stats['downloads']['total_datasets']}")
        print(f"Download Size: {stats['downloads']['total_size_gb']:.2f} GB")


def cmd_process(manager: DatasetManager, args):
    """Process datasets and create training data"""
    try:
        print("Creating training data...")
        training_data = manager.create_training_data(
            dataset_ids=args.datasets,
            format_type=args.format,
            split_data=args.split
        )

        print(f"✓ Created training data with {training_data['total_examples']} examples")
        print(f"Format: {training_data['format']}")

        if args.split and 'splits' in training_data:
            print("Data splits:")
            for split_name, split_data in training_data['splits'].items():
                print(f"  {split_name}: {len(split_data)} examples")

        # Export if requested
        if args.output:
            manager.export_data(training_data, args.output, args.export_format)
            print(f"✓ Exported to: {args.output}")

    except Exception as e:
        print(f"Error: {str(e)}")


def cmd_validate(manager: DatasetManager, args):
    """Validate datasets"""
    try:
        datasets_to_validate = args.datasets or manager.list_datasets().keys()

        for dataset_id in datasets_to_validate:
            print(f"\nValidating {dataset_id}...")
            results = manager.validate_dataset(dataset_id)
            score = results['quality_score']
            status = "✓ VALID" if results['is_valid'] else "✗ INVALID"

            print(f"Status: {status} (Score: {score}/100)")

            if results['errors']:
                print("Errors:")
                for error in results['errors']:
                    print(f"  - {error}")

            if results['warnings']:
                print("Warnings:")
                for warning in results['warnings']:
                    print(f"  - {warning}")

    except Exception as e:
        print(f"Error: {str(e)}")


def cmd_config(manager: DatasetManager, args):
    """Configuration management"""
    if args.show:
        manager.config.print_config(args.section)
    elif args.reset:
        manager.config.reset_to_defaults()
        print("Configuration reset to defaults")
    elif args.validate:
        validation = manager.config.validate_config()
        if validation['is_valid']:
            print("✓ Configuration is valid")
        else:
            print("✗ Configuration has issues:")
            for issue in validation['issues']:
                print(f"  - {issue}")
    else:
        print("Use --show, --reset, or --validate with config command")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Coding Datasets Manager - Manage 18+ coding datasets from Kaggle"
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--download-path", help="Path for dataset downloads")
    parser.add_argument("--kaggle-username", help="Kaggle username (or set KAGGLE_USERNAME env var)")
    parser.add_argument("--kaggle-key", help="Kaggle API key (or set KAGGLE_KEY env var)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List datasets command
    list_parser = subparsers.add_parser("list", help="List all available datasets")
    list_parser.set_defaults(func=cmd_list_datasets)

    # Download command
    download_parser = subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument("datasets", nargs="*", help="Dataset IDs to download")
    download_parser.add_argument("--all", action="store_true", help="Download all datasets")
    download_parser.add_argument("--force", action="store_true", help="Force re-download")
    download_parser.set_defaults(func=cmd_download_datasets)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show dataset information")
    info_parser.add_argument("--dataset", help="Specific dataset ID")
    info_parser.set_defaults(func=cmd_info)

    # Process command
    process_parser = subparsers.add_parser("process", help="Process datasets into training data")
    process_parser.add_argument("datasets", nargs="*", help="Dataset IDs to process")
    process_parser.add_argument("--format", default="instruction",
                               choices=["instruction", "alpaca", "chat", "code_completion"],
                               help="Training format")
    process_parser.add_argument("--split", action="store_true", default=True, help="Split into train/val/test")
    process_parser.add_argument("--no-split", dest="split", action="store_false", help="Don't split data")
    process_parser.add_argument("--output", help="Output file path")
    process_parser.add_argument("--export-format", default="jsonl", choices=["json", "jsonl"],
                               help="Export format")
    process_parser.set_defaults(func=cmd_process)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset quality")
    validate_parser.add_argument("datasets", nargs="*", help="Dataset IDs to validate")
    validate_parser.set_defaults(func=cmd_validate)

    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_group = config_parser.add_mutually_exclusive_group()
    config_group.add_argument("--show", action="store_true", help="Show configuration")
    config_group.add_argument("--section", help="Show specific configuration section")
    config_group.add_argument("--reset", action="store_true", help="Reset to defaults")
    config_group.add_argument("--validate", action="store_true", help="Validate configuration")
    config_parser.set_defaults(func=cmd_config)

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Print banner
    print_banner()

    # Set Kaggle credentials if provided via command line
    if args.kaggle_username:
        os.environ['KAGGLE_USERNAME'] = args.kaggle_username
    if args.kaggle_key:
        os.environ['KAGGLE_KEY'] = args.kaggle_key

    # Initialize manager
    try:
        manager = DatasetManager(
            config_path=args.config,
            download_path=args.download_path
        )
    except Exception as e:
        print(f"Error initializing manager: {str(e)}")
        sys.exit(1)

    # Execute command
    if args.command:
        try:
            args.func(manager, args)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()