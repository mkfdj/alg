"""
Dataset download scripts for TRM Coding Agent.

This module provides scripts to download and prepare all supported coding datasets
from Kaggle and other sources.
"""

import os
import sys
import json
import subprocess
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import List, Dict, Optional, Any
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# Dataset configurations
DATASETS = {
    "OpenAI HumanEval": {
        "kaggle_name": "thedevastator/handcrafted-dataset-for-code-generation-models",
        "local_name": "humaneval",
        "format": "csv",
        "files": ["test.csv"],
        "description": "164 programming problems with unit tests"
    },
    "OpenAI HumanEval JSONL": {
        "kaggle_name": "inoueu1/openai-human-eval",
        "local_name": "humaneval_jsonl",
        "format": "jsonl",
        "files": ["data/human_eval/HumanEval.jsonl"],
        "description": "Original HumanEval evaluation harness"
    },
    "OpenAI HumanEval Code Gen": {
        "kaggle_name": "thedevastator/openai-humaneval-code-gen",
        "local_name": "humaneval_codegen",
        "format": "csv",
        "files": ["data.csv"],
        "description": "HumanEval with code generation format"
    },
    "MBPP Python Problems": {
        "kaggle_name": "mpwolke/mbppjsonl",
        "local_name": "mbpp",
        "format": "jsonl",
        "files": ["mbpp.jsonl"],
        "description": "Mostly Basic Python Programming Problems"
    },
    "Mostly Basic Python Problems": {
        "kaggle_name": "buntyshah/mostly-basic-python-problems-dataset",
        "local_name": "basic_python",
        "format": "json",
        "files": ["mbpp.json"],
        "description": "Basic Python programming problems"
    },
    "CodeParrot 1M": {
        "kaggle_name": "heyytanay/codeparrot-1m",
        "local_name": "codeparrot_1m",
        "format": "lance",
        "files": ["_latest.manifest", "data/", "_versions/", "_transactions/"],
        "description": "1M tokenized Python files in Lance format"
    },
    "Python Code Instruction Dataset": {
        "kaggle_name": "thedevastator/python-code-instruction-dataset",
        "local_name": "python_instructions",
        "format": "csv",
        "files": ["train.csv"],
        "description": "Python code instruction dataset"
    },
    "Python Code Instructions 18K Alpaca": {
        "kaggle_name": "nikitakudriashov/python-code-instructions-18k-alpaca",
        "local_name": "alpaca_python_18k",
        "format": "json",
        "files": ["data.json"],
        "description": "18K Python code instructions in Alpaca format"
    },
    "Alpaca Language Instruction Training": {
        "kaggle_name": "thedevastator/alpaca-language-instruction-training",
        "local_name": "alpaca_training",
        "format": "json",
        "files": ["alpaca_data.json"],
        "description": "Alpaca instruction training data"
    },
    "Glaive Python Code QA Dataset": {
        "kaggle_name": "thedevastator/glaive-python-code-qa-dataset",
        "local_name": "glaive_python_qa",
        "format": "csv",
        "files": ["train.csv"],
        "description": "Python code question-answer pairs"
    },
    "Code Contests Dataset": {
        "kaggle_name": "lallucycle/code-contests-dataset",
        "local_name": "code_contests",
        "format": "csv",
        "files": ["code_contests.csv"],
        "description": "Competitive programming problems"
    },
    "LiveCodeBench": {
        "kaggle_name": "open-benchmarks/livecodebench",
        "local_name": "livecodebench",
        "format": "json",
        "files": ["livecodebench.json"],
        "description": "Live updating code benchmark"
    }
}


class DatasetDownloader:
    """Downloads and prepares coding datasets."""

    def __init__(self, base_dir: str = "datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def check_kaggle_cli(self) -> bool:
        """Check if Kaggle CLI is available and authenticated."""
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def download_kaggle_dataset(self, kaggle_name: str, local_dir: Path) -> bool:
        """Download dataset from Kaggle."""
        try:
            print(f"Downloading {kaggle_name}...")

            cmd = [
                "kaggle", "datasets", "download",
                "-d", kaggle_name,
                "--unzip",
                "-p", str(local_dir)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode == 0:
                print(f"Successfully downloaded {kaggle_name}")
                return True
            else:
                print(f"Failed to download {kaggle_name}: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"Download timeout for {kaggle_name}")
            return False
        except Exception as e:
            print(f"Error downloading {kaggle_name}: {e}")
            return False

    def download_url(self, url: str, local_path: Path) -> bool:
        """Download file from URL."""
        try:
            print(f"Downloading {url}...")

            urllib.request.urlretrieve(url, local_path)
            print(f"Successfully downloaded {url}")
            return True

        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract archive file."""
        try:
            print(f"Extracting {archive_path}...")

            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                print(f"Unsupported archive format: {archive_path.suffix}")
                return False

            print(f"Successfully extracted {archive_path}")
            return True

        except Exception as e:
            print(f"Error extracting {archive_path}: {e}")
            return False

    def download_dataset(self, dataset_name: str, force_redownload: bool = False) -> bool:
        """Download a specific dataset."""
        if dataset_name not in DATASETS:
            print(f"Unknown dataset: {dataset_name}")
            return False

        dataset_info = DATASETS[dataset_name]
        local_dir = self.base_dir / dataset_info['local_name']

        # Check if already downloaded
        if local_dir.exists() and not force_redownload:
            # Check if required files exist
            files_exist = any(
                (local_dir / file_name).exists()
                for file_name in dataset_info['files']
            )
            if files_exist:
                print(f"Dataset {dataset_name} already exists")
                return True

        # Create local directory
        local_dir.mkdir(exist_ok=True)

        # Download from Kaggle
        if 'kaggle_name' in dataset_info:
            success = self.download_kaggle_dataset(dataset_info['kaggle_name'], local_dir)
        else:
            print(f"No download method for {dataset_name}")
            return False

        if success:
            # Create metadata file
            metadata = {
                'name': dataset_name,
                'kaggle_name': dataset_info.get('kaggle_name', ''),
                'format': dataset_info['format'],
                'description': dataset_info['description'],
                'files': dataset_info['files'],
                'downloaded_at': str(Path.cwd()),
                'local_dir': str(local_dir)
            }

            metadata_path = local_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Dataset {dataset_name} prepared successfully")
            return True
        else:
            return False

    def download_all_datasets(self, selected_datasets: Optional[List[str]] = None) -> Dict[str, bool]:
        """Download multiple datasets."""
        if selected_datasets is None:
            selected_datasets = list(DATASETS.keys())

        results = {}

        for dataset_name in selected_datasets:
            if dataset_name in DATASETS:
                success = self.download_dataset(dataset_name)
                results[dataset_name] = success
            else:
                print(f"Unknown dataset: {dataset_name}")
                results[dataset_name] = False

        return results

    def list_downloaded_datasets(self) -> List[Dict[str, Any]]:
        """List all downloaded datasets."""
        downloaded = []

        for dataset_name, dataset_info in DATASETS.items():
            local_dir = self.base_dir / dataset_info['local_name']
            metadata_path = local_dir / 'metadata.json'

            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    downloaded.append(metadata)
                except Exception as e:
                    print(f"Error reading metadata for {dataset_name}: {e}")

        return downloaded

    def verify_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Verify dataset integrity."""
        if dataset_name not in DATASETS:
            return {'valid': False, 'error': 'Unknown dataset'}

        dataset_info = DATASETS[dataset_name]
        local_dir = self.base_dir / dataset_info['local_name']

        verification = {
            'dataset': dataset_name,
            'local_dir': str(local_dir),
            'exists': local_dir.exists(),
            'files_found': [],
            'files_missing': [],
            'valid': False
        }

        if not local_dir.exists():
            verification['error'] = 'Directory does not exist'
            return verification

        # Check for required files
        for file_pattern in dataset_info['files']:
            if file_pattern.endswith('/'):
                # Directory
                file_path = local_dir / file_pattern.rstrip('/')
                if file_path.exists() and file_path.is_dir():
                    verification['files_found'].append(file_pattern)
                else:
                    verification['files_missing'].append(file_pattern)
            else:
                # File
                file_path = local_dir / file_pattern
                if file_path.exists():
                    verification['files_found'].append(file_pattern)
                else:
                    verification['files_missing'].append(file_pattern)

        verification['valid'] = len(verification['files_missing']) == 0

        return verification

    def create_dataset_index(self) -> None:
        """Create an index file of all datasets."""
        index = {
            'base_dir': str(self.base_dir),
            'datasets': {},
            'created_at': str(Path.cwd())
        }

        for dataset_name in DATASETS:
            verification = self.verify_dataset(dataset_name)
            index['datasets'][dataset_name] = verification

        index_path = self.base_dir / 'dataset_index.json'
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)

        print(f"Dataset index created at {index_path}")

    def cleanup_downloads(self) -> None:
        """Clean up corrupted downloads."""
        print("Cleaning up corrupted downloads...")

        for dataset_name in DATASETS:
            local_dir = self.base_dir / DATASETS[dataset_name]['local_name']

            if local_dir.exists():
                verification = self.verify_dataset(dataset_name)

                if not verification['valid']:
                    print(f"Removing corrupted dataset: {dataset_name}")
                    import shutil
                    shutil.rmtree(local_dir)

        print("Cleanup completed")


def setup_kaggle_cli():
    """Setup Kaggle CLI."""
    print("Setting up Kaggle CLI...")

    # Check if already installed
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        print("Kaggle CLI is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Install Kaggle CLI
    try:
        print("Installing Kaggle CLI...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "kaggle"
        ], check=True)
        print("Kaggle CLI installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Kaggle CLI: {e}")
        return False


def authenticate_kaggle():
    """Authenticate with Kaggle."""
    print("Authenticating with Kaggle...")
    print("Please ensure you have a Kaggle API token file at ~/.kaggle/kaggle.json")
    print("You can download it from: https://www.kaggle.com/account")

    # Check if token exists
    kaggle_dir = Path.home() / ".kaggle"
    token_file = kaggle_dir / "kaggle.json"

    if token_file.exists():
        print("Kaggle token found")
        return True
    else:
        print("Kaggle token not found")
        print("Please download your API token and place it at ~/.kaggle/kaggle.json")
        return False


def main():
    """Main function for dataset downloading."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download datasets for TRM Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  OpenAI HumanEval              - 164 programming problems with unit tests
  OpenAI HumanEval JSONL         - Original HumanEval evaluation harness
  MBPP Python Problems           - Mostly Basic Python Programming Problems
  CodeParrot 1M                  - 1M tokenized Python files
  Python Code Instructions       - Python code instruction dataset
  Alpaca Python 18K              - 18K Python code instructions
  Glaive Python QA               - Python code question-answer pairs
  Code Contests                  - Competitive programming problems
  LiveCodeBench                  - Live updating code benchmark

Examples:
  python download_datasets.py --all
  python download_datasets.py --datasets "OpenAI HumanEval" "MBPP Python Problems"
  python download_datasets.py --list
  python download_datasets.py --verify "OpenAI HumanEval"
        """
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets"
    )
    parser.add_argument(
        "--verify",
        nargs="+",
        help="Verify specific datasets"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="datasets",
        help="Base directory for datasets"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up corrupted downloads"
    )
    parser.add_argument(
        "--setup-kaggle",
        action="store_true",
        help="Setup Kaggle CLI"
    )

    args = parser.parse_args()

    # Setup downloader
    downloader = DatasetDownloader(args.base_dir)

    # Setup Kaggle CLI if requested
    if args.setup_kaggle:
        setup_kaggle_cli()
        authenticate_kaggle()
        return

    # List available datasets
    if args.list:
        print("Available datasets:")
        for name, info in DATASETS.items():
            print(f"  {name}: {info['description']}")
        return

    # Verify datasets
    if args.verify:
        print("Verifying datasets:")
        for dataset_name in args.verify:
            verification = downloader.verify_dataset(dataset_name)
            status = "✓ Valid" if verification['valid'] else "✗ Invalid"
            print(f"  {dataset_name}: {status}")
            if not verification['valid']:
                print(f"    Missing files: {verification['files_missing']}")
        return

    # Clean up downloads
    if args.cleanup:
        downloader.cleanup_downloads()
        return

    # Check Kaggle CLI
    if not downloader.check_kaggle_cli():
        print("Kaggle CLI not found. Please run with --setup-kaggle first.")
        return

    # Download datasets
    if args.all:
        print("Downloading all datasets...")
        datasets_to_download = list(DATASETS.keys())
    elif args.datasets:
        datasets_to_download = args.datasets
    else:
        print("No datasets specified. Use --all or --datasets <dataset names>")
        return

    # Download datasets
    results = downloader.download_all_datasets(datasets_to_download)

    # Print results
    print("\nDownload Results:")
    successful = 0
    for dataset_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {dataset_name}: {status}")
        if success:
            successful += 1

    print(f"\nSuccessfully downloaded {successful}/{len(results)} datasets")

    # Create index
    downloader.create_dataset_index()

    if successful == len(results):
        print("All datasets downloaded successfully!")
    else:
        print("Some datasets failed to download. Check the output above for details.")


if __name__ == "__main__":
    main()