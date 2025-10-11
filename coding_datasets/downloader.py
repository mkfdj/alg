"""
Kaggle dataset downloader for coding datasets
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

from .registry import DatasetRegistry, DatasetInfo


class DatasetDownloader:
    """Downloader for Kaggle datasets"""

    def __init__(self, download_path: str = None, kaggle_json_path: str = None,
                 kaggle_username: str = None, kaggle_key: str = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry = DatasetRegistry()

        # Set default download path
        if download_path is None:
            self.download_path = Path.cwd() / "coding_datasets" / "data"
        else:
            self.download_path = Path(download_path)

        self.download_path.mkdir(parents=True, exist_ok=True)

        # Initialize Kaggle API
        self.api = None
        if KAGGLE_AVAILABLE:
            try:
                self.api = KaggleApi()

                # Try different authentication methods in order of preference
                authenticated = self._authenticate_kaggle(
                    kaggle_username, kaggle_key, kaggle_json_path
                )

                if authenticated:
                    self.logger.info("Kaggle API authenticated successfully")
                else:
                    self.logger.warning("Kaggle API authentication failed")
                    self.logger.info("Please set up your Kaggle API credentials")
                    self.api = None

            except Exception as e:
                self.logger.warning(f"Kaggle API authentication failed: {str(e)}")
                self.logger.info("Please set up your Kaggle API credentials")
                self.api = None
        else:
            self.logger.error("Kaggle library not installed. Install with: pip install kaggle")

    def _authenticate_kaggle(self, username: str = None, key: str = None,
                            json_path: str = None) -> bool:
        """
        Try multiple authentication methods for Kaggle API

        Args:
            username: Kaggle username (for environment variable auth)
            key: Kaggle API key (for environment variable auth)
            json_path: Path to kaggle.json file

        Returns:
            True if authentication successful
        """
        # Method 1: Use provided credentials directly (highest priority)
        if username and key:
            try:
                os.environ['KAGGLE_USERNAME'] = username
                os.environ['KAGGLE_KEY'] = key
                self.api.authenticate()
                self.logger.info("Authenticated using provided credentials")
                return True
            except Exception as e:
                self.logger.debug(f"Failed to authenticate with provided credentials: {e}")

        # Method 2: Check environment variables
        env_username = username or os.environ.get('KAGGLE_USERNAME')
        env_key = key or os.environ.get('KAGGLE_KEY')

        if env_username and env_key:
            try:
                os.environ['KAGGLE_USERNAME'] = env_username
                os.environ['KAGGLE_KEY'] = env_key
                self.api.authenticate()
                self.logger.info("Authenticated using environment variables")
                return True
            except Exception as e:
                self.logger.debug(f"Failed to authenticate with environment variables: {e}")

        # Method 3: Use kaggle.json file
        if json_path:
            try:
                config_dir = Path(json_path).parent
                os.environ['KAGGLE_CONFIG_DIR'] = str(config_dir)
                self.api.authenticate()
                self.logger.info(f"Authenticated using kaggle.json at {json_path}")
                return True
            except Exception as e:
                self.logger.debug(f"Failed to authenticate with specified json file: {e}")

        # Method 4: Try default kaggle.json locations
        default_locations = [
            Path.home() / '.kaggle' / 'kaggle.json',
            Path.cwd() / 'kaggle.json',
            Path('kaggle.json')
        ]

        for location in default_locations:
            if location.exists():
                try:
                    config_dir = location.parent
                    os.environ['KAGGLE_CONFIG_DIR'] = str(config_dir)
                    self.api.authenticate()
                    self.logger.info(f"Authenticated using kaggle.json at {location}")
                    return True
                except Exception as e:
                    self.logger.debug(f"Failed to authenticate with {location}: {e}")
                    continue

        # Method 5: Try with default KAGGLE_CONFIG_DIR if set
        if 'KAGGLE_CONFIG_DIR' in os.environ:
            try:
                self.api.authenticate()
                self.logger.info("Authenticated using KAGGLE_CONFIG_DIR")
                return True
            except Exception as e:
                self.logger.debug(f"Failed to authenticate with KAGGLE_CONFIG_DIR: {e}")

        self.logger.error("All authentication methods failed")
        return False

    def download_dataset(self, dataset_id: str, force_redownload: bool = False) -> str:
        """
        Download a specific dataset

        Args:
            dataset_id: Dataset ID from registry
            force_redownload: Force re-download even if file exists

        Returns:
            Path to downloaded dataset
        """
        if self.api is None:
            raise RuntimeError("Kaggle API not available. Please install kaggle library and set up credentials.")

        # Get dataset info
        dataset_info = self.registry.get_dataset_info(dataset_id)

        # Create dataset-specific directory
        dataset_dir = self.download_path / dataset_id
        dataset_dir.mkdir(exist_ok=True)

        # Check if dataset already exists
        expected_file = dataset_dir / dataset_info.local_filename
        if expected_file.exists() and not force_redownload:
            self.logger.info(f"Dataset already exists: {expected_file}")
            return str(expected_file)

        try:
            self.logger.info(f"Downloading dataset: {dataset_info.name}")

            # Download from Kaggle
            self.api.dataset_download_files(
                dataset_info.kaggle_dataset,
                path=str(dataset_dir),
                unzip=True,
                quiet=False
            )

            # Find the actual downloaded file (might be different name)
            downloaded_file = self._find_downloaded_file(dataset_dir, dataset_info)

            if downloaded_file is None:
                # Try to find any file that matches the expected format
                downloaded_file = self._find_any_matching_file(dataset_dir, dataset_info.format)

            if downloaded_file is None:
                raise FileNotFoundError(f"Could not find downloaded file for {dataset_info.name}")

            self.logger.info(f"Successfully downloaded to: {downloaded_file}")
            return str(downloaded_file)

        except Exception as e:
            self.logger.error(f"Error downloading dataset {dataset_id}: {str(e)}")
            raise

    def download_all_datasets(self, force_redownload: bool = False) -> Dict[str, str]:
        """
        Download all datasets in the registry

        Args:
            force_redownload: Force re-download even if files exist

        Returns:
            Dictionary mapping dataset IDs to file paths
        """
        results = {}
        dataset_ids = self.registry.list_datasets()

        self.logger.info(f"Starting download of {len(dataset_ids)} datasets")

        for i, dataset_id in enumerate(dataset_ids, 1):
            try:
                self.logger.info(f"Downloading dataset {i}/{len(dataset_ids)}: {dataset_id}")
                file_path = self.download_dataset(dataset_id, force_redownload)
                results[dataset_id] = file_path

            except Exception as e:
                self.logger.error(f"Failed to download {dataset_id}: {str(e)}")
                results[dataset_id] = None

        successful = sum(1 for v in results.values() if v is not None)
        self.logger.info(f"Downloaded {successful}/{len(dataset_ids)} datasets successfully")

        return results

    def _find_downloaded_file(self, dataset_dir: Path, dataset_info: DatasetInfo) -> Optional[Path]:
        """Find the expected downloaded file"""
        # Try exact match first
        exact_file = dataset_dir / dataset_info.local_filename
        if exact_file.exists():
            return exact_file

        # Try case-insensitive search
        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file() and file_path.name.lower() == dataset_info.local_filename.lower():
                return file_path

        return None

    def _find_any_matching_file(self, dataset_dir: Path, format_type: str) -> Optional[Path]:
        """Find any file matching the expected format"""
        format_extensions = {
            'CSV': ['.csv'],
            'JSON': ['.json'],
            'JSONL': ['.jsonl', '.jsonlines'],
            'Lance/Parquet': ['.lance', '.parquet'],
            'Parquet': ['.parquet']
        }

        extensions = format_extensions.get(format_type, ['.csv', '.json', '.jsonl', '.parquet'])

        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                return file_path

        return None

    def list_downloaded_datasets(self) -> Dict[str, Dict[str, Any]]:
        """List all downloaded datasets with their info"""
        downloaded = {}

        for dataset_id in self.registry.list_datasets():
            dataset_dir = self.download_path / dataset_id
            dataset_info = self.registry.get_dataset_info(dataset_id)

            if dataset_dir.exists():
                files = list(dataset_dir.rglob("*"))
                files = [f for f in files if f.is_file()]

                downloaded[dataset_id] = {
                    "name": dataset_info.name,
                    "directory": str(dataset_dir),
                    "files": [str(f) for f in files],
                    "file_count": len(files),
                    "size_mb": round(sum(f.stat().st_size for f in files) / (1024 * 1024), 2),
                    "format": dataset_info.format,
                    "info": dataset_info
                }

        return downloaded

    def get_download_stats(self) -> Dict[str, Any]:
        """Get statistics about downloaded datasets"""
        downloaded = self.list_downloaded_datasets()

        total_datasets = len(downloaded)
        total_files = sum(info["file_count"] for info in downloaded.values())
        total_size_mb = sum(info["size_mb"] for info in downloaded.values())

        format_counts = {}
        for info in downloaded.values():
            fmt = info["format"]
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

        return {
            "total_datasets": total_datasets,
            "total_files": total_files,
            "total_size_mb": round(total_size_mb, 2),
            "total_size_gb": round(total_size_mb / 1024, 2),
            "format_distribution": format_counts,
            "download_path": str(self.download_path)
        }

    def cleanup_downloads(self, dataset_ids: List[str] = None):
        """Clean up downloaded datasets"""
        if dataset_ids is None:
            dataset_ids = self.registry.list_datasets()

        for dataset_id in dataset_ids:
            dataset_dir = self.download_path / dataset_id
            if dataset_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(dataset_dir)
                    self.logger.info(f"Cleaned up dataset: {dataset_id}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up {dataset_id}: {str(e)}")

    def verify_download_integrity(self, dataset_id: str) -> bool:
        """Verify if a downloaded dataset is complete and readable"""
        downloaded_info = self.list_downloaded_datasets()

        if dataset_id not in downloaded_info:
            return False

        dataset_info = downloaded_info[dataset_id]

        # Check if at least one file exists
        if dataset_info["file_count"] == 0:
            return False

        # Check file sizes (should not be empty)
        for file_path_str in dataset_info["files"]:
            file_path = Path(file_path_str)
            if file_path.stat().st_size == 0:
                self.logger.warning(f"Empty file found: {file_path}")
                return False

        return True