"""
Main dataset manager class that orchestrates all dataset operations
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json
import os

from registry import DatasetRegistry, DatasetInfo
from downloader import DatasetDownloader
from loaders import CSVLoader, JSONLoader, JSONLLoader, LanceLoader, ParquetLoader
from utils import DataPreprocessor, CodeCleaner, PromptFormatter, DataValidator
from configs import ConfigManager, get_default_config


class DatasetManager:
    """Main manager for coding datasets operations"""

    def __init__(self, config_path: Optional[str] = None, download_path: Optional[str] = None):
        """
        Initialize the DatasetManager

        Args:
            config_path: Path to configuration file
            download_path: Path for dataset downloads
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize configuration
        self.config = ConfigManager(config_path)
        if download_path:
            self.config.set('data.download_path', download_path)

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.registry = DatasetRegistry()
        self.downloader = DatasetDownloader(
            download_path=self.config.get('data.download_path'),
            kaggle_json_path=self.config.get('data.kaggle_api_path'),
            kaggle_username=self.config.get('data.kaggle_username') or os.environ.get('KAGGLE_USERNAME'),
            kaggle_key=self.config.get('data.kaggle_key') or os.environ.get('KAGGLE_KEY')
        )

        # Initialize loaders
        self.loaders = {
            'csv': CSVLoader(self.config.get('data.download_path')),
            'json': JSONLoader(self.config.get('data.download_path')),
            'jsonl': JSONLLoader(self.config.get('data.download_path')),
            'lance': LanceLoader(self.config.get('data.download_path')),
            'parquet': ParquetLoader(self.config.get('data.download_path'))
        }

        # Initialize utilities
        self.preprocessor = DataPreprocessor()
        self.code_cleaner = CodeCleaner()
        self.prompt_formatter = PromptFormatter()
        self.validator = DataValidator()

        # Cache for loaded datasets
        self._dataset_cache = {}

        self.logger.info("DatasetManager initialized successfully")

    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.get_logging_config()
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create logger
        logger = logging.getLogger('coding_datasets')
        logger.setLevel(log_level)

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler
        if log_config.get('console_logging', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(log_format)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File handler
        if log_config.get('file_logging', False):
            log_file = log_config.get('log_file', './coding_datasets/logs/datasets.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(log_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    def download_dataset(self, dataset_id: str, force_redownload: bool = False) -> str:
        """
        Download a specific dataset

        Args:
            dataset_id: Dataset ID from registry
            force_redownload: Force re-download even if file exists

        Returns:
            Path to downloaded dataset
        """
        try:
            self.logger.info(f"Starting download for dataset: {dataset_id}")
            file_path = self.downloader.download_dataset(dataset_id, force_redownload)
            self.logger.info(f"Successfully downloaded dataset: {dataset_id}")
            return file_path
        except Exception as e:
            self.logger.error(f"Failed to download dataset {dataset_id}: {str(e)}")
            raise

    def download_all_datasets(self, force_redownload: bool = False) -> Dict[str, str]:
        """
        Download all datasets in the registry

        Args:
            force_redownload: Force re-download even if files exist

        Returns:
            Dictionary mapping dataset IDs to file paths
        """
        try:
            self.logger.info("Starting download of all datasets")
            results = self.downloader.download_all_datasets(force_redownload)
            successful = sum(1 for v in results.values() if v is not None)
            self.logger.info(f"Downloaded {successful}/{len(results)} datasets successfully")
            return results
        except Exception as e:
            self.logger.error(f"Failed to download all datasets: {str(e)}")
            raise

    def load_dataset(self, dataset_id: str, use_cache: bool = True) -> Union[List[Dict], Dict]:
        """
        Load a dataset from file

        Args:
            dataset_id: Dataset ID from registry
            use_cache: Use cached data if available

        Returns:
            Loaded dataset data
        """
        if use_cache and dataset_id in self._dataset_cache:
            self.logger.info(f"Loading {dataset_id} from cache")
            return self._dataset_cache[dataset_id]

        try:
            # Get dataset info
            dataset_info = self.registry.get_dataset_info(dataset_id)

            # Get downloaded file path
            downloaded_info = self.downloader.list_downloaded_datasets()
            if dataset_id not in downloaded_info:
                raise FileNotFoundError(f"Dataset {dataset_id} not downloaded")

            # Find the data file
            dataset_dir = Path(downloaded_info[dataset_id]['directory'])
            data_file = self._find_data_file(dataset_dir, dataset_info)

            if not data_file:
                raise FileNotFoundError(f"Data file not found for dataset {dataset_id}")

            # Load data using appropriate loader
            loader = self._get_loader(dataset_info.format)
            data = loader.load(str(data_file))

            # Cache the data
            if use_cache:
                self._dataset_cache[dataset_id] = data

            self.logger.info(f"Successfully loaded dataset: {dataset_id}")
            return data

        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_id}: {str(e)}")
            raise

    def _find_data_file(self, dataset_dir: Path, dataset_info: DatasetInfo) -> Optional[Path]:
        """Find the main data file in dataset directory"""
        # Try exact match first
        exact_file = dataset_dir / dataset_info.local_filename
        if exact_file.exists():
            return exact_file

        # Search for files with matching extension
        extensions = {
            'CSV': ['.csv'],
            'JSON': ['.json'],
            'JSONL': ['.jsonl', '.jsonlines'],
            'Lance/Parquet': ['.lance', '.parquet'],
            'Parquet': ['.parquet']
        }

        target_extensions = extensions.get(dataset_info.format, ['.csv', '.json', '.jsonl', '.parquet'])

        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in target_extensions:
                return file_path

        return None

    def _get_loader(self, format_type: str):
        """Get appropriate loader for format type"""
        format_mapping = {
            'CSV': 'csv',
            'JSON': 'json',
            'JSONL': 'jsonl',
            'Lance/Parquet': 'lance',
            'Parquet': 'parquet'
        }

        loader_type = format_mapping.get(format_type, 'csv')
        return self.loaders[loader_type]

    def extract_prompts(self, dataset_id: str, prompt_field: Optional[str] = None,
                       use_cache: bool = True) -> List[str]:
        """
        Extract prompts from a dataset

        Args:
            dataset_id: Dataset ID from registry
            prompt_field: Field containing prompts (auto-detect if None)
            use_cache: Use cached data if available

        Returns:
            List of prompt strings
        """
        try:
            # Load dataset
            data = self.load_dataset(dataset_id, use_cache)

            # Get dataset info
            dataset_info = self.registry.get_dataset_info(dataset_id)

            # Get loader
            loader = self._get_loader(dataset_info.format)

            # Extract prompts
            prompts = loader.get_prompts(data, prompt_field)

            self.logger.info(f"Extracted {len(prompts)} prompts from {dataset_id}")
            return prompts

        except Exception as e:
            self.logger.error(f"Failed to extract prompts from {dataset_id}: {str(e)}")
            raise

    def preprocess_dataset(self, dataset_id: str, prompts: List[str] = None,
                          config: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Preprocess a dataset

        Args:
            dataset_id: Dataset ID from registry
            prompts: List of prompts (extract if None)
            config: Preprocessing configuration

        Returns:
            Preprocessed prompts
        """
        try:
            if prompts is None:
                prompts = self.extract_prompts(dataset_id)

            # Get preprocessing config
            if config is None:
                config = self.config.get_preprocessing_config()

            # Apply preprocessing
            processed_prompts = self.preprocessor.preprocess_prompts(prompts, config)

            self.logger.info(f"Preprocessed {len(prompts)} -> {len(processed_prompts)} prompts")
            return processed_prompts

        except Exception as e:
            self.logger.error(f"Failed to preprocess dataset {dataset_id}: {str(e)}")
            raise

    def validate_dataset(self, dataset_id: str, prompts: List[str] = None,
                        responses: List[str] = None) -> Dict[str, Any]:
        """
        Validate dataset quality

        Args:
            dataset_id: Dataset ID from registry
            prompts: List of prompts (extract if None)
            responses: List of responses (optional)

        Returns:
            Validation results
        """
        try:
            if prompts is None:
                prompts = self.extract_prompts(dataset_id)

            validation_config = self.config.get('validation', {})
            strict_mode = validation_config.get('strict_mode', False)

            # Validate dataset
            results = self.validator.validate_dataset_quality(
                prompts, responses, strict_mode
            )

            self.logger.info(f"Validation completed for {dataset_id}: "
                           f"Score={results['quality_score']}/100")

            return results

        except Exception as e:
            self.logger.error(f"Failed to validate dataset {dataset_id}: {str(e)}")
            raise

    def create_training_data(self, dataset_ids: List[str] = None,
                            format_type: str = "instruction",
                            split_data: bool = True) -> Dict[str, Any]:
        """
        Create training data from specified datasets

        Args:
            dataset_ids: List of dataset IDs (use config default if None)
            format_type: Training format type
            split_data: Whether to split into train/val/test

        Returns:
            Training data dictionary
        """
        try:
            if dataset_ids is None:
                dataset_ids = self.config.get_active_datasets()
                if not dataset_ids:
                    dataset_ids = self.config.get('datasets.enabled', [])

            all_examples = []
            dataset_stats = {}

            for dataset_id in dataset_ids:
                try:
                    # Extract and preprocess prompts
                    prompts = self.extract_prompts(dataset_id)
                    prompts = self.preprocess_dataset(dataset_id, prompts)

                    # Create examples
                    examples = self.prompt_formatter.create_training_examples(prompts)

                    # Add dataset metadata
                    for example in examples:
                        example['dataset_id'] = dataset_id

                    all_examples.extend(examples)
                    dataset_stats[dataset_id] = len(examples)

                    self.logger.info(f"Added {len(examples)} examples from {dataset_id}")

                except Exception as e:
                    self.logger.warning(f"Skipping dataset {dataset_id}: {str(e)}")
                    continue

            # Format examples
            from .utils.prompt_formatter import PromptFormat
            format_enum = PromptFormat(format_type)
            formatted_examples = self.prompt_formatter.format_prompts(
                [ex['prompt'] for ex in all_examples],
                [ex.get('response', '') for ex in all_examples],
                format_enum
            )

            # Add metadata
            for i, example in enumerate(formatted_examples):
                example.update(all_examples[i])

            training_data = {
                'examples': formatted_examples,
                'dataset_stats': dataset_stats,
                'total_examples': len(formatted_examples),
                'format': format_type
            }

            # Split data if requested
            if split_data:
                training_config = self.config.get_training_config()
                splits = self.preprocessor.split_dataset(
                    formatted_examples,
                    training_config.get('train_ratio', 0.8),
                    training_config.get('val_ratio', 0.1),
                    training_config.get('test_ratio', 0.1),
                    training_config.get('random_seed', 42)
                )
                training_data['splits'] = splits

            self.logger.info(f"Created training data with {len(formatted_examples)} examples")
            return training_data

        except Exception as e:
            self.logger.error(f"Failed to create training data: {str(e)}")
            raise

    def export_data(self, data: Dict[str, Any], output_path: str,
                   format_type: str = "jsonl") -> None:
        """
        Export training data to file

        Args:
            data: Training data dictionary
            output_path: Output file path
            format_type: Export format
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if format_type == "jsonl":
                if 'splits' in data:
                    # Export splits separately
                    for split_name, split_data in data['splits'].items():
                        split_path = output_path.replace('.jsonl', f'_{split_name}.jsonl')
                        self.preprocessor.export_to_jsonl(split_data, split_path)
                else:
                    self.preprocessor.export_to_jsonl(data['examples'], output_path)

            elif format_type == "json":
                self.preprocessor.export_to_json(data, output_path)

            else:
                raise ValueError(f"Unsupported export format: {format_type}")

            self.logger.info(f"Data exported to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export data: {str(e)}")
            raise

    def get_dataset_info(self, dataset_id: str = None) -> Union[DatasetInfo, Dict[str, DatasetInfo]]:
        """
        Get information about datasets

        Args:
            dataset_id: Specific dataset ID (return all if None)

        Returns:
            Dataset information
        """
        if dataset_id:
            return self.registry.get_dataset_info(dataset_id)
        else:
            return {ds_id: self.registry.get_dataset_info(ds_id)
                   for ds_id in self.registry.list_datasets()}

    def list_datasets(self) -> Dict[str, Dict[str, Any]]:
        """List all datasets with their status"""
        datasets = {}
        downloaded = self.downloader.list_downloaded_datasets()

        for dataset_id in self.registry.list_datasets():
            info = self.registry.get_dataset_info(dataset_id)
            datasets[dataset_id] = {
                'name': info.name,
                'size': info.size,
                'format': info.format,
                'downloaded': dataset_id in downloaded,
                'download_path': downloaded[dataset_id]['directory'] if dataset_id in downloaded else None
            }

        return datasets

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the datasets"""
        stats = {
            'registry': {
                'total_datasets': len(self.registry.list_datasets()),
                'format_distribution': self.registry.get_datasets_by_format('CSV'),
                'size_info': self.registry.get_total_size_info()
            },
            'downloads': self.downloader.get_download_stats(),
            'config': {
                'active_datasets': len(self.config.get_active_datasets()),
                'config_path': self.config.config_path
            }
        }

        return stats

    def cleanup(self, dataset_ids: List[str] = None, clear_cache: bool = True) -> None:
        """
        Clean up datasets and cache

        Args:
            dataset_ids: List of dataset IDs to clean up (all if None)
            clear_cache: Clear in-memory cache
        """
        try:
            if dataset_ids is None:
                dataset_ids = self.registry.list_datasets()

            # Clean up downloaded datasets
            self.downloader.cleanup_downloads(dataset_ids)

            # Clear cache
            if clear_cache:
                self._dataset_cache.clear()

            self.logger.info(f"Cleanup completed for {len(dataset_ids)} datasets")

        except Exception as e:
            self.logger.error(f"Failed to cleanup: {str(e)}")
            raise

    def __str__(self) -> str:
        """String representation"""
        return f"DatasetManager(datasets={len(self.registry.list_datasets())})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"DatasetManager(datasets={len(self.registry.list_datasets())}, "
                f"download_path='{self.config.get('data.download_path')}', "
                f"config_path='{self.config.config_path}')")