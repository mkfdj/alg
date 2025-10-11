"""
Configuration management system for coding datasets
"""

import json
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import os


class ConfigManager:
    """Manages configuration for the coding datasets system"""

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()

    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        current_dir = Path(__file__).parent
        return str(current_dir / "default_config.yaml")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not os.path.exists(self.config_path):
            self.logger.warning(f"Config file not found: {self.config_path}")
            return self._get_fallback_config()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    import yaml
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)

            self.logger.info(f"Loaded configuration from: {self.config_path}")
            return config or {}

        except Exception as e:
            self.logger.error(f"Error loading config file: {str(e)}")
            return self._get_fallback_config()

    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configuration"""
        return {
            "data": {
                "download_path": "./coding_datasets/data",
                "cache_enabled": True,
                "max_concurrent_downloads": 3
            },
            "preprocessing": {
                "normalize_whitespace": True,
                "remove_duplicates": True,
                "filter_by_length": True,
                "min_length": 10,
                "max_length": 10000
            },
            "training": {
                "format": "instruction",
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "random_seed": 42
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, config_path: Optional[str] = None) -> None:
        """Save configuration to file"""
        save_path = config_path or self.config_path

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    import yaml
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Configuration saved to: {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving config file: {str(e)}")
            raise

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        def deep_update(base: Dict, updates: Dict) -> Dict:
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_update(base[key], value)
                else:
                    base[key] = value
            return base

        self.config = deep_update(self.config, updates)

    def get_dataset_config(self, dataset_id: str) -> Dict[str, Any]:
        """Get configuration for a specific dataset"""
        dataset_configs = self.get('datasets', {})
        return dataset_configs.get(dataset_id, {})

    def set_dataset_config(self, dataset_id: str, config: Dict[str, Any]) -> None:
        """Set configuration for a specific dataset"""
        if 'datasets' not in self.config:
            self.config['datasets'] = {}

        self.config['datasets'][dataset_id] = config

    def get_active_datasets(self) -> list:
        """Get list of active datasets"""
        return self.get('active_datasets', [])

    def set_active_datasets(self, dataset_ids: list) -> None:
        """Set list of active datasets"""
        self.set('active_datasets', dataset_ids)

    def get_download_config(self) -> Dict[str, Any]:
        """Get download configuration"""
        return self.get('data', {})

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration"""
        return self.get('preprocessing', {})

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.get('training', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get('logging', {})

    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results"""
        issues = []

        # Check required sections
        required_sections = ['data', 'preprocessing', 'training']
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required section: {section}")

        # Check data configuration
        data_config = self.get('data', {})
        if 'download_path' not in data_config:
            issues.append("Missing download_path in data configuration")

        # Check training ratios
        training_config = self.get('training', {})
        ratios = ['train_ratio', 'val_ratio', 'test_ratio']
        total_ratio = sum(training_config.get(ratio, 0) for ratio in ratios)
        if abs(total_ratio - 1.0) > 0.01:
            issues.append(f"Training ratios must sum to 1.0, got {total_ratio}")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }

    def merge_from_file(self, config_path: str) -> None:
        """Merge configuration from another file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    import yaml
                    new_config = yaml.safe_load(f)
                else:
                    new_config = json.load(f)

            if new_config:
                self.update(new_config)
                self.logger.info(f"Merged configuration from: {config_path}")

        except Exception as e:
            self.logger.error(f"Error merging config file: {str(e)}")
            raise

    def export_config(self, export_path: str, sections: Optional[list] = None) -> None:
        """Export specific configuration sections to file"""
        if sections:
            export_data = {section: self.config.get(section) for section in sections}
        else:
            export_data = self.config

        try:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)

            with open(export_path, 'w', encoding='utf-8') as f:
                if export_path.endswith('.yaml') or export_path.endswith('.yml'):
                    import yaml
                    yaml.dump(export_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Configuration exported to: {export_path}")

        except Exception as e:
            self.logger.error(f"Error exporting config: {str(e)}")
            raise

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults"""
        self.config = self._get_fallback_config()
        self.logger.info("Configuration reset to defaults")

    def print_config(self, section: Optional[str] = None) -> None:
        """Print configuration (or specific section) in readable format"""
        import yaml

        if section:
            config_to_print = self.get(section, {})
            print(f"=== {section.upper()} CONFIGURATION ===")
        else:
            config_to_print = self.config
            print("=== FULL CONFIGURATION ===")

        print(yaml.dump(config_to_print, default_flow_style=False, indent=2))

    def __str__(self) -> str:
        """String representation of configuration"""
        return f"ConfigManager(config_path='{self.config_path}')"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"ConfigManager(config_path='{self.config_path}', sections={list(self.config.keys())})"