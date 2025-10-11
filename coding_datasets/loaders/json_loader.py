"""
JSON data loader
"""

import json
from typing import List, Dict, Union, Any, Optional
from .base_loader import BaseLoader


class JSONLoader(BaseLoader):
    """Loader for JSON format datasets"""

    def load(self, file_path: str, **kwargs) -> Union[List[Dict], Dict]:
        """
        Load JSON file

        Args:
            file_path: Path to JSON file
            **kwargs: Additional arguments (encoding, etc.)

        Returns:
            JSON data as dictionary or list
        """
        if not self.validate_file(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        try:
            encoding = kwargs.get('encoding', 'utf-8')

            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)

            self.logger.info(f"Loaded JSON from {file_path}")
            return data

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in file {file_path}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading JSON file {file_path}: {str(e)}")
            raise

    def get_prompts(self, data: Union[List[Dict], Dict], prompt_field: str = None,
                   data_path: str = None) -> List[str]:
        """
        Extract prompts from JSON data

        Args:
            data: Loaded JSON data
            prompt_field: Field/path containing prompts (auto-detect if None)
            data_path: Dot-separated path to data within JSON (e.g., "data", "problems")

        Returns:
            List of prompt strings
        """
        # Navigate to specific data path if provided
        if data_path:
            data = self._navigate_path(data, data_path)

        if not data:
            return []

        prompts = []

        if isinstance(data, list):
            # Handle list of items
            first_item = data[0] if data else {}

            if prompt_field is None:
                prompt_field = self._detect_prompt_field(first_item)

            if prompt_field:
                for item in data:
                    prompt = self._extract_field_value(item, prompt_field)
                    if prompt and str(prompt).strip():
                        prompts.append(str(prompt))
            else:
                # If no field found, try to extract from string items
                for item in data:
                    if isinstance(item, str) and len(item.strip()) > 10:
                        prompts.append(item.strip())

        elif isinstance(data, dict):
            # Handle single dictionary
            if prompt_field is None:
                prompt_field = self._detect_prompt_field(data)

            if prompt_field:
                prompt = self._extract_field_value(data, prompt_field)
                if prompt and str(prompt).strip():
                    prompts.append(str(prompt))
            else:
                # Look for any string values that might be prompts
                for value in data.values():
                    if isinstance(value, str) and len(value.strip()) > 50:
                        prompts.append(value.strip())

        self.logger.info(f"Extracted {len(prompts)} prompts")
        return prompts

    def _navigate_path(self, data: Union[Dict, List], path: str) -> Union[Dict, List]:
        """Navigate through nested JSON structure using dot-separated path"""
        current = data
        for key in path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                self.logger.warning(f"Path '{path}' not found in JSON data")
                return data
        return current

    def _extract_field_value(self, item: Any, field_path: str) -> Any:
        """Extract value from item using field path (supports nested paths)"""
        if not isinstance(item, dict):
            return item

        current = item
        for key in field_path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _detect_prompt_field(self, sample_item: Any) -> Optional[str]:
        """Auto-detect the field containing prompts"""
        if not isinstance(sample_item, dict):
            return None

        # Common field names for prompts in coding datasets
        prompt_fields = [
            'prompt', 'question', 'problem', 'instruction', 'text',
            'description', 'task', 'query', 'input', 'code_prompt',
            'problem_statement', 'challenge', 'exercise', 'content'
        ]

        # Check exact matches first
        for field in prompt_fields:
            if field in sample_item:
                return field

        # Check nested paths for common patterns
        nested_paths = [
            'data.instruction', 'data.prompt', 'items.instruction', 'items.prompt',
            'problems.text', 'problems.prompt', 'examples.input'
        ]

        for path in nested_paths:
            if self._extract_field_value(sample_item, path):
                return path

        # Check partial matches
        for field in sample_item.keys():
            field_lower = field.lower()
            for prompt_keyword in prompt_fields:
                if prompt_keyword in field_lower:
                    return field

        # If no match found, return first string field
        for field, value in sample_item.items():
            if isinstance(value, str) and len(str(value).strip()) > 10:
                return field

        return None

    def get_data_stats(self, data: Union[List[Dict], Dict]) -> Dict[str, Any]:
        """Get statistics about the loaded JSON data"""
        stats = {}

        if isinstance(data, list):
            stats["total_records"] = len(data)
            if data:
                stats["sample_record"] = data[0]
                if isinstance(data[0], dict):
                    stats["fields"] = list(data[0].keys())
        elif isinstance(data, dict):
            stats["total_records"] = 1
            stats["fields"] = list(data.keys())
            stats["sample_record"] = data

        # Calculate text statistics for string fields
        if isinstance(data, list) and data and isinstance(data[0], dict):
            for field in stats.get("fields", []):
                text_lengths = []
                sample_size = min(100, len(data))

                for item in data[:sample_size]:
                    if field in item and item[field] is not None:
                        text_lengths.append(len(str(item[field])))

                if text_lengths:
                    stats[f"{field}_stats"] = {
                        "avg_length": round(sum(text_lengths) / len(text_lengths), 2),
                        "min_length": min(text_lengths),
                        "max_length": max(text_lengths)
                    }

        return stats