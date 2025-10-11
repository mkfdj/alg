"""
JSONL data loader
"""

import json
from typing import List, Dict, Union, Any, Optional
from base_loader import BaseLoader


class JSONLLoader(BaseLoader):
    """Loader for JSONL (JSON Lines) format datasets"""

    def load(self, file_path: str, **kwargs) -> List[Dict]:
        """
        Load JSONL file

        Args:
            file_path: Path to JSONL file
            **kwargs: Additional arguments (encoding, limit, etc.)

        Returns:
            List of dictionaries
        """
        if not self.validate_file(file_path):
            raise FileNotFoundError(f"JSONL file not found: {file_path}")

        try:
            encoding = kwargs.get('encoding', 'utf-8')
            limit = kwargs.get('limit', None)

            data = []
            line_count = 0

            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            data.append(item)
                            line_count += 1

                            if limit and line_count >= limit:
                                break

                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Skipping invalid JSON line {line_count + 1}: {str(e)}")
                            continue

            self.logger.info(f"Loaded {len(data)} valid JSON lines from {file_path}")
            return data

        except Exception as e:
            self.logger.error(f"Error loading JSONL file {file_path}: {str(e)}")
            raise

    def get_prompts(self, data: List[Dict], prompt_field: str = None) -> List[str]:
        """
        Extract prompts from JSONL data

        Args:
            data: Loaded JSONL data (list of dictionaries)
            prompt_field: Field name containing prompts (auto-detect if None)

        Returns:
            List of prompt strings
        """
        if not data:
            return []

        # Auto-detect prompt field if not specified
        if prompt_field is None:
            prompt_field = self._detect_prompt_field(data[0])

        if prompt_field is None:
            raise ValueError("Could not detect prompt field. Please specify prompt_field parameter.")

        prompts = []
        for item in data:
            if isinstance(item, dict) and prompt_field in item:
                prompt = item[prompt_field]
                if prompt is not None and str(prompt).strip():
                    prompts.append(str(prompt))

        self.logger.info(f"Extracted {len(prompts)} prompts from field '{prompt_field}'")
        return prompts

    def _detect_prompt_field(self, sample_item: Dict) -> Optional[str]:
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

    def get_data_stats(self, data: List[Dict]) -> Dict[str, Any]:
        """Get statistics about the loaded JSONL data"""
        if not data:
            return {"total_records": 0}

        stats = {
            "total_records": len(data),
            "fields": list(data[0].keys()) if isinstance(data[0], dict) else [],
            "sample_record": data[0]
        }

        # Calculate text statistics for each field
        if isinstance(data[0], dict):
            for field in stats["fields"]:
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

    def sample_data(self, data: List[Dict], n: int = 10) -> List[Dict]:
        """Sample n random items from the data"""
        import random
        if len(data) <= n:
            return data
        return random.sample(data, n)

    def filter_by_length(self, data: List[Dict], prompt_field: str,
                         min_length: int = 10, max_length: int = 10000) -> List[Dict]:
        """Filter data by prompt length"""
        filtered_data = []
        for item in data:
            if isinstance(item, dict) and prompt_field in item:
                prompt = str(item[prompt_field])
                if min_length <= len(prompt) <= max_length:
                    filtered_data.append(item)
        return filtered_data