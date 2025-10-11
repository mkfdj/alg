"""
Data preprocessing utilities for coding datasets
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union
import json
from pathlib import Path


class DataPreprocessor:
    """Main preprocessor for coding datasets"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def preprocess_prompts(self, prompts: List[str], config: Dict[str, Any] = None) -> List[str]:
        """
        Apply preprocessing pipeline to prompts

        Args:
            prompts: List of prompt strings
            config: Preprocessing configuration

        Returns:
            Preprocessed prompts
        """
        if config is None:
            config = self._get_default_config()

        processed = prompts.copy()

        # Apply preprocessing steps
        if config.get('normalize_whitespace', True):
            processed = self._normalize_whitespace(processed)

        if config.get('remove_duplicates', True):
            processed = self._remove_duplicates(processed)

        if config.get('filter_by_length', True):
            min_len = config.get('min_length', 10)
            max_len = config.get('max_length', 10000)
            processed = self._filter_by_length(processed, min_len, max_len)

        if config.get('remove_code_blocks', False):
            processed = self._remove_code_blocks(processed)

        if config.get('fix_encoding_issues', True):
            processed = self._fix_encoding_issues(processed)

        if config.get('normalize_punctuation', True):
            processed = self._normalize_punctuation(processed)

        self.logger.info(f"Preprocessed {len(prompts)} -> {len(processed)} prompts")
        return processed

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration"""
        return {
            'normalize_whitespace': True,
            'remove_duplicates': True,
            'filter_by_length': True,
            'min_length': 10,
            'max_length': 10000,
            'remove_code_blocks': False,
            'fix_encoding_issues': True,
            'normalize_punctuation': True
        }

    def _normalize_whitespace(self, prompts: List[str]) -> List[str]:
        """Normalize whitespace in prompts"""
        normalized = []
        for prompt in prompts:
            # Replace multiple spaces with single space
            normalized_prompt = re.sub(r'\s+', ' ', prompt)
            # Fix leading/trailing whitespace
            normalized_prompt = normalized_prompt.strip()
            normalized.append(normalized_prompt)
        return normalized

    def _remove_duplicates(self, prompts: List[str]) -> List[str]:
        """Remove duplicate prompts while preserving order"""
        seen = set()
        unique_prompts = []
        for prompt in prompts:
            if prompt not in seen:
                seen.add(prompt)
                unique_prompts.append(prompt)
        return unique_prompts

    def _filter_by_length(self, prompts: List[str], min_length: int, max_length: int) -> List[str]:
        """Filter prompts by length"""
        filtered = []
        for prompt in prompts:
            if min_length <= len(prompt) <= max_length:
                filtered.append(prompt)
        return filtered

    def _remove_code_blocks(self, prompts: List[str]) -> List[str]:
        """Remove code blocks from prompts"""
        cleaned = []
        code_block_pattern = r'```[\s\S]*?```'
        for prompt in prompts:
            cleaned_prompt = re.sub(code_block_pattern, '', prompt)
            cleaned.append(cleaned_prompt)
        return cleaned

    def _fix_encoding_issues(self, prompts: List[str]) -> List[str]:
        """Fix common encoding issues"""
        fixed = []
        for prompt in prompts:
            # Fix common encoding issues
            fixed_prompt = prompt.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
            fixed_prompt = fixed_prompt.replace('â€"', '"').replace('â€"', '"')
            fixed_prompt = fixed_prompt.replace('â€¦', '...').replace('â€"', '"')
            fixed.append(fixed_prompt)
        return fixed

    def _normalize_punctuation(self, prompts: List[str]) -> List[str]:
        """Normalize punctuation in prompts"""
        normalized = []
        for prompt in prompts:
            # Fix multiple punctuation
            normalized_prompt = re.sub(r'([.!?])\1+', r'\1', prompt)
            # Fix spacing around punctuation
            normalized_prompt = re.sub(r'\s+([.!?])', r'\1', normalized_prompt)
            normalized.append(normalized_prompt)
        return normalized

    def create_training_examples(self, prompts: List[str], responses: List[str] = None,
                                config: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Create training examples from prompts and optional responses

        Args:
            prompts: List of prompts
            responses: List of responses (optional)
            config: Configuration for example creation

        Returns:
            List of training examples
        """
        if config is None:
            config = self._get_default_training_config()

        examples = []

        for i, prompt in enumerate(prompts):
            example = {
                'prompt': prompt,
                'instruction': prompt,
                'input': '',
                'output': responses[i] if responses and i < len(responses) else '',
                'id': f"example_{i}"
            }

            # Add metadata if requested
            if config.get('include_metadata', True):
                example['metadata'] = {
                    'length': len(prompt),
                    'word_count': len(prompt.split()),
                    'source': 'coding_dataset'
                }

            examples.append(example)

        self.logger.info(f"Created {len(examples)} training examples")
        return examples

    def _get_default_training_config(self) -> Dict[str, Any]:
        """Get default training configuration"""
        return {
            'include_metadata': True,
            'add_prefix': False,
            'prefix_text': ''
        }

    def split_dataset(self, data: List[Any], train_ratio: float = 0.8,
                     val_ratio: float = 0.1, test_ratio: float = 0.1,
                     random_seed: int = 42) -> Dict[str, List[Any]]:
        """
        Split dataset into train/validation/test sets

        Args:
            data: List of data items
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with train/val/test splits
        """
        import random

        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        # Shuffle data
        random.seed(random_seed)
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)

        # Calculate split indices
        n = len(shuffled_data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split data
        splits = {
            'train': shuffled_data[:train_end],
            'val': shuffled_data[train_end:val_end],
            'test': shuffled_data[val_end:]
        }

        self.logger.info(f"Split dataset: train={len(splits['train'])}, "
                        f"val={len(splits['val'])}, test={len(splits['test'])}")

        return splits

    def export_to_jsonl(self, data: List[Dict], file_path: str):
        """Export data to JSONL format"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            self.logger.info(f"Exported {len(data)} items to {file_path}")
        except Exception as e:
            self.logger.error(f"Error exporting to JSONL: {str(e)}")
            raise

    def export_to_json(self, data: List[Dict], file_path: str):
        """Export data to JSON format"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Exported {len(data)} items to {file_path}")
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {str(e)}")
            raise

    def analyze_data_quality(self, prompts: List[str]) -> Dict[str, Any]:
        """Analyze data quality metrics"""
        if not prompts:
            return {}

        lengths = [len(p) for p in prompts]
        word_counts = [len(p.split()) for p in prompts]

        analysis = {
            'total_count': len(prompts),
            'length_stats': {
                'min': min(lengths),
                'max': max(lengths),
                'mean': round(sum(lengths) / len(lengths), 2),
                'median': sorted(lengths)[len(lengths) // 2]
            },
            'word_count_stats': {
                'min': min(word_counts),
                'max': max(word_counts),
                'mean': round(sum(word_counts) / len(word_counts), 2),
                'median': sorted(word_counts)[len(word_counts) // 2]
            }
        }

        # Check for potential issues
        issues = []
        empty_count = sum(1 for p in prompts if not p.strip())
        if empty_count > 0:
            issues.append(f"{empty_count} empty prompts")

        very_long_count = sum(1 for p in prompts if len(p) > 5000)
        if very_long_count > 0:
            issues.append(f"{very_long_count} very long prompts (>5000 chars)")

        duplicate_count = len(prompts) - len(set(prompts))
        if duplicate_count > 0:
            issues.append(f"{duplicate_count} duplicate prompts")

        analysis['issues'] = issues
        analysis['quality_score'] = max(0, 100 - len(issues) * 10)

        return analysis