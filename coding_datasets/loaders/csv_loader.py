"""
CSV data loader
"""

import pandas as pd
from typing import List, Dict, Union, Any, Optional
from .base_loader import BaseLoader


class CSVLoader(BaseLoader):
    """Loader for CSV format datasets"""

    def load(self, file_path: str, **kwargs) -> Union[List[Dict], Dict]:
        """
        Load CSV file

        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pandas.read_csv

        Returns:
            List of dictionaries or pandas DataFrame
        """
        if not self.validate_file(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        try:
            # Default CSV reading options for coding datasets
            default_kwargs = {
                'encoding': 'utf-8',
                'low_memory': False
            }
            default_kwargs.update(kwargs)

            df = pd.read_csv(file_path, **default_kwargs)
            self.logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

            # Return as list of dictionaries for consistency
            return df.to_dict('records')

        except Exception as e:
            self.logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            raise

    def get_prompts(self, data: Union[List[Dict], Dict], prompt_field: str = None) -> List[str]:
        """
        Extract prompts from CSV data

        Args:
            data: Loaded CSV data
            prompt_field: Field name containing prompts (auto-detect if None)

        Returns:
            List of prompt strings
        """
        if not data:
            return []

        # Auto-detect prompt field if not specified
        if prompt_field is None:
            first_row = data[0] if isinstance(data, list) else data
            prompt_field = self._detect_prompt_field(first_row)

        if prompt_field is None:
            raise ValueError("Could not detect prompt field. Please specify prompt_field parameter.")

        prompts = []
        for row in data:
            if isinstance(row, dict) and prompt_field in row:
                prompt = row[prompt_field]
                if pd.notna(prompt) and str(prompt).strip():
                    prompts.append(str(prompt))
            elif isinstance(row, str):
                prompts.append(row)

        self.logger.info(f"Extracted {len(prompts)} prompts from field '{prompt_field}'")
        return prompts

    def _detect_prompt_field(self, sample_row: Dict) -> Optional[str]:
        """Auto-detect the field containing prompts"""
        if not isinstance(sample_row, dict):
            return None

        # Common field names for prompts in coding datasets
        prompt_fields = [
            'prompt', 'question', 'problem', 'instruction', 'text',
            'description', 'task', 'query', 'input', 'code_prompt',
            'problem_statement', 'challenge', 'exercise'
        ]

        # Check exact matches first
        for field in prompt_fields:
            if field in sample_row:
                return field

        # Check partial matches
        for field in sample_row.keys():
            field_lower = field.lower()
            for prompt_keyword in prompt_fields:
                if prompt_keyword in field_lower:
                    return field

        # If no match found, return first string field
        for field, value in sample_row.items():
            if isinstance(value, str) and len(str(value).strip()) > 10:
                return field

        return None

    def get_data_stats(self, data: Union[List[Dict], Dict]) -> Dict[str, Any]:
        """Get statistics about the loaded data"""
        if not data:
            return {"total_records": 0}

        stats = {
            "total_records": len(data),
            "columns": list(data[0].keys()) if isinstance(data, list) else list(data.keys()),
            "sample_record": data[0] if isinstance(data, list) else data
        }

        # Calculate text statistics for each column
        if isinstance(data, list) and data:
            for col in stats["columns"]:
                text_lengths = []
                for row in data[:100]:  # Sample first 100 rows for efficiency
                    if col in row and row[col] is not None:
                        text_lengths.append(len(str(row[col])))

                if text_lengths:
                    stats[f"{col}_stats"] = {
                        "avg_length": round(sum(text_lengths) / len(text_lengths), 2),
                        "min_length": min(text_lengths),
                        "max_length": max(text_lengths)
                    }

        return stats