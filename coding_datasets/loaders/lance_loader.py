"""
Lance data loader
"""

import os
from typing import List, Dict, Union, Any, Optional
from base_loader import BaseLoader

try:
    import lance
    import pyarrow as pa
    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False


class LanceLoader(BaseLoader):
    """Loader for Lance format datasets"""

    def __init__(self, data_path: str):
        super().__init__(data_path)
        if not LANCE_AVAILABLE:
            raise ImportError("Lance is not installed. Install with: pip install lance")

    def load(self, file_path: str, **kwargs) -> pa.Table:
        """
        Load Lance dataset

        Args:
            file_path: Path to Lance dataset directory
            **kwargs: Additional arguments for lance.dataset()

        Returns:
            PyArrow Table
        """
        if not self.validate_file(file_path):
            raise FileNotFoundError(f"Lance dataset not found: {file_path}")

        try:
            # Load Lance dataset
            dataset = lance.dataset(file_path)

            # Get limit if provided
            limit = kwargs.get('limit', None)
            if limit:
                table = dataset.to_table(limit=limit)
            else:
                table = dataset.to_table()

            self.logger.info(f"Loaded Lance dataset with {len(table)} rows")
            return table

        except Exception as e:
            self.logger.error(f"Error loading Lance dataset {file_path}: {str(e)}")
            raise

    def get_prompts(self, table: pa.Table, prompt_field: str = None) -> List[str]:
        """
        Extract prompts from Lance data

        Args:
            table: PyArrow Table from Lance dataset
            prompt_field: Field name containing prompts (auto-detect if None)

        Returns:
            List of prompt strings
        """
        if table.num_rows == 0:
            return []

        # Convert to pandas for easier processing
        df = table.to_pandas()

        # Auto-detect prompt field if not specified
        if prompt_field is None:
            prompt_field = self._detect_prompt_field(df)

        if prompt_field is None:
            raise ValueError("Could not detect prompt field. Please specify prompt_field parameter.")

        if prompt_field not in df.columns:
            raise ValueError(f"Prompt field '{prompt_field}' not found in dataset")

        # Extract prompts
        prompts = []
        for value in df[prompt_field]:
            if value is not None and str(value).strip():
                prompts.append(str(value))

        self.logger.info(f"Extracted {len(prompts)} prompts from field '{prompt_field}'")
        return prompts

    def _detect_prompt_field(self, df) -> Optional[str]:
        """Auto-detect the field containing prompts in DataFrame"""
        # Common field names for prompts in coding datasets
        prompt_fields = [
            'prompt', 'question', 'problem', 'instruction', 'text',
            'description', 'task', 'query', 'input', 'code_prompt',
            'problem_statement', 'challenge', 'exercise', 'content'
        ]

        # Check exact matches first
        for field in prompt_fields:
            if field in df.columns:
                return field

        # Check partial matches
        for col in df.columns:
            col_lower = col.lower()
            for prompt_keyword in prompt_fields:
                if prompt_keyword in col_lower:
                    return col

        # If no match found, return first string column with substantial content
        for col in df.columns:
            if df[col].dtype == 'object':  # Object dtype usually means strings
                sample_values = df[col].dropna().head(5)
                if any(len(str(val)) > 20 for val in sample_values):
                    return col

        return None

    def get_data_stats(self, table: pa.Table) -> Dict[str, Any]:
        """Get statistics about the loaded Lance data"""
        if table.num_rows == 0:
            return {"total_records": 0}

        df = table.to_pandas()

        stats = {
            "total_records": len(df),
            "columns": list(df.columns),
            "schema": str(table.schema)
        }

        # Calculate text statistics for each column
        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    text_lengths = non_null_values.astype(str).str.len()
                    stats[f"{col}_stats"] = {
                        "count": len(non_null_values),
                        "avg_length": round(text_lengths.mean(), 2),
                        "min_length": int(text_lengths.min()),
                        "max_length": int(text_lengths.max())
                    }

        return stats

    def sample_data(self, table: pa.Table, n: int = 10) -> pa.Table:
        """Sample n random rows from the dataset"""
        import random
        if table.num_rows <= n:
            return table

        indices = random.sample(range(table.num_rows), n)
        return table.take(indices)

    def filter_by_length(self, table: pa.Table, prompt_field: str,
                         min_length: int = 10, max_length: int = 10000) -> pa.Table:
        """Filter data by prompt length"""
        df = table.to_pandas()

        if prompt_field not in df.columns:
            raise ValueError(f"Prompt field '{prompt_field}' not found")

        # Filter by length
        mask = df[prompt_field].astype(str).str.len().between(min_length, max_length)
        filtered_df = df[mask]

        # Convert back to PyArrow table
        return pa.Table.from_pandas(filtered_df)

    def validate_file(self, file_path: str) -> bool:
        """Check if Lance dataset directory exists and is valid"""
        if not os.path.exists(file_path):
            self.logger.error(f"Lance dataset directory not found: {file_path}")
            return False

        if not os.path.isdir(file_path):
            self.logger.error(f"Path is not a directory: {file_path}")
            return False

        # Check for required Lance files
        required_files = ['_versions', 'data']
        for req_file in required_files:
            if not os.path.exists(os.path.join(file_path, req_file)):
                self.logger.error(f"Required Lance file not found: {req_file}")
                return False

        return True