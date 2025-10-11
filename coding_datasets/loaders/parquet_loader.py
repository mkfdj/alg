"""
Parquet data loader
"""

import os
from typing import List, Dict, Union, Any, Optional
from .base_loader import BaseLoader

try:
    import pyarrow.parquet as pq
    import pandas as pd
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


class ParquetLoader(BaseLoader):
    """Loader for Parquet format datasets"""

    def __init__(self, data_path: str):
        super().__init__(data_path)
        if not PARQUET_AVAILABLE:
            raise ImportError("PyArrow is not installed. Install with: pip install pyarrow")

    def load(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load Parquet file

        Args:
            file_path: Path to Parquet file
            **kwargs: Additional arguments for pandas.read_parquet

        Returns:
            pandas DataFrame
        """
        if not self.validate_file(file_path):
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        try:
            # Default reading options
            default_kwargs = {}
            default_kwargs.update(kwargs)

            df = pd.read_parquet(file_path, **default_kwargs)
            self.logger.info(f"Loaded Parquet with {len(df)} rows and {len(df.columns)} columns")

            return df

        except Exception as e:
            self.logger.error(f"Error loading Parquet file {file_path}: {str(e)}")
            raise

    def get_prompts(self, df: pd.DataFrame, prompt_field: str = None) -> List[str]:
        """
        Extract prompts from Parquet data

        Args:
            df: pandas DataFrame from Parquet file
            prompt_field: Field name containing prompts (auto-detect if None)

        Returns:
            List of prompt strings
        """
        if df.empty:
            return []

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

    def _detect_prompt_field(self, df: pd.DataFrame) -> Optional[str]:
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

    def get_data_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the loaded Parquet data"""
        if df.empty:
            return {"total_records": 0}

        stats = {
            "total_records": len(df),
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
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

    def sample_data(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Sample n random rows from the DataFrame"""
        if len(df) <= n:
            return df
        return df.sample(n=n, random_state=42)

    def filter_by_length(self, df: pd.DataFrame, prompt_field: str,
                         min_length: int = 10, max_length: int = 10000) -> pd.DataFrame:
        """Filter data by prompt length"""
        if prompt_field not in df.columns:
            raise ValueError(f"Prompt field '{prompt_field}' not found")

        # Filter by length
        mask = df[prompt_field].astype(str).str.len().between(min_length, max_length)
        return df[mask]

    def get_column_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed information about DataFrame columns"""
        column_info = {}

        for col in df.columns:
            series = df[col]
            info = {
                "dtype": str(series.dtype),
                "non_null_count": series.count(),
                "null_count": series.isnull().sum(),
                "unique_count": series.nunique()
            }

            if series.dtype == 'object':
                # String column statistics
                non_null_series = series.dropna().astype(str)
                if len(non_null_series) > 0:
                    info.update({
                        "avg_length": round(non_null_series.str.len().mean(), 2),
                        "min_length": int(non_null_series.str.len().min()),
                        "max_length": int(non_null_series.str.len().max()),
                        "sample_values": non_null_series.head(3).tolist()
                    })
            elif series.dtype in ['int64', 'float64']:
                # Numeric column statistics
                info.update({
                    "min": series.min(),
                    "max": series.max(),
                    "mean": round(series.mean(), 2),
                    "std": round(series.std(), 2)
                })

            column_info[col] = info

        return column_info