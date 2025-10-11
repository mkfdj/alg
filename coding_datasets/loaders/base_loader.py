"""
Base loader interface for all dataset formats
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import os
import logging


class BaseLoader(ABC):
    """Abstract base class for dataset loaders"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load(self, file_path: str, **kwargs) -> Union[List[Dict], Dict]:
        """Load data from file"""
        pass

    @abstractmethod
    def get_prompts(self, data: Union[List[Dict], Dict], prompt_field: str = None) -> List[str]:
        """Extract prompts from loaded data"""
        pass

    def validate_file(self, file_path: str) -> bool:
        """Check if file exists and is readable"""
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return False
        if not os.access(file_path, os.R_OK):
            self.logger.error(f"File not readable: {file_path}")
            return False
        return True

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get basic metadata about the file"""
        if not self.validate_file(file_path):
            return {}

        stat = os.stat(file_path)
        return {
            "file_path": file_path,
            "file_size_bytes": stat.st_size,
            "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
            "last_modified": stat.st_mtime,
            "loader_type": self.__class__.__name__
        }