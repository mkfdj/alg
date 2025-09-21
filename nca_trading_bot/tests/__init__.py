"""
Test package for NCA Trading Bot.

This package contains unit tests for all modules in the NCA trading system.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up test configuration
os.environ.setdefault('NCA_CONFIG_PATH', 'test_config.yaml')
os.environ.setdefault('NCA_LOG_LEVEL', 'WARNING')

# Test constants
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
TEST_LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')

# Create test directories
for test_dir in [TEST_DATA_DIR, TEST_MODELS_DIR, TEST_LOGS_DIR]:
    os.makedirs(test_dir, exist_ok=True)