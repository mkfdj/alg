#!/usr/bin/env python3
"""
Simple runner script for coding datasets system
"""

import os
import sys

# Set credentials here
os.environ['KAGGLE_USERNAME'] = 'mautlej'
# os.environ['KAGGLE_KEY'] = 'your_kaggle_api_key_here'  # Uncomment and set your key

# Add the coding_datasets directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'coding_datasets'))

if __name__ == "__main__":
    from coding_datasets.main import main
    main()