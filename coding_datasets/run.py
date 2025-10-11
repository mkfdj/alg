#!/usr/bin/env python3
"""
Standalone runner for coding datasets system
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import main functionality
from main import main

if __name__ == "__main__":
    # Set Kaggle credentials if environment variables exist
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')

    if kaggle_username:
        print(f"🔐 Using Kaggle username: {kaggle_username}")
    if kaggle_key:
        print("🔐 Kaggle API key is set")
    elif kaggle_username:
        print("⚠️  Kaggle username set but no API key - set KAGGLE_KEY environment variable")

    # Run main application
    main()