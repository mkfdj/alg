#!/usr/bin/env python3
"""
Simple runner for NCA Trading Bot that avoids import issues
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main runner function"""
    print("🧬 NCA Trading Bot - Kaggle TPU v5e-8 Runner")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("nca_trading_bot").exists():
        print("❌ Error: nca_trading_bot directory not found!")
        print("Make sure you're running this from the correct directory")
        return

    # Add current directory to Python path
    sys.path.insert(0, '.')

    # Set Kaggle credentials from environment variables or prompt user
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')

    if not kaggle_username or not kaggle_key:
        print("\n⚠️  Kaggle credentials not found!")
        print("Please set them using:")
        print("export KAGGLE_USERNAME='your_username'")
        print("export KAGGLE_KEY='your_api_key'")
        print("\nOr add them to your Kaggle notebook secrets.")
        print("Continuing with synthetic data...\n")

    # Import and run the main function
    try:
        from nca_trading_bot.main import main as bot_main

        # Run with kaggle mode
        sys.argv = ['run_nca_bot.py', '--mode', 'train', '--kaggle', '--nca-iterations', '500', '--ppo-iterations', '500']
        bot_main()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Falling back to direct execution...")

        # Fallback: run main.py directly
        cmd = [
            'python', 'nca_trading_bot/main.py',
            '--mode', 'train',
            '--kaggle',
            '--nca-iterations', '500',
            '--ppo-iterations', '500'
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

    except Exception as e:
        print(f"❌ Error running bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()