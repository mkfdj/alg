#!/usr/bin/env python3
"""
Kaggle Test Runner for NCA Trading Bot

This script provides a simple way to run the integration tests in a Kaggle environment.
It handles the setup of the environment and runs the tests with appropriate configuration.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up the environment for testing."""
    logger.info("Setting up environment for testing")

    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))

    # Set environment variables for testing
    os.environ['NCA_DEVICE'] = 'cpu'  # Use CPU for testing
    os.environ['NCA_LOG_LEVEL'] = 'INFO'

    # Create necessary directories
    for dir_name in ['data', 'models', 'logs']:
        dir_path = current_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

    logger.info("Environment setup complete")


def run_tests():
    """Run the integration tests."""
    logger.info("Running integration tests")

    try:
        # Run the tests
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/test_integration.py', '-v'],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True
        )

        # Print the output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Return the result
        return result.returncode == 0

    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False


def main():
    """Main function."""
    logger.info("Starting Kaggle Test Runner for NCA Trading Bot")

    # Setup environment
    setup_environment()

    # Run tests
    success = run_tests()

    # Print result
    if success:
        logger.info("All tests passed successfully!")
        sys.exit(0)
    else:
        logger.error("Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()