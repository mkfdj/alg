"""
Kaggle TPU test runner for NCA Trading Bot.

This script runs tests with proper TPU v5e-8 initialization using JAX,
addressing the "Device or resource busy" error by correctly initializing
JAX for TPU usage in Kaggle environment.
"""

import os
import sys
import unittest
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_tpu_tests():
    """Run tests with proper TPU initialization."""
    
    logger.info("Starting TPU test runner for NCA Trading Bot")
    
    # Step 1: Initialize TPU
    try:
        from kaggle_tpu_initializer import initialize_tpu_for_kaggle
        
        logger.info("Initializing TPU v5e-8...")
        init_results = initialize_tpu_for_kaggle()
        
        if not init_results['initialization_success']:
            logger.error(f"TPU initialization failed: {init_results['error']}")
            return False
        
        logger.info("✅ TPU initialization successful!")
        logger.info(f"Device info: {init_results['device_info']}")
        
    except Exception as e:
        logger.error(f"Failed to import or run TPU initializer: {e}")
        return False
    
    # Step 2: Configure JAX environment
    try:
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh, PartitionSpec as P
        from jax.experimental import mesh_utils
        
        # Set up mesh for TPU v5e-8
        devices = jax.devices()
        logger.info(f"Found {len(devices)} JAX devices")
        
        if len(devices) == 8:
            # TPU v5e-8 configuration
            device_mesh = mesh_utils.create_device_mesh((1, 8))
            mesh = Mesh(device_mesh, axis_names=('data', 'model'))
            logger.info("Set up 1x8 mesh for TPU v5e-8")
        else:
            # Fallback for other configurations
            device_mesh = mesh_utils.create_device_mesh((len(devices),))
            mesh = Mesh(device_mesh, axis_names=('data',))
            logger.info(f"Set up {len(devices)}x1 mesh for {len(devices)} devices")
        
        # Configure JAX for TPU
        jax.config.update('jax_default_matmul_precision', 'bfloat16')
        jax.config.update('jax_enable_x64', False)
        
    except Exception as e:
        logger.error(f"Failed to configure JAX: {e}")
        return False
    
    # Step 3: Run TPU-specific tests
    try:
        # Set up Python path for relative imports
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Add parent directory to path if not already there
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Now import the test modules
        from nca_trading_bot.tests.test_tpu_integration import TestTPUDetection, TestTPUConfiguration
        from nca_trading_bot.tests.test_tpu_integration import TestXLACompilation, TestTPUOptimizations
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add TPU-specific test cases
        suite.addTests(loader.loadTestsFromTestCase(TestTPUDetection))
        suite.addTests(loader.loadTestsFromTestCase(TestTPUConfiguration))
        suite.addTests(loader.loadTestsFromTestCase(TestXLACompilation))
        suite.addTests(loader.loadTestsFromTestCase(TestTPUOptimizations))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Check results
        success = result.wasSuccessful()
        if success:
            logger.info("✅ All TPU tests passed!")
        else:
            logger.error(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
            
            # Log failures
            for test, traceback in result.failures:
                logger.error(f"FAILURE: {test}")
                logger.error(traceback)
            
            # Log errors
            for test, traceback in result.errors:
                logger.error(f"ERROR: {test}")
                logger.error(traceback)
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to run TPU tests: {e}")
        return False


def run_basic_functionality_tests():
    """Run basic functionality tests that don't require TPU."""
    
    logger.info("Running basic functionality tests...")
    
    try:
        # Set up Python path for relative imports
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Add parent directory to path if not already there
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Import basic test modules
        from nca_trading_bot.tests.test_config import TestConfigManager
        from nca_trading_bot.tests.test_adaptivity import TestAdaptiveNCA
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add basic test cases
        suite.addTests(loader.loadTestsFromTestCase(TestConfigManager))
        suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveNCA))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Check results
        success = result.wasSuccessful()
        if success:
            logger.info("✅ All basic functionality tests passed!")
        else:
            logger.error(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to run basic functionality tests: {e}")
        return False


def run_jax_model_tests():
    """Run JAX model tests if JAX NCA model is available."""
    
    logger.info("Running JAX model tests...")
    
    try:
        # Check if JAX NCA model exists
        if not Path("jax_nca_model.py").exists():
            logger.warning("JAX NCA model not found, skipping JAX model tests")
            return True
        
        # Set up Python path for relative imports
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Add parent directory to path if not already there
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Import JAX model test
        from nca_trading_bot.tests.test_jax_nca_model import TestJAXNCAModel
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add JAX model test cases
        suite.addTests(loader.loadTestsFromTestCase(TestJAXNCAModel))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Check results
        success = result.wasSuccessful()
        if success:
            logger.info("✅ All JAX model tests passed!")
        else:
            logger.error(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to run JAX model tests: {e}")
        return False


def main():
    """Main test runner function."""
    
    logger.info("=" * 60)
    logger.info("NCA Trading Bot TPU Test Runner for Kaggle")
    logger.info("=" * 60)
    
    # Change to project directory
    os.chdir(project_root)
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Run tests in order
    test_results = {}
    
    # 1. Basic functionality tests (should always pass)
    test_results['basic'] = run_basic_functionality_tests()
    
    # 2. TPU initialization and tests
    test_results['tpu'] = run_tpu_tests()
    
    # 3. JAX model tests (if available)
    test_results['jax_model'] = run_jax_model_tests()
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name.upper()}: {status}")
    
    # Overall result
    all_passed = all(test_results.values())
    
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! TPU is working correctly.")
        return 0
    else:
        logger.error("💥 SOME TESTS FAILED. Check the logs above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)