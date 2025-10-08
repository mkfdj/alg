"""
Simple TPU functionality verification script for Kaggle.

This script performs basic TPU initialization and tests to verify
that the TPU v5e-8 is working correctly with JAX.
"""

import os
import sys
import time
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


def test_basic_jax_functionality():
    """Test basic JAX functionality without TPU."""
    logger.info("Testing basic JAX functionality...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Simple computation
        x = jnp.ones((1000, 1000))
        y = jnp.ones((1000, 1000))
        
        @jax.jit
        def matrix_mult(x, y):
            return jnp.matmul(x, y)
        
        result = matrix_mult(x, y)
        logger.info(f"✅ Basic JAX computation successful: {result.shape}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic JAX computation failed: {e}")
        return False


def test_tpu_initialization():
    """Test TPU initialization using the initializer."""
    logger.info("Testing TPU initialization...")
    
    try:
        from kaggle_tpu_initializer import initialize_tpu_for_kaggle
        
        # Initialize TPU
        results = initialize_tpu_for_kaggle()
        
        if results['initialization_success']:
            logger.info("✅ TPU initialization successful!")
            logger.info(f"Device info: {results['device_info']}")
            return True
        else:
            logger.error(f"❌ TPU initialization failed: {results['error']}")
            return False
            
    except Exception as e:
        logger.error(f"❌ TPU initialization failed with exception: {e}")
        return False


def test_tpu_computation():
    """Test TPU computation."""
    logger.info("Testing TPU computation...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Check if TPU devices are available
        devices = jax.devices()
        tpu_devices = [d for d in devices if 'tpu' in str(d).lower()]
        
        if len(tpu_devices) == 0:
            logger.error("❌ No TPU devices found")
            return False
        
        logger.info(f"Found {len(tpu_devices)} TPU devices")
        
        # Test computation on TPU
        @jax.jit
        def tpu_computation():
            # Create data on TPU
            x = jnp.ones((1000, 1000), dtype=jnp.float32)
            y = jnp.ones((1000, 1000), dtype=jnp.float32)
            
            # Perform computation
            result = jnp.matmul(x, y)
            return jnp.sum(result)
        
        # Run computation
        result = tpu_computation()
        expected = 1000.0 * 1000.0  # 1M
        
        if abs(float(result) - expected) < 1e-3:
            logger.info(f"✅ TPU computation successful: {result}")
            return True
        else:
            logger.error(f"❌ TPU computation incorrect: got {result}, expected {expected}")
            return False
            
    except Exception as e:
        logger.error(f"❌ TPU computation failed: {e}")
        return False


def test_tpu_mesh():
    """Test TPU mesh setup."""
    logger.info("Testing TPU mesh setup...")
    
    try:
        import jax
        from jax.sharding import Mesh, PartitionSpec as P
        from jax.experimental import mesh_utils
        
        # Get devices
        devices = jax.devices()
        logger.info(f"Found {len(devices)} devices")
        
        # Create mesh
        if len(devices) == 8:
            # TPU v5e-8
            device_mesh = mesh_utils.create_device_mesh((1, 8))
            mesh = Mesh(device_mesh, axis_names=('data', 'model'))
            logger.info("✅ Created 1x8 mesh for TPU v5e-8")
        else:
            # Fallback
            device_mesh = mesh_utils.create_device_mesh((len(devices),))
            mesh = Mesh(device_mesh, axis_names=('data',))
            logger.info(f"✅ Created {len(devices)}x1 mesh")
        
        # Test sharding
        data = jnp.ones((100, 10))
        data_sharding = P('data', None)
        
        with mesh:
            sharded_data = jax.device_put(data, data_sharding)
            logger.info("✅ Data sharding successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ TPU mesh setup failed: {e}")
        return False


def test_mixed_precision():
    """Test mixed precision computation."""
    logger.info("Testing mixed precision computation...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Configure for bfloat16
        jax.config.update('jax_default_matmul_precision', 'bfloat16')
        
        # Create bfloat16 tensors
        x = jnp.ones((512, 512), dtype=jnp.bfloat16)
        y = jnp.ones((512, 512), dtype=jnp.bfloat16)
        
        @jax.jit
        def bfloat16_matmul(x, y):
            return jnp.matmul(x, y)
        
        result = bfloat16_matmul(x, y)
        
        if result.dtype == jnp.bfloat16:
            logger.info("✅ Mixed precision computation successful")
            return True
        else:
            logger.error(f"❌ Mixed precision failed: got {result.dtype}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Mixed precision computation failed: {e}")
        return False


def main():
    """Main verification function."""
    logger.info("=" * 60)
    logger.info("TPU Functionality Verification for NCA Trading Bot")
    logger.info("=" * 60)
    
    # Change to project directory
    os.chdir(project_root)
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Run tests
    tests = [
        ("Basic JAX Functionality", test_basic_jax_functionality),
        ("TPU Initialization", test_tpu_initialization),
        ("TPU Computation", test_tpu_computation),
        ("TPU Mesh Setup", test_tpu_mesh),
        ("Mixed Precision", test_mixed_precision),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        results[test_name] = test_func()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All TPU functionality tests passed!")
        logger.info("Your TPU v5e-8 is ready for NCA Trading Bot training.")
        return 0
    else:
        logger.error("💥 Some TPU functionality tests failed.")
        logger.error("Please check the errors above and try again.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)