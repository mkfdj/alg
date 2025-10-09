"""
Simple TPU test for NCA Trading Bot.

This script performs basic TPU initialization and tests to verify
that the TPU v5e-8 is working correctly with JAX, avoiding conflicts
with PyTorch/XLA.
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


def test_jax_tpu_initialization():
    """Test JAX TPU initialization."""
    logger.info("Testing JAX TPU initialization...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Check devices
        devices = jax.devices()
        logger.info(f"Found {len(devices)} JAX devices")
        
        for i, device in enumerate(devices):
            logger.info(f"  Device {i}: {device}")
        
        # Check if TPU devices are available
        tpu_devices = [d for d in devices if 'tpu' in str(d).lower()]
        
        if len(tpu_devices) == 0:
            logger.error("❌ No TPU devices found")
            return False
        
        logger.info(f"✅ Found {len(tpu_devices)} TPU devices")
        return True
        
    except Exception as e:
        logger.error(f"❌ JAX TPU initialization failed: {e}")
        return False


def test_tpu_computation():
    """Test TPU computation."""
    logger.info("Testing TPU computation...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Simple computation
        @jax.jit
        def simple_computation(x, y):
            return jnp.matmul(x, y)
        
        # Create test data
        x = jnp.ones((100, 100), dtype=jnp.float32)
        y = jnp.ones((100, 100), dtype=jnp.float32)
        
        # Run computation
        result = simple_computation(x, y)
        expected = 100.0 * 100.0  # 10K
        
        # Check result
        result_float = float(jnp.sum(result))
        
        if abs(result_float - expected) < 1e-3:
            logger.info(f"✅ TPU computation successful: {result_float}")
            return True
        else:
            logger.error(f"❌ TPU computation incorrect: got {result_float}, expected {expected}")
            return False
            
    except Exception as e:
        logger.error(f"❌ TPU computation failed: {e}")
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


def test_tpu_mesh():
    """Test TPU mesh setup."""
    logger.info("Testing TPU mesh setup...")
    
    try:
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
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
        data_sharding = P('data')
        named_sharding = NamedSharding(mesh, data_sharding)
        
        # Put data on device
        sharded_data = jax.device_put(data, named_sharding)
        logger.info("✅ Data sharding successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ TPU mesh setup failed: {e}")
        return False


def test_jax_config():
    """Test JAX configuration."""
    logger.info("Testing JAX configuration...")
    
    try:
        import jax
        
        # Check JAX version
        logger.info(f"JAX version: {jax.__version__}")
        
        # Check backend
        logger.info(f"JAX backend: {jax.devices()[0].platform}")
        
        # Check XLA compilation
        logger.info(f"JAX XLA compilation: {jax.config.jax_experimental_cache}")
        
        # Check if jax is using TPU
        devices = jax.devices()
        if devices and 'tpu' in str(devices[0]).lower():
            logger.info("✅ JAX is configured to use TPU")
            return True
        else:
            logger.error("❌ JAX is not configured to use TPU")
            return False
            
    except Exception as e:
        logger.error(f"❌ JAX configuration test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("Simple TPU Test for NCA Trading Bot")
    logger.info("=" * 60)
    
    # Change to project directory
    os.chdir(project_root)
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Run tests
    tests = [
        ("JAX Configuration", test_jax_config),
        ("JAX TPU Initialization", test_jax_tpu_initialization),
        ("TPU Computation", test_tpu_computation),
        ("Mixed Precision", test_mixed_precision),
        ("TPU Mesh Setup", test_tpu_mesh),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        results[test_name] = test_func()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
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
        logger.info("🎉 All TPU tests passed!")
        logger.info("Your TPU v5e-8 is ready for NCA Trading Bot training.")
        return 0
    else:
        logger.error("💥 Some TPU tests failed.")
        logger.error("Please check the errors above and try again.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)