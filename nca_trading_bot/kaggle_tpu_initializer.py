"""
Kaggle TPU v5e-8 initializer for JAX-based NCA Trading Bot.

This module properly initializes JAX for TPU v5e-8 usage in Kaggle environment,
handling the common "Device or resource busy" error by properly configuring
JAX distributed training and TPU resources.
"""

import os
import sys
import time
import logging
from typing import Optional, Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_jax_for_tpu(
    coordinator_address: Optional[str] = None,
    num_processes: int = 1,
    process_id: int = 0,
    max_retries: int = 3,
    retry_delay: float = 5.0
) -> bool:
    """
    Initialize JAX for TPU v5e-8 usage in Kaggle.
    
    Args:
        coordinator_address: Address for distributed coordination (None for single host)
        num_processes: Number of processes (1 for single host)
        process_id: Process ID (0 for single host)
        max_retries: Maximum number of initialization retries
        retry_delay: Delay between retries in seconds
        
    Returns:
        True if initialization successful, False otherwise
    """
    # Set JAX environment variables for TPU
    os.environ["JAX_PLATFORMS"] = "tpu"
    os.environ["JAX_PLATFORM_NAME"] = "tpu"
    os.environ["JAX_XLA_BACKEND"] = "tpu"
    
    # Configure XLA for TPU v5e-8
    os.environ["XLA_FLAGS"] = (
        "--xla_tpu_enable_data_parallel_all_reduce_opt=true "
        "--xla_tpu_enable_async_collective_all_reduce=true "
        "--xla_tpu_enable_async_collective_all_to_all=true "
        "--xla_enable_async_collectives=true "
        f"--xla_tpu_memory_fraction=0.8 "
        "--xla_disable_hlo_passes=all-gather-combine"
    )
    
    # Configure JAX compilation cache
    os.environ["JAX_CACHE_DIR"] = "/tmp/jax_cache"
    os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
    os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
    
    # Create cache directory if it doesn't exist
    os.makedirs("/tmp/jax_cache", exist_ok=True)
    
    # Try initialization with retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Initializing JAX for TPU (attempt {attempt + 1}/{max_retries})")
            
            # Import JAX after setting environment variables
            import jax
            import jax.distributed
            
            # Initialize JAX distributed if coordinator address provided
            if coordinator_address:
                logger.info(f"Initializing JAX distributed with coordinator: {coordinator_address}")
                jax.distributed.initialize(
                    coordinator_address=coordinator_address,
                    num_processes=num_processes,
                    process_id=process_id
                )
            else:
                # For single host, try to initialize without distributed
                logger.info("Initializing JAX for single host TPU")
                # No explicit initialization needed for single host
            
            # Check if TPU devices are available
            devices = jax.devices()
            logger.info(f"Found {len(devices)} JAX devices: {[str(d) for d in devices]}")
            
            # Verify TPU devices
            tpu_devices = [d for d in devices if 'tpu' in str(d).lower()]
            if len(tpu_devices) == 0:
                logger.warning("No TPU devices found")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                return False
            
            logger.info(f"Successfully initialized {len(tpu_devices)} TPU devices")
            
            # Set JAX config for TPU v5e-8
            jax.config.update('jax_default_matmul_precision', 'bfloat16')
            jax.config.update('jax_enable_x64', False)  # Use float32 for efficiency
            
            return True
            
        except Exception as e:
            logger.error(f"TPU initialization failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries exceeded, TPU initialization failed")
                return False
    
    return False


def setup_tpu_mesh_for_v5e8() -> Tuple[Any, Any, Any, Any]:
    """
    Set up JAX sharding mesh for TPU v5e-8 (8 cores).
    
    Returns:
        Tuple of (mesh, data_sharding, model_sharding, batch_sharding)
    """
    try:
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh, PartitionSpec as P
        from jax.experimental import mesh_utils
        
        # Get available devices
        devices = jax.devices()
        logger.info(f"Setting up mesh for {len(devices)} devices")
        
        if len(devices) == 8:
            # TPU v5e-8 configuration: 1x8 mesh
            device_mesh = mesh_utils.create_device_mesh((1, 8))
            mesh = Mesh(device_mesh, axis_names=('data', 'model'))
            
            # Define sharding specifications
            data_sharding = P('data', None)      # Shard data across cores
            model_sharding = P(None, 'model')    # Replicate model across cores
            batch_sharding = P('data')           # Shard batch dimension
            
            logger.info("Set up 1x8 mesh for TPU v5e-8")
            
        elif len(devices) == 4:
            # Fallback for 4-core TPU
            device_mesh = mesh_utils.create_device_mesh((1, 4))
            mesh = Mesh(device_mesh, axis_names=('data', 'model'))
            
            data_sharding = P('data', None)
            model_sharding = P(None, 'model')
            batch_sharding = P('data')
            
            logger.info("Set up 1x4 mesh for 4-core TPU")
            
        else:
            # Fallback for other configurations
            logger.warning(f"Unexpected number of devices ({len(devices)}), using replicated sharding")
            mesh = Mesh(devices, axis_names=('data',))
            data_sharding = P('data')
            model_sharding = P()
            batch_sharding = P('data')
        
        return mesh, data_sharding, model_sharding, batch_sharding
        
    except Exception as e:
        logger.error(f"Failed to set up TPU mesh: {e}")
        return None, None, None, None


def configure_jax_memory_for_tpu(max_memory_gb: float = 35.0) -> bool:
    """
    Configure JAX memory settings for TPU v5e-8.
    
    Args:
        max_memory_gb: Maximum memory to use in GB (out of 40GB available)
        
    Returns:
        True if configuration successful, False otherwise
    """
    try:
        import jax
        
        # Set memory fraction
        memory_fraction = max_memory_gb / 40.0  # TPU v5e-8 has 40GB
        os.environ["JAX_PLATFORMS"] = "tpu"
        
        # Configure XLA memory
        xla_flags = os.environ.get("XLA_FLAGS", "")
        if f"--xla_tpu_memory_fraction={memory_fraction}" not in xla_flags:
            xla_flags += f" --xla_tpu_memory_fraction={memory_fraction}"
        os.environ["XLA_FLAGS"] = xla_flags
        
        # Pre-allocate some memory to avoid fragmentation
        logger.info(f"Configured JAX to use {max_memory_gb}GB ({memory_fraction:.1%}) of TPU memory")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to configure JAX memory: {e}")
        return False


def verify_tpu_functionality() -> Dict[str, Any]:
    """
    Verify TPU functionality with a simple computation.
    
    Returns:
        Dictionary containing verification results
    """
    results = {
        'initialized': False,
        'device_count': 0,
        'computation_success': False,
        'memory_info': None,
        'error': None
    }
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Check device count
        devices = jax.devices()
        results['device_count'] = len(devices)
        results['initialized'] = True
        
        # Simple computation test
        logger.info("Testing TPU computation...")
        
        @jax.jit
        def simple_computation(x):
            return jnp.sum(x ** 2)
        
        # Create test data on TPU
        test_data = jnp.ones((1000, 1000), dtype=jnp.float32)
        result = simple_computation(test_data)
        
        # Get actual result
        result_value = float(result)
        expected_value = 1000.0 * 1000.0  # sum of 1^2 for 1M elements
        
        if abs(result_value - expected_value) < 1e-3:
            results['computation_success'] = True
            logger.info(f"TPU computation test passed: {result_value}")
        else:
            logger.error(f"TPU computation test failed: got {result_value}, expected {expected_value}")
        
        # Get memory info (if available)
        try:
            from jax._src import xla_bridge
            backend = xla_bridge.get_backend()
            if hasattr(backend, 'memory_stats'):
                results['memory_info'] = backend.memory_stats()
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
        
    except Exception as e:
        logger.error(f"TPU verification failed: {e}")
        results['error'] = str(e)
    
    return results


def initialize_tpu_for_kaggle() -> Dict[str, Any]:
    """
    Complete TPU initialization for Kaggle environment.
    
    Returns:
        Dictionary containing initialization status and configuration
    """
    logger.info("Starting TPU initialization for Kaggle...")
    
    results = {
        'initialization_success': False,
        'mesh_setup_success': False,
        'memory_config_success': False,
        'verification_success': False,
        'device_info': {},
        'error': None
    }
    
    try:
        # Step 1: Initialize JAX for TPU
        if not initialize_jax_for_tpu():
            results['error'] = "JAX TPU initialization failed"
            return results
        
        # Step 2: Configure memory
        if not configure_jax_memory_for_tpu(max_memory_gb=35.0):
            results['error'] = "Memory configuration failed"
            return results
        
        results['memory_config_success'] = True
        
        # Step 3: Set up mesh
        mesh, data_sharding, model_sharding, batch_sharding = setup_tpu_mesh_for_v5e8()
        if mesh is None:
            results['error'] = "Mesh setup failed"
            return results
        
        results['mesh_setup_success'] = True
        
        # Step 4: Verify functionality
        verification = verify_tpu_functionality()
        if not verification['computation_success']:
            results['error'] = "TPU verification failed"
            return results
        
        results['verification_success'] = True
        results['initialization_success'] = True
        
        # Collect device info
        import jax
        devices = jax.devices()
        results['device_info'] = {
            'device_count': len(devices),
            'device_type': str(devices[0].device_kind) if devices else 'unknown',
            'platform': jax.devices()[0].platform if devices else 'unknown'
        }
        
        logger.info("TPU initialization completed successfully!")
        logger.info(f"Device info: {results['device_info']}")
        
        return results
        
    except Exception as e:
        logger.error(f"TPU initialization failed: {e}")
        results['error'] = str(e)
        return results


if __name__ == "__main__":
    # Run initialization when script is executed directly
    results = initialize_tpu_for_kaggle()
    
    if results['initialization_success']:
        print("✅ TPU initialization successful!")
        print(f"Device info: {results['device_info']}")
    else:
        print(f"❌ TPU initialization failed: {results['error']}")
        sys.exit(1)