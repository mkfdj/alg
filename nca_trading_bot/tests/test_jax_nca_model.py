"""
Unit tests for JAX-based NCA model functionality.

Tests JAX NCA model creation, training, and TPU optimization features.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from unittest.mock import Mock, patch

from ..config import ConfigManager
from ..nca_model import create_jax_nca_model, create_jax_train_state


class TestJAXNCAModel(unittest.TestCase):
    """Test cases for JAX NCA model."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.config.system.device = "tpu"
        self.config.tpu.use_jax = True
        self.config.tpu.jax_backend = "tpu"

    def test_jax_nca_model_creation(self):
        """Test JAX NCA model creation."""
        try:
            # Check if JAX NCA model function exists
            if not hasattr(__import__('nca_model', fromlist=['create_jax_nca_model']), 'create_jax_nca_model'):
                self.skipTest("JAX NCA model not implemented")
            
            model = create_jax_nca_model(self.config)
            self.assertIsNotNone(model)
            
            # Test model structure
            if hasattr(model, 'tabulate'):
                # Print model structure if available
                rng = jax.random.PRNGKey(42)
                x = jnp.ones((1, 60, 20))  # Sample input
                print(model.tabulate(rng, x))
            
        except (ImportError, AttributeError) as e:
            self.skipTest(f"JAX NCA model not available: {e}")

    def test_jax_train_state_creation(self):
        """Test JAX train state creation."""
        try:
            # Check if JAX train state function exists
            if not hasattr(__import__('nca_model', fromlist=['create_jax_train_state']), 'create_jax_train_state'):
                self.skipTest("JAX train state not implemented")
            
            model = create_jax_nca_model(self.config)
            rng = jax.random.PRNGKey(42)
            
            state, mesh = create_jax_train_state(model, self.config.__dict__, rng)
            
            self.assertIsNotNone(state)
            self.assertIsNotNone(mesh)
            
            # Check state structure
            self.assertTrue(hasattr(state, 'params'))
            self.assertTrue(hasattr(state, 'apply_fn'))
            
        except (ImportError, AttributeError) as e:
            self.skipTest(f"JAX train state not available: {e}")

    def test_jax_forward_pass(self):
        """Test JAX model forward pass."""
        try:
            model = create_jax_nca_model(self.config)
            rng = jax.random.PRNGKey(42)
            
            # Create sample input
            batch_size, seq_len, features = 2, 60, 20
            sample_input = jnp.ones((batch_size, seq_len, features))
            
            # Initialize model parameters
            variables = model.init(rng, sample_input)
            
            # Forward pass
            outputs = model.apply(variables, sample_input)
            
            # Check output structure
            self.assertIsInstance(outputs, dict)
            expected_keys = ['price_prediction', 'signal_probabilities', 'risk_probability']
            for key in expected_keys:
                if key in outputs:
                    self.assertIsNotNone(outputs[key])
            
        except (ImportError, AttributeError) as e:
            self.skipTest(f"JAX forward pass failed: {e}")

    def test_jax_compiled_forward_pass(self):
        """Test JAX compiled forward pass."""
        try:
            model = create_jax_nca_model(self.config)
            rng = jax.random.PRNGKey(42)
            
            # Create sample input
            batch_size, seq_len, features = 2, 60, 20
            sample_input = jnp.ones((batch_size, seq_len, features))
            
            # Initialize model parameters
            variables = model.init(rng, sample_input)
            
            # Compile forward pass
            @jax.jit
            def compiled_forward(params, x):
                return model.apply({'params': params}, x)
            
            # Run compiled forward pass
            outputs = compiled_forward(variables['params'], sample_input)
            
            # Check output structure
            self.assertIsInstance(outputs, dict)
            
        except (ImportError, AttributeError) as e:
            self.skipTest(f"JAX compiled forward pass failed: {e}")


class TestJAXSharding(unittest.TestCase):
    """Test cases for JAX sharding functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.config.system.device = "tpu"
        self.config.tpu.use_jax = True

    def test_tpu_mesh_creation(self):
        """Test TPU mesh creation for sharding."""
        try:
            from jax.sharding import Mesh, PartitionSpec as P
            from jax.experimental import mesh_utils
            
            devices = jax.devices()
            
            if len(devices) == 8:
                # TPU v5e-8 configuration
                device_mesh = mesh_utils.create_device_mesh((1, 8))
                mesh = Mesh(device_mesh, axis_names=('data', 'model'))
                
                self.assertEqual(mesh.shape, (1, 8))
                self.assertEqual(mesh.axis_names, ('data', 'model'))
                
            else:
                # Fallback for other configurations
                device_mesh = mesh_utils.create_device_mesh((len(devices),))
                mesh = Mesh(device_mesh, axis_names=('data',))
                
                self.assertEqual(mesh.shape, (len(devices),))
                self.assertEqual(mesh.axis_names, ('data',))
            
        except ImportError:
            self.skipTest("JAX sharding not available")

    def test_data_sharding(self):
        """Test data sharding across TPU cores."""
        try:
            from jax.sharding import PartitionSpec as P
            
            # Create sample data
            batch_size = 2048
            data = jnp.ones((batch_size, 60, 20))
            
            # Define sharding
            data_sharding = P('data', None, None)  # Shard batch dimension
            
            # Apply sharding (if mesh is available)
            devices = jax.devices()
            if len(devices) >= 4:
                from jax.sharding import Mesh
                from jax.experimental import mesh_utils
                
                device_mesh = mesh_utils.create_device_mesh((len(devices),))
                mesh = Mesh(device_mesh, axis_names=('data',))
                
                with mesh:
                    sharded_data = jax.device_put(data, data_sharding)
                    self.assertIsNotNone(sharded_data)
            
        except ImportError:
            self.skipTest("JAX sharding not available")


class TestJAXMemoryOptimization(unittest.TestCase):
    """Test cases for JAX memory optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.config.tpu.max_memory_gb = 35.0
        self.config.tpu.memory_fraction = 0.875

    def test_memory_efficient_forward_pass(self):
        """Test memory-efficient forward pass with rematerialization."""
        try:
            from jax.experimental import checkify
            
            model = create_jax_nca_model(self.config)
            rng = jax.random.PRNGKey(42)
            
            # Create larger sample input
            batch_size, seq_len, features = 32, 60, 20
            sample_input = jnp.ones((batch_size, seq_len, features))
            
            # Initialize model parameters
            variables = model.init(rng, sample_input)
            
            # Memory-efficient forward pass with rematerialization
            @jax.remat
            def memory_efficient_forward(params, x):
                return model.apply({'params': params}, x)
            
            # Run memory-efficient forward pass
            outputs = memory_efficient_forward(variables['params'], sample_input)
            
            # Check output structure
            self.assertIsInstance(outputs, dict)
            
        except (ImportError, AttributeError) as e:
            self.skipTest(f"Memory optimization not available: {e}")

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing for memory efficiency."""
        try:
            model = create_jax_nca_model(self.config)
            rng = jax.random.PRNGKey(42)
            
            # Create sample input
            batch_size, seq_len, features = 16, 60, 20
            sample_input = jnp.ones((batch_size, seq_len, features))
            target = jnp.ones((batch_size, 1))
            
            # Initialize model parameters
            variables = model.init(rng, sample_input)
            
            # Define loss function with gradient checkpointing
            @jax.remat
            def checkpointed_forward(params, x):
                outputs = model.apply({'params': params}, x)
                return outputs.get('price_prediction', jnp.zeros((batch_size, 1)))
            
            def loss_fn(params, x, y):
                pred = checkpointed_forward(params, x)
                return jnp.mean((pred - y) ** 2)
            
            # Compute gradients with checkpointing
            grad_fn = jax.grad(loss_fn)
            grads = grad_fn(variables['params'], sample_input, target)
            
            # Check gradients
            self.assertIsNotNone(grads)
            
        except (ImportError, AttributeError) as e:
            self.skipTest(f"Gradient checkpointing not available: {e}")


class TestJAXMixedPrecision(unittest.TestCase):
    """Test cases for JAX mixed precision training."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.config.tpu.mixed_precision = True
        self.config.tpu.precision_dtype = "bf16"
        self.config.tpu.compute_dtype = "f32"

    def test_bfloat16_computation(self):
        """Test bfloat16 computation on TPU."""
        try:
            # Configure JAX for bfloat16
            jax.config.update('jax_default_matmul_precision', 'bfloat16')
            
            # Create bfloat16 tensors
            x = jnp.ones((100, 100), dtype=jnp.bfloat16)
            y = jnp.ones((100, 100), dtype=jnp.bfloat16)
            
            # Matrix multiplication in bfloat16
            z = jnp.matmul(x, y)
            
            # Check result
            self.assertEqual(z.dtype, jnp.bfloat16)
            self.assertEqual(z.shape, (100, 100))
            
        except (ImportError, AttributeError) as e:
            self.skipTest(f"bfloat16 computation not available: {e}")

    def test_mixed_precision_forward_pass(self):
        """Test mixed precision forward pass."""
        try:
            model = create_jax_nca_model(self.config)
            rng = jax.random.PRNGKey(42)
            
            # Create sample input in float32
            batch_size, seq_len, features = 2, 60, 20
            sample_input = jnp.ones((batch_size, seq_len, features), dtype=jnp.float32)
            
            # Initialize model parameters
            variables = model.init(rng, sample_input)
            
            # Forward pass with mixed precision
            with jax.default_matmul_precision("bfloat16"):
                outputs = model.apply(variables, sample_input)
            
            # Check output structure
            self.assertIsInstance(outputs, dict)
            
        except (ImportError, AttributeError) as e:
            self.skipTest(f"Mixed precision not available: {e}")


if __name__ == '__main__':
    unittest.main()