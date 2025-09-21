"""
Unit tests for NCA model module.

Tests Neural Cellular Automata architecture, training, and caching functionality.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch

from ..nca_model import (
    ConvGRUCell, NCACell, NCATradingModel, NCAModelCache, NCATrainer,
    create_nca_model, load_nca_model
)
from ..config import ConfigManager


class TestConvGRUCell(unittest.TestCase):
    """Test cases for ConvGRUCell class."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 3
        self.hidden_dim = 5
        self.kernel_size = 3
        self.cell = ConvGRUCell(self.input_dim, self.hidden_dim, self.kernel_size)

    def test_initialization(self):
        """Test ConvGRU cell initialization."""
        self.assertEqual(self.cell.input_dim, self.input_dim)
        self.assertEqual(self.cell.hidden_dim, self.hidden_dim)
        self.assertEqual(self.cell.kernel_size, self.kernel_size)

        # Check that convolutional layers are created
        self.assertIsNotNone(self.cell.conv_z)
        self.assertIsNotNone(self.cell.conv_r)
        self.assertIsNotNone(self.cell.conv_h)

    def test_forward_pass(self):
        """Test forward pass through ConvGRU cell."""
        batch_size, channels, height, width = 2, self.input_dim, 10, 10

        # Create input tensors
        x = torch.randn(batch_size, channels, height, width)
        h = torch.randn(batch_size, self.hidden_dim, height, width)

        # Forward pass
        output = self.cell(x, h)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.hidden_dim, height, width))

        # Check output is different from input
        self.assertFalse(torch.allclose(output, h))


class TestNCACell(unittest.TestCase):
    """Test cases for NCACell class."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 4
        self.hidden_dim = 8
        self.kernel_size = 3
        self.cell = NCACell(self.state_dim, self.hidden_dim, self.kernel_size)

    def test_initialization(self):
        """Test NCA cell initialization."""
        self.assertEqual(self.cell.state_dim, self.state_dim)
        self.assertEqual(self.cell.hidden_dim, self.hidden_dim)

        # Check that parameters are created
        self.assertIsNotNone(self.cell.alpha)
        self.assertIsNotNone(self.cell.beta)
        self.assertIsNotNone(self.cell.gamma)

        # Check that convolutional layers are created
        self.assertIsNotNone(self.cell.conv1)
        self.assertIsNotNone(self.cell.conv2)
        self.assertIsNotNone(self.cell.adaptation_conv)

    def test_forward_pass(self):
        """Test forward pass through NCA cell."""
        batch_size, channels, height, width = 2, self.state_dim, 8, 8

        # Create input tensors
        x = torch.randn(batch_size, channels, height, width)
        h = torch.randn(batch_size, self.hidden_dim, height, width)

        # Forward pass
        new_state, new_hidden = self.cell(x, h)

        # Check output shapes
        self.assertEqual(new_state.shape, (batch_size, channels, height, width))
        self.assertEqual(new_hidden.shape, (batch_size, self.hidden_dim, height, width))

    def test_evolve(self):
        """Test NCA evolution over multiple steps."""
        batch_size, channels, height, width = 2, self.state_dim, 6, 6

        # Create input tensor
        x = torch.randn(batch_size, channels, height, width)

        # Evolve for multiple steps
        evolved = self.cell.evolve(x, steps=3)

        # Check output shape
        self.assertEqual(evolved.shape, (batch_size, channels, height, width))


class TestNCATradingModel(unittest.TestCase):
    """Test cases for NCATradingModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.model = NCATradingModel(self.config)

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.state_dim, self.config.nca.state_dim)
        self.assertEqual(self.model.hidden_dim, self.config.nca.hidden_dim)
        self.assertEqual(self.model.num_layers, self.config.nca.num_layers)

        # Check that layers are created
        self.assertIsNotNone(self.model.input_conv)
        self.assertIsNotNone(self.model.nca_layers)
        self.assertIsNotNone(self.model.output_conv)

        # Check that trading heads are created
        self.assertIsNotNone(self.model.price_predictor)
        self.assertIsNotNone(self.model.signal_classifier)
        self.assertIsNotNone(self.model.risk_predictor)

    def test_forward_pass(self):
        """Test forward pass through model."""
        batch_size, seq_len, features = 4, 10, 20

        # Create input tensor
        x = torch.randn(batch_size, seq_len, features)

        # Forward pass
        outputs = self.model(x)

        # Check output structure
        expected_keys = ['price_prediction', 'signal_probabilities', 'risk_probability',
                        'final_state', 'adaptation_rate', 'selection_pressure']

        for key in expected_keys:
            self.assertIn(key, outputs)

        # Check output shapes
        self.assertEqual(outputs['signal_probabilities'].shape[0], batch_size)
        self.assertEqual(outputs['signal_probabilities'].shape[1], 3)  # Buy, Hold, Sell

    def test_predict(self):
        """Test prediction method."""
        batch_size, seq_len, features = 2, 5, 10

        # Create input tensor
        x = torch.randn(batch_size, seq_len, features)

        # Get prediction
        prediction = self.model.predict(x)

        # Check prediction structure
        expected_keys = ['trading_signal', 'confidence', 'price_prediction', 'risk_probability']

        for key in expected_keys:
            self.assertIn(key, prediction)

        # Check signal values
        self.assertIn(prediction['trading_signal'], ['buy', 'hold', 'sell'])
        self.assertIsInstance(prediction['confidence'], float)
        self.assertTrue(0 <= prediction['confidence'] <= 1)

    def test_adapt_online(self):
        """Test online adaptation."""
        batch_size, seq_len, features = 2, 5, 10

        # Create input and target tensors
        x = torch.randn(batch_size, seq_len, features)
        target = torch.randn(batch_size)

        # Perform online adaptation
        metrics = self.model.adapt_online(x, target)

        # Check metrics structure
        expected_keys = ['total_loss', 'price_loss', 'signal_loss', 'adaptation_strength']

        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)

    def test_update_performance(self):
        """Test performance tracking and adaptation."""
        # Create sample metrics
        metrics = {
            'reward': 0.1,
            'win_rate': 0.6,
            'total_trades': 10
        }

        # Update performance
        self.model.update_performance(metrics)

        # Check that performance history is updated
        self.assertEqual(len(self.model.performance_history), 1)
        self.assertEqual(self.model.performance_history[0], metrics)


class TestNCAModelCache(unittest.TestCase):
    """Test cases for NCAModelCache class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = NCAModelCache()

    def test_cache_key_generation(self):
        """Test cache key generation."""
        # Test with tensor
        tensor = torch.randn(10, 5)
        key = self.cache._generate_cache_key(tensor)
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 16)

        # Test with numpy array
        array = np.random.randn(10, 5)
        key = self.cache._generate_cache_key(array)
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 16)

    def test_cache_operations(self):
        """Test cache set and get operations."""
        # Create test data
        test_data = torch.randn(5, 3)

        # Test cache set
        self.cache.cache_state('model_hash', 'input_hash', test_data)

        # Test cache get
        retrieved = self.cache.get_cached_state('model_hash', 'input_hash')
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.allclose(test_data, retrieved))

    def test_cache_ttl(self):
        """Test cache TTL functionality."""
        # Create test data
        test_data = torch.randn(3, 3)

        # Cache data
        self.cache.cache_state('model_hash', 'input_hash', test_data)

        # Verify data is cached
        retrieved = self.cache.get_cached_state('model_hash', 'input_hash')
        self.assertIsNotNone(retrieved)

        # Manually expire the cache entry (simulate TTL)
        cache_path = self.cache.cache_dir / 'model_hash_input_hash.pt'
        if cache_path.exists():
            # Modify timestamp to be old
            import os
            os.utime(cache_path, (0, 0))  # Set to epoch time

        # Try to retrieve (should return None due to TTL)
        retrieved = self.cache.get_cached_state('model_hash', 'input_hash')
        self.assertIsNone(retrieved)


class TestNCATrainer(unittest.TestCase):
    """Test cases for NCATrainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.model = NCATradingModel(self.config)
        self.trainer = NCATrainer(self.model, self.config)

    def test_initialization(self):
        """Test trainer initialization."""
        self.assertEqual(self.trainer.model, self.model)
        self.assertEqual(self.trainer.config, self.config)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.scheduler)

        # Check device
        self.assertIsInstance(self.trainer.device, torch.device)

    def test_train_step(self):
        """Test single training step."""
        batch_size, seq_len, features = 4, 10, 20

        # Create batch data
        batch = {
            'x': torch.randn(batch_size, seq_len, features),
            'price_target': torch.randn(batch_size),
            'signal_target': torch.randint(0, 3, (batch_size,)),
            'risk_target': torch.randn(batch_size)
        }

        # Perform training step
        metrics = self.trainer.train_step(batch)

        # Check metrics structure
        expected_keys = ['total_loss', 'policy_loss', 'value_loss', 'entropy', 'learning_rate']

        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)

    def test_validate(self):
        """Test validation method."""
        # Create mock validation data
        val_data = [
            {
                'x': torch.randn(2, 10, 20),
                'price_target': torch.randn(2),
                'signal_target': torch.randint(0, 3, (2,)),
                'risk_target': torch.randn(2)
            }
        ]

        # Run validation
        metrics = self.trainer.validate(val_data)

        # Check metrics
        self.assertIn('val_loss', metrics)
        self.assertIsInstance(metrics['val_loss'], float)

    def test_checkpointing(self):
        """Test model checkpointing."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pt')

            # Create sample metrics
            metrics = {'loss': 0.5, 'accuracy': 0.8}

            # Save checkpoint
            self.trainer.save_checkpoint(checkpoint_path, epoch=5, metrics=metrics)

            # Verify file exists
            self.assertTrue(os.path.exists(checkpoint_path))

            # Load checkpoint
            loaded_data = self.trainer.load_checkpoint(checkpoint_path)

            # Check loaded data
            self.assertEqual(loaded_data['epoch'], 5)
            self.assertEqual(loaded_data['metrics'], metrics)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_create_nca_model(self):
        """Test create_nca_model function."""
        config = ConfigManager()
        model = create_nca_model(config)

        self.assertIsInstance(model, NCATradingModel)
        self.assertEqual(model.state_dim, config.nca.state_dim)

    def test_load_nca_model(self):
        """Test load_nca_model function."""
        config = ConfigManager()
        model = NCATradingModel(config)

        # Save model to temporary file
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)

            try:
                # Load model
                loaded_model = load_nca_model(f.name, config)

                self.assertIsInstance(loaded_model, NCATradingModel)
                self.assertEqual(loaded_model.state_dim, model.state_dim)

            finally:
                os.unlink(f.name)


if __name__ == '__main__':
    unittest.main()