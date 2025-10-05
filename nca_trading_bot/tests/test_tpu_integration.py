"""
Unit tests for TPU integration functionality.

Tests TPU detection, configuration, and optimization features.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch

from ..config import ConfigManager, detect_tpu_availability, get_tpu_device_count
from ..nca_model import create_nca_model, NCATradingModel
from ..trainer import PPOTrainer, DistributedTrainer
from ..utils import PerformanceMonitor


class TestTPUDetection(unittest.TestCase):
    """Test cases for TPU detection functionality."""

    def test_tpu_detection_function(self):
        """Test TPU detection function."""
        # Should not raise exception
        tpu_available = detect_tpu_availability()
        self.assertIsInstance(tpu_available, bool)

    def test_tpu_device_count_function(self):
        """Test TPU device count function."""
        # Should not raise exception
        tpu_count = get_tpu_device_count()
        self.assertIsInstance(tpu_count, int)
        self.assertGreaterEqual(tpu_count, 0)


class TestTPUConfiguration(unittest.TestCase):
    """Test cases for TPU configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()

    def test_tpu_config_settings(self):
        """Test TPU configuration settings."""
        # Test TPU device setting
        self.config.system.device = "tpu"
        self.config.tpu.tpu_cores = 8
        self.config.tpu.xla_compile = True
        self.config.tpu.sharding_strategy = "2d"

        self.assertEqual(self.config.system.device, "tpu")
        self.assertEqual(self.config.tpu.tpu_cores, 8)
        self.assertTrue(self.config.tpu.xla_compile)
        self.assertEqual(self.config.tpu.sharding_strategy, "2d")


class TestTPUModelCreation(unittest.TestCase):
    """Test cases for TPU-optimized model creation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.config.system.device = "tpu"

    def test_tpu_model_creation(self):
        """Test TPU model creation."""
        try:
            model = create_nca_model(self.config)
            self.assertIsInstance(model, NCATradingModel)
            
            # Check parameter count
            param_count = sum(p.numel() for p in model.parameters())
            self.assertGreater(param_count, 0)
        except ImportError:
            # Skip test if TPU dependencies not available
            self.skipTest("TPU dependencies not available")


class TestTPUTrainingSetup(unittest.TestCase):
    """Test cases for TPU training setup."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.config.system.device = "tpu"

    def test_tpu_trainer_creation(self):
        """Test TPU trainer creation."""
        try:
            model = create_nca_model(self.config)
            trainer = PPOTrainer(model, self.config)
            
            self.assertEqual(trainer.model, model)
            self.assertEqual(trainer.config, self.config)
            self.assertIsNotNone(trainer.optimizer)
        except ImportError:
            # Skip test if TPU dependencies not available
            self.skipTest("TPU dependencies not available")


class TestTPUDistributedTraining(unittest.TestCase):
    """Test cases for TPU distributed training."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.config.system.device = "tpu"

    def test_distributed_trainer_creation(self):
        """Test distributed trainer creation."""
        try:
            distributed_trainer = DistributedTrainer(self.config)
            
            self.assertEqual(distributed_trainer.world_size, self.config.training.num_gpus)
            self.assertEqual(distributed_trainer.local_rank, self.config.training.local_rank)
        except ImportError:
            # Skip test if TPU dependencies not available
            self.skipTest("TPU dependencies not available")


class TestTPUPerformanceMonitoring(unittest.TestCase):
    """Test cases for TPU performance monitoring."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()

    def test_system_metrics_collection(self):
        """Test system metrics collection."""
        metrics = self.monitor.get_system_metrics()
        
        expected_keys = ['cpu_percent', 'memory_percent', 'gpu_utilization', 
                        'gpu_memory_percent', 'disk_percent']
        
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))

    def test_tpu_metrics_collection(self):
        """Test TPU-specific metrics collection."""
        try:
            tpu_metrics = self.monitor.get_tpu_training_metrics()
            
            # Should return a dictionary
            self.assertIsInstance(tpu_metrics, dict)
        except AttributeError:
            # Skip test if TPU metrics not available
            self.skipTest("TPU metrics not available")


class TestXLACompilation(unittest.TestCase):
    """Test cases for XLA compilation functionality."""

    def test_xla_import(self):
        """Test XLA import."""
        try:
            import torch_xla.core.xla_model as xm
            self.assertIsNotNone(xm)
        except ImportError:
            # Skip test if XLA not available
            self.skipTest("XLA not available")

    def test_xla_device_creation(self):
        """Test XLA device creation."""
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            self.assertIsNotNone(device)
        except ImportError:
            # Skip test if XLA not available
            self.skipTest("XLA not available")


class TestTPUOptimizations(unittest.TestCase):
    """Test cases for TPU-specific optimizations."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.config.system.device = "tpu"

    def test_tpu_optimized_forward_pass(self):
        """Test TPU-optimized forward pass."""
        try:
            model = create_nca_model(self.config)
            
            # Create sample input
            batch_size, seq_len, features = 2, 10, 20
            sample_input = torch.randn(batch_size, seq_len, features)
            
            # Test regular forward pass
            outputs1 = model(sample_input, evolution_steps=1)
            
            # Test TPU-optimized forward pass (if available)
            if hasattr(model, 'forward_tpu_optimized'):
                outputs2 = model.forward_tpu_optimized(sample_input, evolution_steps=1)
                
                # Verify outputs have same structure
                self.assertEqual(set(outputs1.keys()), set(outputs2.keys()))
            
            # Verify outputs have expected structure
            expected_keys = ['price_prediction', 'signal_probabilities', 'risk_probability']
            for key in expected_keys:
                self.assertIn(key, outputs1)
                
        except ImportError:
            # Skip test if TPU dependencies not available
            self.skipTest("TPU dependencies not available")


if __name__ == '__main__':
    unittest.main()