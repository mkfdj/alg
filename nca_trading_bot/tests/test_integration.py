"""
Integration tests for NCA Trading Bot components.

Tests the integration between different modules and components.
"""

import unittest
import asyncio
import sys
import os
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ConfigManager
from nca_model import create_nca_model
from data_handler import DataHandler
from trainer import PPOTrainer
from trader import TradingAgent
from utils import PerformanceMonitor


class TestComponentIntegration(unittest.TestCase):
    """Test cases for component integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.model = create_nca_model(self.config)
        self.data_handler = DataHandler()
        self.trainer = PPOTrainer(self.model, self.config)
        self.trading_agent = TradingAgent(self.model, self.config)
        self.monitor = PerformanceMonitor()

    def test_model_trainer_integration(self):
        """Test integration between model and trainer."""
        # Create sample batch
        batch_size, seq_len, features = 4, 10, 20
        batch = {
            'states': torch.randn(batch_size, seq_len, features),
            'actions': torch.randint(0, 3, (batch_size,)),
            'log_probs': torch.randn(batch_size),
            'advantages': torch.randn(batch_size),
            'returns': torch.randn(batch_size)
        }

        # Test training step
        metrics = self.trainer.train_step(batch)

        # Check metrics structure
        expected_keys = ['total_loss', 'policy_loss', 'value_loss', 'entropy', 'learning_rate']
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)

    def test_model_trader_integration(self):
        """Test integration between model and trading agent."""
        # Create sample market data
        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'Open': np.random.randn(10) + 100,
            'High': np.random.randn(10) + 101,
            'Low': np.random.randn(10) + 99,
            'Close': np.random.randn(10) + 100,
            'Volume': np.random.randint(1000, 10000, 10)
        })

        # Test decision making
        decision = self.trading_agent.make_decision(market_data)

        # Check decision structure
        expected_keys = ['action', 'confidence', 'position_size', 'stop_loss', 'take_profit', 'risk_amount']
        for key in expected_keys:
            self.assertIn(key, decision)

        # Check action values
        self.assertIn(decision['action'], ['buy', 'hold', 'sell'])
        self.assertIsInstance(decision['confidence'], float)

    def test_data_handler_integration(self):
        """Test data handler integration with other components."""
        # Test data handler initialization
        self.assertIsNotNone(self.data_handler.fetcher)
        self.assertIsNotNone(self.data_handler.indicators)
        self.assertIsNotNone(self.data_handler.preprocessor)

    def test_monitor_integration(self):
        """Test performance monitor integration."""
        # Test metrics logging
        metrics = {'loss': 0.5, 'accuracy': 0.85}
        self.monitor.log_metrics(metrics, step=10)

        # Check that metrics are stored
        self.assertIn('loss', self.monitor.metrics)
        self.assertIn('accuracy', self.monitor.metrics)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test cases for end-to-end workflows."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()

    def test_model_creation_to_training_workflow(self):
        """Test workflow from model creation to training."""
        # Create model
        model = create_nca_model(self.config)
        self.assertIsInstance(model, type(self.model))

        # Create trainer
        trainer = PPOTrainer(model, self.config)
        self.assertEqual(trainer.model, model)

        # Create sample training data
        batch_size, seq_len, features = 2, 10, 20
        batch = {
            'states': torch.randn(batch_size, seq_len, features),
            'actions': torch.randint(0, 3, (batch_size,)),
            'log_probs': torch.randn(batch_size),
            'advantages': torch.randn(batch_size),
            'returns': torch.randn(batch_size)
        }

        # Test training step
        metrics = trainer.train_step(batch)
        self.assertIn('total_loss', metrics)

    def test_model_to_trading_workflow(self):
        """Test workflow from model to trading."""
        # Create model
        model = create_nca_model(self.config)

        # Create trading agent
        agent = TradingAgent(model, self.config)
        self.assertEqual(agent.model, model)

        # Create sample market data
        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'Open': np.random.randn(10) + 100,
            'High': np.random.randn(10) + 101,
            'Low': np.random.randn(10) + 99,
            'Close': np.random.randn(10) + 100,
            'Volume': np.random.randint(1000, 10000, 10)
        })

        # Test decision making
        decision = agent.make_decision(market_data)
        self.assertIn('action', decision)

    def test_data_to_model_workflow(self):
        """Test workflow from data to model."""
        # Create data handler
        data_handler = DataHandler()

        # Create model
        model = create_nca_model(self.config)

        # Create sample input data
        batch_size, seq_len, features = 2, 10, 20
        input_data = torch.randn(batch_size, seq_len, features)

        # Test model prediction
        outputs = model(input_data)
        self.assertIn('price_prediction', outputs)
        self.assertIn('signal_probabilities', outputs)


class TestConfigurationIntegration(unittest.TestCase):
    """Test cases for configuration integration."""

    def test_config_model_integration(self):
        """Test configuration integration with model."""
        config = ConfigManager()
        
        # Modify configuration
        config.nca.state_dim = 128
        config.nca.hidden_dim = 256
        
        # Create model with modified config
        model = create_nca_model(config)
        
        # Check that model uses configuration
        self.assertEqual(model.state_dim, 128)
        self.assertEqual(model.hidden_dim, 256)

    def test_config_trainer_integration(self):
        """Test configuration integration with trainer."""
        config = ConfigManager()
        
        # Modify configuration
        config.training.batch_size = 32
        config.training.learning_rate = 0.0005
        
        # Create model and trainer
        model = create_nca_model(config)
        trainer = PPOTrainer(model, config)
        
        # Check that trainer uses configuration
        self.assertEqual(trainer.batch_size, 32)
        self.assertEqual(trainer.optimizer.param_groups[0]['lr'], 0.0005)


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling in integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()

    def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        model = create_nca_model(self.config)
        
        # Test with invalid input shape
        invalid_input = torch.randn(2, 5)  # Wrong shape
        
        try:
            outputs = model(invalid_input)
            # If no exception, check that outputs are reasonable
            self.assertIsInstance(outputs, dict)
        except Exception:
            # Exception is expected for invalid input
            pass

    def test_missing_config_handling(self):
        """Test handling of missing configuration."""
        # Test with default configuration
        config = ConfigManager()
        
        # Should not raise exception
        model = create_nca_model(config)
        self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()