"""
Integration tests for NCA Trading Bot components.

Tests the integration between different components of the trading system,
including model training, trading execution, and data handling.
"""

import os
import sys
import unittest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import numpy as np
import pandas as pd
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ConfigManager
from nca_model import create_nca_model, NCATradingModel
from trainer import PPOTrainer, JAXPPOTrainer, TrainingManager
from trader import TradingAgent, TradingEnvironment
from data_handler import DataHandler


class TestModelTrainingIntegration(unittest.TestCase):
    """Test integration between model and training components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.config.training.batch_size = 4  # Small batch for testing
        self.config.training.num_epochs = 2  # Few epochs for testing
        self.model = create_nca_model(self.config)
        self.trainer = PPOTrainer(self.model, self.config)

    def test_model_trainer_integration(self):
        """Test integration between model and trainer."""
        # Create dummy data
        batch_size = 4
        seq_len = 10
        feature_dim = 20

        states = torch.randn(batch_size, seq_len, feature_dim)
        actions = torch.randint(0, 3, (batch_size,))
        old_log_probs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)

        batch = {
            'states': states,
            'actions': actions,
            'log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns
        }

        # Test training step
        metrics = self.trainer.train_step(batch)

        # Check that metrics are returned
        self.assertIn('policy_loss', metrics)
        self.assertIn('value_loss', metrics)
        self.assertIn('entropy', metrics)
        self.assertIn('kl_divergence', metrics)
        self.assertIn('learning_rate', metrics)

        # Check that metrics are numeric
        for key, value in metrics.items():
            self.assertIsInstance(value, float)

    def test_trajectory_collection(self):
        """Test trajectory collection and policy update."""
        # Create mock environment
        env = Mock()
        env.reset.return_value = (np.zeros((10, 20)), {})
        env.step.return_value = (np.zeros((10, 20)), 0.1, False, False, {})

        # Test trajectory collection
        trajectories = self.trainer.collect_trajectories(env, num_episodes=2)

        # Check that trajectories are collected
        self.assertIn('states', trajectories)
        self.assertIn('actions', trajectories)
        self.assertIn('rewards', trajectories)
        self.assertIn('dones', trajectories)
        self.assertIn('log_probs', trajectories)
        self.assertIn('values', trajectories)

        # Check that trajectories have correct shape
        self.assertEqual(trajectories['states'].shape[1], 10)  # sequence length
        self.assertEqual(trajectories['states'].shape[2], 20)  # feature dimension

        # Test policy update
        metrics = self.trainer.update_policy(trajectories)

        # Check that metrics are returned
        self.assertIn('policy_loss', metrics)
        self.assertIn('value_loss', metrics)
        self.assertIn('entropy', metrics)


class TestTradingIntegration(unittest.TestCase):
    """Test integration between trading components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.model = create_nca_model(self.config)
        self.trading_agent = TradingAgent(self.model, self.config)

        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 101,
            'Low': np.random.randn(100).cumsum() + 99,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        })

    def test_trading_agent_decision(self):
        """Test trading agent decision making."""
        # Get decision from agent
        decision = self.trading_agent.make_decision(self.sample_data)

        # Check that decision contains expected fields
        self.assertIn('action', decision)
        self.assertIn('confidence', decision)
        self.assertIn('position_size', decision)
        self.assertIn('stop_loss', decision)
        self.assertIn('take_profit', decision)

        # Check that action is valid
        self.assertIn(decision['action'], ['buy', 'sell', 'hold'])

        # Check that confidence is between 0 and 1
        self.assertGreaterEqual(decision['confidence'], 0)
        self.assertLessEqual(decision['confidence'], 1)

    def test_trading_environment(self):
        """Test trading environment."""
        # Create trading environment
        env = TradingEnvironment(self.sample_data, self.config)

        # Test reset
        state, info = env.reset()
        self.assertEqual(state.shape, (10, 20))  # sequence length x feature dimension

        # Test step
        action = [1, 5]  # Buy with amount 5
        next_state, reward, terminated, truncated, info = env.step(action)

        # Check that step returns expected values
        self.assertEqual(next_state.shape, (10, 20))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)


class TestJAXPPOIntegration(unittest.TestCase):
    """Test integration of JAX PPO trainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.observation_dim = 10
        self.action_dim = 3
        self.jax_ppo = JAXPPOTrainer(self.observation_dim, self.action_dim, self.config)

    def test_jax_ppo_initialization(self):
        """Test JAX PPO trainer initialization."""
        # Check that trainer is initialized
        self.assertIsNotNone(self.jax_ppo)
        self.assertEqual(self.jax_ppo.observation_dim, self.observation_dim)
        self.assertEqual(self.jax_ppo.action_dim, self.action_dim)

    def test_jax_ppo_action(self):
        """Test JAX PPO action selection."""
        # Create dummy observation
        observation = np.random.randn(self.observation_dim)

        # Get action
        action, log_prob, value = self.jax_ppo.get_action(observation)

        # Check that action is valid
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)

        # Check that log_prob and value are numeric
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)

    def test_jax_ppo_experience_storage(self):
        """Test JAX PPO experience storage."""
        # Store a transition
        obs = np.random.randn(self.observation_dim)
        action = 1
        log_prob = 0.5
        value = 0.1
        reward = 1.0
        done = False

        self.jax_ppo.store_transition(obs, action, log_prob, value, reward, done)

        # Check that transition is stored
        self.assertEqual(len(self.jax_ppo.buffer['observations']), 1)
        self.assertEqual(len(self.jax_ppo.buffer['actions']), 1)
        self.assertEqual(len(self.jax_ppo.buffer['log_probs']), 1)
        self.assertEqual(len(self.jax_ppo.buffer['values']), 1)
        self.assertEqual(len(self.jax_ppo.buffer['rewards']), 1)
        self.assertEqual(len(self.jax_ppo.buffer['dones']), 1)


class TestTrainingManagerIntegration(unittest.TestCase):
    """Test integration of training manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.training_manager = TrainingManager(self.config)

    def test_training_manager_initialization(self):
        """Test training manager initialization."""
        # Check that manager is initialized
        self.assertIsNotNone(self.training_manager)
        self.assertEqual(self.training_manager.config, self.config)

    def test_model_creation(self):
        """Test model creation in training manager."""
        # Create model
        model = self.training_manager.create_model()

        # Check that model is created
        self.assertIsNotNone(model)
        self.assertIsInstance(model, NCATradingModel)

    def test_jax_ppo_initialization(self):
        """Test JAX PPO initialization in training manager."""
        # Create model
        self.training_manager.create_model()

        # Initialize JAX PPO
        self.training_manager.initialize_jax_ppo(10, 3)

        # Check that JAX PPO is initialized
        self.assertTrue(self.training_manager.jax_ppo_initialized)
        self.assertIsNotNone(self.training_manager.trainer.jax_ppo_trainer)
        self.assertIsNotNone(self.training_manager.streaming_trainer)

    def test_streaming_action(self):
        """Test streaming action in training manager."""
        # Create model and initialize JAX PPO
        self.training_manager.create_model()
        self.training_manager.initialize_jax_ppo(10, 3)

        # Get streaming action
        observation = np.random.randn(10)
        action, metadata = self.training_manager.get_streaming_action(observation)

        # Check that action is valid
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 3)

        # Check that metadata is returned
        self.assertIsInstance(metadata, dict)
        self.assertIn('log_prob', metadata)
        self.assertIn('value', metadata)


class TestDataHandlingIntegration(unittest.TestCase):
    """Test integration between data handling components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.data_handler = DataHandler()

    def test_data_handler_initialization(self):
        """Test data handler initialization."""
        # Check that handler is initialized
        self.assertIsNotNone(self.data_handler)

    def test_technical_indicators_fallback(self):
        """Test technical indicators fallback when libraries are not available."""
        # Create sample data
        data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        })

        # Calculate technical indicators
        indicators = self.data_handler.calculate_technical_indicators(data)

        # Check that indicators are calculated
        self.assertIsInstance(indicators, pd.DataFrame)
        self.assertGreater(len(indicators), 0)

        # Check that expected indicators are present
        expected_indicators = ['RSI', 'MACD', 'BB_upper', 'BB_lower', 'BB_middle']
        for indicator in expected_indicators:
            self.assertIn(indicator, indicators.columns)


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end integration of all components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.config.training.batch_size = 4  # Small batch for testing
        self.config.training.num_epochs = 2  # Few epochs for testing

        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 101,
            'Low': np.random.randn(100).cumsum() + 99,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        })

    def test_full_training_pipeline(self):
        """Test full training pipeline."""
        # Create training manager
        training_manager = TrainingManager(self.config)

        # Create model
        model = training_manager.create_model()

        # Create trading environment
        env = TradingEnvironment(self.sample_data, self.config)

        # Initialize trainer
        trainer = PPOTrainer(model, self.config)

        # Collect trajectories
        trajectories = trainer.collect_trajectories(env, num_episodes=2)

        # Update policy
        metrics = trainer.update_policy(trajectories)

        # Check that training completed
        self.assertIn('policy_loss', metrics)
        self.assertIn('value_loss', metrics)
        self.assertIn('entropy', metrics)

    def test_full_trading_pipeline(self):
        """Test full trading pipeline."""
        # Create model
        model = create_nca_model(self.config)

        # Create trading agent
        trading_agent = TradingAgent(model, self.config)

        # Get decision
        decision = trading_agent.make_decision(self.sample_data)

        # Check that decision is valid
        self.assertIn('action', decision)
        self.assertIn('confidence', decision)
        self.assertIn('position_size', decision)

    def test_model_adaptation(self):
        """Test model adaptation based on performance."""
        # Create model
        model = create_nca_model(self.config)

        # Create trainer
        trainer = PPOTrainer(model, self.config)

        # Initialize JAX PPO
        trainer.initialize_jax_ppo(10, 3)

        # Update performance with poor returns
        poor_returns = [-0.1, -0.2, -0.15, -0.05, -0.1]
        trainer.update_performance_and_adapt(poor_returns)

        # Check that performance is updated
        self.assertGreater(len(trainer.performance_history), 0)


if __name__ == '__main__':
    unittest.main()