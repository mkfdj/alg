"""
Unit tests for trainer module.

Tests training functionality, DDP support, and live training capabilities.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from ..trainer import PPOTrainer, DistributedTrainer, LiveTrainer, TrainingManager
from ..config import ConfigManager
from ..nca_model import NCATradingModel


class TestPPOTrainer(unittest.TestCase):
    """Test cases for PPOTrainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.model = NCATradingModel(self.config)
        self.trainer = PPOTrainer(self.model, self.config)

    def test_initialization(self):
        """Test PPO trainer initialization."""
        self.assertEqual(self.trainer.model, self.model)
        self.assertEqual(self.trainer.config, self.config)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.scheduler)

        # Check PPO parameters
        self.assertEqual(self.trainer.gamma, self.config.training.gamma)
        self.assertEqual(self.trainer.clip_ratio, self.config.training.clip_ratio)
        self.assertEqual(self.trainer.batch_size, self.config.training.batch_size)

    def test_compute_gae(self):
        """Test Generalized Advantage Estimation."""
        batch_size = 10

        # Create sample data
        rewards = torch.randn(batch_size)
        values = torch.randn(batch_size)
        dones = torch.randint(0, 2, (batch_size,)).float()
        next_values = torch.randn(batch_size)

        # Compute GAE
        advantages, returns = self.trainer.compute_gae(rewards, values, dones, next_values)

        # Check shapes
        self.assertEqual(advantages.shape, rewards.shape)
        self.assertEqual(returns.shape, rewards.shape)

        # Check that advantages are normalized
        self.assertAlmostEqual(advantages.mean().item(), 0, places=1)
        self.assertAlmostEqual(advantages.std().item(), 1, places=1)

    def test_compute_ppo_loss(self):
        """Test PPO loss computation."""
        batch_size, seq_len, features = 4, 10, 20

        # Create batch data
        batch = {
            'states': torch.randn(batch_size, seq_len, features),
            'actions': torch.randint(0, 3, (batch_size,)),
            'log_probs': torch.randn(batch_size),
            'advantages': torch.randn(batch_size),
            'returns': torch.randn(batch_size)
        }

        # Compute losses
        losses = self.trainer.compute_ppo_loss(batch)

        # Check loss structure
        expected_keys = ['total_loss', 'policy_loss', 'value_loss', 'entropy', 'kl_divergence']
        for key in expected_keys:
            self.assertIn(key, losses)
            self.assertIsInstance(losses[key], torch.Tensor)

    def test_train_step(self):
        """Test single training step."""
        batch_size, seq_len, features = 4, 10, 20

        # Create batch data
        batch = {
            'states': torch.randn(batch_size, seq_len, features),
            'actions': torch.randint(0, 3, (batch_size,)),
            'log_probs': torch.randn(batch_size),
            'advantages': torch.randn(batch_size),
            'returns': torch.randn(batch_size)
        }

        # Perform training step
        metrics = self.trainer.train_step(batch)

        # Check metrics structure
        expected_keys = ['total_loss', 'policy_loss', 'value_loss', 'entropy', 'learning_rate']
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)

    def test_collect_trajectories(self):
        """Test trajectory collection."""
        # Create mock environment
        mock_env = Mock()
        mock_env.reset.return_value = (torch.randn(10, 5), {})
        mock_env.step.return_value = (torch.randn(10, 5), 0.1, False, False, {})

        # Collect trajectories
        trajectories = self.trainer.collect_trajectories(mock_env, num_episodes=2)

        # Check trajectory structure
        expected_keys = ['states', 'actions', 'rewards', 'dones', 'log_probs', 'values']
        for key in expected_keys:
            self.assertIn(key, trajectories)
            self.assertIsInstance(trajectories[key], torch.Tensor)

    def test_update_policy(self):
        """Test policy update."""
        batch_size, seq_len, features = 8, 10, 20

        # Create trajectory data
        trajectory_data = {
            'states': torch.randn(batch_size, seq_len, features),
            'actions': torch.randint(0, 3, (batch_size,)),
            'rewards': torch.randn(batch_size),
            'dones': torch.randint(0, 2, (batch_size,)).float(),
            'log_probs': torch.randn(batch_size),
            'values': torch.randn(batch_size)
        }

        # Update policy
        metrics = self.trainer.update_policy(trajectory_data)

        # Check metrics structure
        expected_keys = ['policy_loss', 'value_loss', 'entropy', 'kl_divergence']
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)


class TestDistributedTrainer(unittest.TestCase):
    """Test cases for DistributedTrainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.trainer = DistributedTrainer(self.config)

    def test_initialization(self):
        """Test distributed trainer initialization."""
        self.assertEqual(self.trainer.world_size, self.config.training.num_gpus)
        self.assertEqual(self.trainer.local_rank, self.config.training.local_rank)

    @patch('torch.distributed.init_process_group')
    @patch('torch.distributed.is_initialized', return_value=False)
    def test_setup_ddp(self, mock_is_initialized, mock_init_process_group):
        """Test DDP setup."""
        model = NCATradingModel(self.config)

        # Mock DDP
        with patch('torch.nn.parallel.DistributedDataParallel') as mock_ddp:
            mock_ddp_instance = Mock()
            mock_ddp.return_value = mock_ddp_instance

            result = self.trainer.setup_ddp(model)

            if self.config.training.num_gpus > 1:
                mock_ddp.assert_called_once()
                self.assertEqual(result, mock_ddp_instance)
            else:
                self.assertEqual(result, model)

    @patch('torch.distributed.destroy_process_group')
    def test_cleanup_ddp(self, mock_destroy):
        """Test DDP cleanup."""
        self.trainer.cleanup_ddp()
        mock_destroy.assert_called_once()


class TestLiveTrainer(unittest.TestCase):
    """Test cases for LiveTrainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.model = NCATradingModel(self.config)
        self.trainer = LiveTrainer(self.model, self.config)

    def test_initialization(self):
        """Test live trainer initialization."""
        self.assertEqual(self.trainer.model, self.model)
        self.assertEqual(self.trainer.config, self.config)
        self.assertFalse(self.trainer.is_training)
        self.assertIsNone(self.trainer.trading_agent)

    def test_evaluate_trade_outcome(self):
        """Test trade outcome evaluation."""
        trade = {
            'decision': {'expected_profit': 100}
        }

        outcome = self.trainer._evaluate_trade_outcome(trade)
        self.assertEqual(outcome, 1.0)  # Profitable trade

        # Test losing trade
        trade['decision']['expected_profit'] = -50
        outcome = self.trainer._evaluate_trade_outcome(trade)
        self.assertEqual(outcome, -1.0)  # Losing trade

    def test_extract_features(self):
        """Test feature extraction."""
        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'Open': np.random.randn(10) + 100,
            'High': np.random.randn(10) + 101,
            'Low': np.random.randn(10) + 99,
            'Close': np.random.randn(10) + 100,
            'Volume': np.random.randint(1000, 10000, 10),
            'RSI': np.random.randn(10),
            'MACD': np.random.randn(10)
        })

        features = self.trainer._extract_features(market_data)

        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 12)  # 10 features + RSI + MACD

    def test_evaluate_recent_performance(self):
        """Test recent performance evaluation."""
        # Mock trading agent
        mock_agent = Mock()
        mock_agent.get_performance_report.return_value = {
            'total_pnl': 1000,
            'win_rate': 0.6,
            'total_trades': 50
        }

        self.trainer.trading_agent = mock_agent
        performance = self.trainer._evaluate_recent_performance()

        expected_keys = ['reward', 'win_rate', 'total_trades']
        for key in expected_keys:
            self.assertIn(key, performance)
            self.assertIsInstance(performance[key], (int, float))


class TestTrainingManager(unittest.TestCase):
    """Test cases for TrainingManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.manager = TrainingManager(self.config)

    def test_initialization(self):
        """Test training manager initialization."""
        self.assertEqual(self.manager.config, self.config)
        self.assertIsNone(self.manager.model)
        self.assertIsNone(self.manager.trainer)
        self.assertFalse(self.manager.is_training)

        # Check directory creation
        self.assertTrue(self.manager.model_dir.exists())
        self.assertTrue(self.manager.log_dir.exists())

    def test_create_model(self):
        """Test model creation."""
        model = self.manager.create_model()

        self.assertIsInstance(model, NCATradingModel)
        self.assertEqual(self.manager.model, model)
        self.assertEqual(model.state_dim, self.config.nca.state_dim)

    def test_save_load_model(self):
        """Test model saving and loading."""
        # Create model
        model = self.manager.create_model()

        # Save model
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name

        try:
            self.manager.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))

            # Load model
            loaded_model = self.manager.load_model(model_path)
            self.assertIsInstance(loaded_model, NCATradingModel)

        finally:
            os.unlink(model_path)

    def test_log_metrics(self):
        """Test metrics logging."""
        metrics = {
            'loss': 0.5,
            'accuracy': 0.85,
            'reward': 100
        }

        # Should not raise exception
        self.manager._log_metrics(metrics, step=10)

    def test_get_training_summary(self):
        """Test training summary generation."""
        summary = self.manager.get_training_summary()

        expected_keys = ['is_training', 'current_mode', 'model_path', 'log_path', 'metrics']
        for key in expected_keys:
            self.assertIn(key, summary)

        self.assertFalse(summary['is_training'])
        self.assertIsNone(summary['current_mode'])


if __name__ == '__main__':
    unittest.main()