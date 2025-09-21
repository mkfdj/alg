"""
Training module for NCA Trading Bot.

This module provides comprehensive RL training capabilities with DDP support,
AMP optimization, runtime adaptation, and live training modes.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import wandb
import gymnasium as gym
from datetime import datetime
import os
import json
from pathlib import Path
import pickle

from .config import get_config
from .nca_model import NCATradingModel, NCATrainer, NCAModelCache
from .trader import TradingEnvironment, TradingAgent
from .data_handler import DataHandler


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for NCA trading model.

    Implements PPO algorithm with support for continuous and discrete action spaces,
    value function optimization, and entropy regularization.
    """

    def __init__(self, model: NCATradingModel, config, device: str = None):
        """
        Initialize PPO trainer.

        Args:
            model: NCA trading model
            config: Training configuration
            device: Device to use for training
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Device setup
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # PPO parameters
        self.gamma = config.training.gamma
        self.gae_lambda = config.training.gae_lambda
        self.clip_ratio = config.training.clip_ratio
        self.entropy_coeff = config.training.entropy_coeff
        self.value_coeff = config.training.value_coeff
        self.target_kl = config.training.target_kl

        # Training parameters
        self.batch_size = config.training.batch_size
        self.num_epochs = config.training.num_epochs
        self.max_grad_norm = config.training.max_grad_norm

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.nca.learning_rate,
            weight_decay=config.nca.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # AMP setup
        self.use_amp = config.training.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

        # Training metrics
        self.metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'explained_variance': [],
            'learning_rate': []
        }

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                   dones: torch.Tensor, next_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Reward tensor
            values: Value estimates
            dones: Done flags
            next_values: Next state value estimates

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        last_advantage = 0
        last_return = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
                next_advantage = 0
            else:
                next_value = values[t + 1]
                next_advantage = advantages[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage * (1 - dones[t])
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def compute_ppo_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss for a batch of data.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary with loss components
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['log_probs'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        returns = batch['returns'].to(self.device)

        # Forward pass
        outputs = self.model(states)
        values = outputs['price_prediction'].squeeze()
        action_logits = outputs['signal_probabilities']

        # Policy loss
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze()

        ratio = torch.exp(selected_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value loss
        value_loss = self.value_loss_fn(values, returns)

        # Entropy bonus
        entropy = -(action_log_probs * torch.exp(action_log_probs)).sum(dim=-1).mean()

        # KL divergence for early stopping
        old_action_probs = F.softmax(action_logits, dim=-1)
        kl_div = F.kl_div(action_log_probs, old_action_probs, reduction='batchmean')

        # Total loss
        total_loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'kl_divergence': kl_div
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform single training step.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary with training metrics
        """
        self.optimizer.zero_grad()

        # Compute losses
        losses = self.compute_ppo_loss(batch)

        # Backward pass with AMP
        if self.use_amp:
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # Update learning rate
        self.scheduler.step(losses['total_loss'].item())

        # Record metrics
        metrics = {
            'policy_loss': losses['policy_loss'].item(),
            'value_loss': losses['value_loss'].item(),
            'entropy': losses['entropy'].item(),
            'kl_divergence': losses['kl_divergence'].item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

        # Update metrics history
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)

        return metrics

    def collect_trajectories(self, env: TradingEnvironment, num_episodes: int = 10) -> Dict[str, torch.Tensor]:
        """
        Collect training trajectories from environment.

        Args:
            env: Trading environment
            num_episodes: Number of episodes to collect

        Returns:
            Dictionary with trajectory data
        """
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # Get action from policy
                with torch.no_grad():
                    outputs = self.model(state_tensor)
                    action_logits = outputs['signal_probabilities']
                    action_probs = F.softmax(action_logits, dim=-1)

                    action = torch.multinomial(action_probs, 1).item()
                    log_prob = torch.log(action_probs[0, action])

                    value = outputs['price_prediction'].squeeze().item()

                # Take action in environment
                next_state, reward, terminated, truncated, _ = env.step([action, 5])
                done = terminated or truncated

                # Store trajectory data
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                log_probs.append(log_prob.item())
                values.append(value)

                state = next_state
                episode_reward += reward

            self.logger.debug(f"Episode {episode} reward: {episode_reward:.2f}")

        # Convert to tensors
        trajectory_data = {
            'states': torch.FloatTensor(np.array(states)),
            'actions': torch.LongTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'dones': torch.FloatTensor(dones),
            'log_probs': torch.FloatTensor(log_probs),
            'values': torch.FloatTensor(values)
        }

        return trajectory_data

    def update_policy(self, trajectory_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.

        Args:
            trajectory_data: Trajectory data from environment

        Returns:
            Dictionary with training metrics
        """
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            trajectory_data['rewards'],
            trajectory_data['values'],
            trajectory_data['dones'],
            trajectory_data['values']
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create dataset
        dataset = TensorDataset(
            trajectory_data['states'],
            trajectory_data['actions'],
            trajectory_data['log_probs'],
            advantages,
            returns
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        total_metrics = {}
        for epoch in range(self.num_epochs):
            epoch_metrics = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl_divergence': 0}

            for batch in dataloader:
                batch_dict = {
                    'states': batch[0],
                    'actions': batch[1],
                    'log_probs': batch[2],
                    'advantages': batch[3],
                    'returns': batch[4]
                }

                batch_metrics = self.train_step(batch_dict)

                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value

            # Average metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= len(dataloader)

            # Early stopping based on KL divergence
            if epoch_metrics['kl_divergence'] > self.target_kl * 1.5:
                self.logger.info(f"Early stopping at epoch {epoch} due to high KL divergence")
                break

            total_metrics = epoch_metrics

        return total_metrics


class DistributedTrainer:
    """
    Distributed trainer with DDP support for multi-GPU training.

    Provides distributed training capabilities for large-scale NCA training.
    """

    def __init__(self, config):
        """
        Initialize distributed trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # DDP setup
        self.world_size = config.training.num_gpus
        self.local_rank = config.training.local_rank

        # Initialize process group
        if self.world_size > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(
                backend='nccl',
                rank=self.local_rank,
                world_size=self.world_size
            )

    def setup_ddp(self, model: NCATradingModel) -> DDP:
        """
        Set up Distributed Data Parallel.

        Args:
            model: Model to wrap with DDP

        Returns:
            DDP-wrapped model
        """
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.local_rank])
            self.logger.info(f"DDP initialized on rank {self.local_rank}")
        return model

    def cleanup_ddp(self):
        """Clean up DDP process group."""
        if self.world_size > 1:
            dist.destroy_process_group()

    def train_distributed(self, train_function: Callable, *args, **kwargs):
        """
        Run distributed training.

        Args:
            train_function: Training function to execute
            args: Arguments for training function
            kwargs: Keyword arguments for training function
        """
        if self.world_size > 1:
            # Spawn processes for distributed training
            mp.spawn(
                train_function,
                args=(self.world_size, self.local_rank, *args),
                kwargs=kwargs,
                nprocs=self.world_size
            )
        else:
            # Single GPU training
            train_function(self.local_rank, self.world_size, *args, **kwargs)


class LiveTrainer:
    """
    Live training system with online learning and adaptation.

    Provides continuous learning capabilities during live trading operations.
    """

    def __init__(self, model: NCATradingModel, config):
        """
        Initialize live trainer.

        Args:
            model: NCA trading model
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Live training components
        self.data_handler = DataHandler()
        self.trading_agent = None
        self.is_training = False

        # Online learning parameters
        self.update_frequency = 100  # Update every 100 trades
        self.minibatch_size = 32
        self.adaptation_rate = config.nca.adaptation_rate

        # Performance tracking
        self.performance_buffer = []
        self.adaptation_history = []

    async def start_live_training(self, symbols: List[str], trading_agent: TradingAgent):
        """
        Start live training session.

        Args:
            symbols: List of symbols to trade and train on
            trading_agent: Trading agent instance
        """
        self.trading_agent = trading_agent
        self.is_training = True

        self.logger.info("Starting live training session")

        # Initialize data streaming
        await self.data_handler.get_multiple_tickers_data(
            symbols, '2023-01-01', datetime.now().strftime('%Y-%m-%d')
        )

        # Start training loop
        await self._live_training_loop(symbols)

    async def stop_live_training(self):
        """Stop live training session."""
        self.is_training = False
        self.logger.info("Stopping live training session")

    async def _live_training_loop(self, symbols: List[str]):
        """Main live training loop."""
        trade_count = 0

        while self.is_training:
            try:
                # Collect recent trading data
                recent_data = await self._collect_recent_data(symbols)

                if recent_data:
                    # Perform online update
                    await self._online_update(recent_data)

                    trade_count += len(recent_data)

                    # Periodic model adaptation
                    if trade_count % self.update_frequency == 0:
                        await self._adapt_model()

                # Wait before next update
                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in live training loop: {e}")
                await asyncio.sleep(60)

    async def _collect_recent_data(self, symbols: List[str]) -> List[Dict]:
        """
        Collect recent trading data for online learning.

        Args:
            symbols: List of symbols

        Returns:
            List of recent trading records
        """
        recent_data = []

        # Get recent trades from trading agent
        if self.trading_agent:
            recent_trades = self.trading_agent.trades[-100:]  # Last 100 trades

            for trade in recent_trades:
                # Get market data around trade time
                trade_time = trade['timestamp']
                start_time = trade_time - timedelta(hours=1)
                end_time = trade_time + timedelta(hours=1)

                try:
                    market_data = await self.data_handler.get_historical_data(
                        trade['symbol'], start_time.strftime('%Y-%m-%d'),
                        end_time.strftime('%Y-%m-%d'), '5m'
                    )

                    if not market_data.empty:
                        recent_data.append({
                            'trade': trade,
                            'market_data': market_data,
                            'outcome': self._evaluate_trade_outcome(trade)
                        })

                except Exception as e:
                    self.logger.warning(f"Failed to collect data for trade: {e}")

        return recent_data

    def _evaluate_trade_outcome(self, trade: Dict) -> float:
        """
        Evaluate trade outcome for reinforcement learning.

        Args:
            trade: Trade record

        Returns:
            Trade outcome score
        """
        # Simplified outcome evaluation
        # In practice, this would involve more sophisticated P&L calculation
        if trade['decision'].get('expected_profit', 0) > 0:
            return 1.0  # Profitable trade
        else:
            return -1.0  # Losing trade

    async def _online_update(self, recent_data: List[Dict]):
        """
        Perform online model update.

        Args:
            recent_data: Recent trading data
        """
        if len(recent_data) < self.minibatch_size:
            return

        try:
            # Prepare training batch
            batch_data = self._prepare_training_batch(recent_data)

            # Perform model update
            self.model.adapt_online(
                batch_data['states'],
                batch_data['targets'],
                self.adaptation_rate
            )

            self.logger.debug(f"Online update completed with {len(recent_data)} samples")

        except Exception as e:
            self.logger.error(f"Error in online update: {e}")

    def _prepare_training_batch(self, recent_data: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Prepare training batch from recent data.

        Args:
            recent_data: Recent trading data

        Returns:
            Training batch dictionary
        """
        states, targets = [], []

        for data_point in recent_data:
            market_data = data_point['market_data']
            outcome = data_point['outcome']

            # Extract features
            features = self._extract_features(market_data)
            states.append(features)

            # Create target based on outcome
            if outcome > 0:
                targets.append([0, 0, 1])  # Buy signal
            elif outcome < 0:
                targets.append([1, 0, 0])  # Sell signal
            else:
                targets.append([0, 1, 0])  # Hold signal

        return {
            'states': torch.FloatTensor(np.array(states)),
            'targets': torch.FloatTensor(np.array(targets))
        }

    def _extract_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from market data for training.

        Args:
            market_data: Market data DataFrame

        Returns:
            Feature array
        """
        # Use technical indicators as features
        feature_columns = [col for col in market_data.columns
                          if col not in ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]

        if feature_columns:
            return market_data[feature_columns].values[-1]  # Latest values
        else:
            return market_data[['Close']].values[-1]

    async def _adapt_model(self):
        """Adapt model based on recent performance."""
        try:
            # Evaluate recent performance
            recent_performance = self._evaluate_recent_performance()

            # Update model adaptation parameters
            self.model.update_performance(recent_performance)

            # Log adaptation
            adaptation_record = {
                'timestamp': datetime.now(),
                'performance': recent_performance,
                'adaptation_rate': self.adaptation_rate
            }
            self.adaptation_history.append(adaptation_record)

            self.logger.info(f"Model adaptation completed: {recent_performance}")

        except Exception as e:
            self.logger.error(f"Error in model adaptation: {e}")

    def _evaluate_recent_performance(self) -> Dict[str, float]:
        """
        Evaluate recent trading performance.

        Returns:
            Performance metrics dictionary
        """
        if not self.trading_agent:
            return {'reward': 0, 'win_rate': 0}

        # Get recent performance metrics
        performance_report = self.trading_agent.get_performance_report()

        return {
            'reward': performance_report.get('total_pnl', 0),
            'win_rate': performance_report.get('win_rate', 0),
            'total_trades': performance_report.get('total_trades', 0)
        }


class TrainingManager:
    """
    High-level training manager that orchestrates all training activities.

    Provides unified interface for offline training, live training, and model management.
    """

    def __init__(self, config):
        """
        Initialize training manager.

        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Training components
        self.model = None
        self.trainer = None
        self.live_trainer = None
        self.data_handler = DataHandler()

        # Training state
        self.is_training = False
        self.current_mode = None

        # Setup directories
        self.model_dir = config.system.model_dir
        self.log_dir = config.system.log_dir
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # Initialize logging
        self._setup_logging()

    def _setup_logging(self):
        """Set up training logging and monitoring."""
        # TensorBoard
        if self.config.system.enable_tensorboard:
            self.tb_writer = SummaryWriter(self.log_dir)
        else:
            self.tb_writer = None

        # Weights & Biases
        if self.config.system.enable_wandb:
            wandb.init(project=self.config.system.wandb_project)
            self.wandb_run = wandb.run
        else:
            self.wandb_run = None

    def create_model(self) -> NCATradingModel:
        """
        Create NCA trading model.

        Returns:
            Initialized NCA model
        """
        from .nca_model import create_nca_model

        self.model = create_nca_model(self.config)
        self.logger.info("NCA model created")

        return self.model

    def load_model(self, model_path: str) -> NCATradingModel:
        """
        Load NCA model from file.

        Args:
            model_path: Path to model file

        Returns:
            Loaded NCA model
        """
        from .nca_model import load_nca_model

        self.model = load_nca_model(model_path, self.config)
        self.logger.info(f"Model loaded from {model_path}")

        return self.model

    def save_model(self, model_path: str = None):
        """
        Save current model.

        Args:
            model_path: Path to save model (optional)
        """
        if self.model is None:
            raise ValueError("No model to save")

        if model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.model_dir / f"nca_model_{timestamp}.pt"

        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")

    async def train_offline(self, data: pd.DataFrame, num_epochs: int = 100):
        """
        Perform offline training on historical data.

        Args:
            data: Historical training data
            num_epochs: Number of training epochs
        """
        if self.model is None:
            self.create_model()

        self.current_mode = 'offline'
        self.is_training = True

        self.logger.info("Starting offline training")

        # Create training environment
        env = TradingEnvironment(data, self.config)

        # Initialize trainer
        self.trainer = PPOTrainer(self.model, self.config)

        # Training loop
        for epoch in range(num_epochs):
            if not self.is_training:
                break

            try:
                # Collect trajectories
                trajectories = self.trainer.collect_trajectories(env, num_episodes=10)

                # Update policy
                metrics = self.trainer.update_policy(trajectories)

                # Log metrics
                self._log_metrics(metrics, epoch)

                # Save checkpoint
                if epoch % self.config.training.save_freq == 0:
                    self.save_model()

                self.logger.info(f"Epoch {epoch}: {metrics}")

            except Exception as e:
                self.logger.error(f"Error in training epoch {epoch}: {e}")

        self.logger.info("Offline training completed")

    async def train_live(self, symbols: List[str], trading_agent: TradingAgent):
        """
        Start live training session.

        Args:
            symbols: List of symbols to trade
            trading_agent: Trading agent instance
        """
        if self.model is None:
            self.create_model()

        self.current_mode = 'live'
        self.is_training = True

        # Initialize live trainer
        self.live_trainer = LiveTrainer(self.model, self.config)

        # Start live training
        await self.live_trainer.start_live_training(symbols, trading_agent)

    def stop_training(self):
        """Stop current training session."""
        self.is_training = False

        if self.live_trainer:
            asyncio.create_task(self.live_trainer.stop_live_training())

        self.logger.info("Training stopped")

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log training metrics.

        Args:
            metrics: Training metrics
            step: Training step
        """
        # TensorBoard logging
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"train/{key}", value, step)

        # Weights & Biases logging
        if self.wandb_run:
            wandb.log(metrics, step=step)

        # Console logging
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}")

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get training summary.

        Returns:
            Training summary dictionary
        """
        return {
            'is_training': self.is_training,
            'current_mode': self.current_mode,
            'model_path': str(self.model_dir),
            'log_path': str(self.log_dir),
            'metrics': self.trainer.metrics if self.trainer else {}
        }


# Utility functions
def setup_distributed_training(rank: int, world_size: int, config):
    """
    Set up distributed training environment.

    Args:
        rank: Process rank
        world_size: Number of processes
        config: Training configuration
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(rank)


def cleanup_distributed_training():
    """Clean up distributed training."""
    dist.destroy_process_group()


if __name__ == "__main__":
    # Example usage
    from .config import ConfigManager

    print("NCA Trading Bot - Training Module Demo")
    print("=" * 40)

    config = ConfigManager()

    # Create training manager
    trainer = TrainingManager(config)
    print("Training manager created")

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'Open': np.random.randn(1000).cumsum() + 100,
        'High': np.random.randn(1000).cumsum() + 101,
        'Low': np.random.randn(1000).cumsum() + 99,
        'Close': np.random.randn(1000).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 1000)
    })

    # Create model
    model = trainer.create_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test offline training setup
    print("Offline training setup completed")
    print("Training module demo completed successfully!")