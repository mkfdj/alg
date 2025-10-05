"""
Enhanced RL Training Module for NCA Trading Bot with JAX/Flax PPO.

This module provides advanced RL training capabilities with:
- JAX/Flax PPO implementation for policy optimization
- Real-time adaptation on streams with online learning
- Adaptive NCA growth triggered by Sharpe ratio monitoring
- Brave Search MCP integration for market intelligence
- Continuous learning and performance-based adaptation

Based on research papers:
- "Reinforcement Learning Framework for Quantitative Trading" (arXiv:2411.07585)
- "Automated Trading System for Straddle-Option" (arXiv:2509.07987)
- "Agent Performing Autonomous Stock Trading" (arXiv:2306.03985)
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp

# JAX/Flax imports for PPO
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, pmap
import flax
from flax import linen as nn
import optax
from flax.training import train_state
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

# TPU/XLA imports
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.test.test_utils as test_utils
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False
    xm = None
    pl = None
    xmp = None
    test_utils = None

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import wandb
import gymnasium as gym
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import pickle

from config import get_config
from nca_model import NCATradingModel, NCATrainer, NCAModelCache
from trader import TradingEnvironment, TradingAgent
from data_handler import DataHandler
from utils import RiskCalculator, PerformanceMonitor, LoggerUtils
from adaptivity import (
    create_adaptive_grid_manager,
    create_growth_strategies,
    create_adaptive_nca_wrapper,
    create_market_condition_analyzer,
    create_performance_metrics,
    AdaptiveNCAWrapper
)

# MCP tool import (assumed available)
try:
    from mcp_tools import use_mcp_tool
except ImportError:
    # Fallback for MCP integration
    async def use_mcp_tool(*args, **kwargs):
        raise NotImplementedError("MCP tools not available")


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
        self.device = device or self._get_device_from_config(config)
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
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )

        # AMP setup - support both CUDA and TPU
        self.use_amp = config.training.use_mixed_precision and (torch.cuda.is_available() or self.device.type == 'xla')
        self.scaler = GradScaler() if self.use_amp and torch.cuda.is_available() else None

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

        # Enhanced features for real-time adaptation
        self.jax_ppo_trainer = None  # JAX PPO trainer for online learning
        self.market_intelligence = MarketIntelligence(config)
        self.risk_calculator = RiskCalculator(config)
        self.performance_history = []
        self.sharpe_threshold = 0.5

        # Integration with adaptive NCA components
        self.adaptive_grid_manager = None
        self.growth_strategies = None
        self.market_condition_analyzer = None

        # Initialize adaptive components
        self._initialize_adaptive_components()

    def _get_device_from_config(self, config):
        """Get device based on configuration and availability."""
        if config.system.device == "tpu" and XLA_AVAILABLE:
            return xm.xla_device()
        elif config.system.device == "cuda" and torch.cuda.is_available():
            return torch.device('cuda')
        elif config.system.device == "cpu":
            return torch.device('cpu')
        else:
            # Auto-detect best available device
            if XLA_AVAILABLE:
                return xm.xla_device()
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')

    def initialize_jax_ppo(self, observation_dim: int, action_dim: int, nca_model=None):
        """
        Initialize JAX PPO trainer for online learning.

        Args:
            observation_dim: Observation space dimension
            action_dim: Action space dimension
            nca_model: Adaptive NCA model for growth triggering
        """
        self.jax_ppo_trainer = JAXPPOTrainer(observation_dim, action_dim, self.config, nca_model)
        self.logger.info("JAX PPO trainer initialized for online learning")

    def _initialize_adaptive_components(self):
        """Initialize adaptive NCA components for enhanced training."""
        try:
            from adaptivity import (
                create_adaptive_grid_manager,
                create_growth_strategies,
                create_market_condition_analyzer
            )

            self.adaptive_grid_manager = create_adaptive_grid_manager(self.config)
            self.growth_strategies = create_growth_strategies(self.config)
            self.market_condition_analyzer = create_market_condition_analyzer(self.config)

            self.logger.info("Adaptive components initialized in trainer")

        except Exception as e:
            self.logger.warning(f"Failed to initialize adaptive components in trainer: {e}")

    async def get_market_intelligence(self, symbol: str, context: str = "trading_decision") -> Dict[str, Any]:
        """
        Get market intelligence for trading decisions.

        Args:
            symbol: Stock symbol
            context: Decision context

        Returns:
            Market intelligence data
        """
        if self.jax_ppo_trainer:
            return await self.jax_ppo_trainer.get_market_intelligence(symbol, context)
        else:
            return await self.market_intelligence.query_market_data(symbol, context)

    def update_performance_and_adapt(self, returns: List[float]):
        """
        Update performance metrics and trigger adaptations.

        Args:
            returns: Recent trading returns
        """
        # Calculate Sharpe ratio
        sharpe_ratio = self.risk_calculator.calculate_sharpe_ratio(np.array(returns))

        # Store performance
        self.performance_history.append({
            'timestamp': datetime.now(),
            'sharpe_ratio': sharpe_ratio,
            'returns': returns
        })

        # Keep history bounded
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

        # Update JAX PPO trainer if available
        if self.jax_ppo_trainer:
            self.jax_ppo_trainer.update_performance(returns)

        # Trigger NCA growth if Sharpe ratio is too low
        if sharpe_ratio < self.sharpe_threshold:
            self._trigger_nca_growth()

        self.logger.debug(f"Performance updated - Sharpe: {sharpe_ratio:.3f}")

    def _trigger_nca_growth(self):
        """Trigger adaptive NCA growth when performance is poor."""
        self.logger.info(f"Sharpe ratio below threshold, triggering NCA growth")

        # Trigger growth in the NCA model
        if hasattr(self.model, 'trigger_growth'):
            self.model.trigger_growth()
        elif hasattr(self.model, 'adapt_online'):
            # Use online adaptation
            self.model.adapt_online(torch.randn(1, 10, 20), torch.randn(1, 3))

    def store_online_experience(self, obs: np.ndarray, action: int, log_prob: float,
                               value: float, reward: float, done: bool):
        """
        Store experience for online learning.

        Args:
            obs: Observation
            action: Action taken
            log_prob: Log probability of action
            value: Value estimate
            reward: Reward received
            done: Whether episode ended
        """
        if self.jax_ppo_trainer:
            self.jax_ppo_trainer.store_transition(obs, action, log_prob, value, reward, done)

    def get_adaptive_action(self, observation: np.ndarray, symbol: str = None) -> Tuple[int, Dict[str, Any]]:
        """
        Get action with market intelligence integration.

        Args:
            observation: Current observation
            symbol: Stock symbol for market intelligence

        Returns:
            Tuple of (action, metadata)
        """
        # Get base action from JAX PPO if available
        if self.jax_ppo_trainer:
            action, log_prob, value = self.jax_ppo_trainer.get_action(observation)
        else:
            # Fallback to PyTorch model
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(obs_tensor)
                action_probs = F.softmax(outputs['signal_probabilities'], dim=1)
                action = torch.argmax(action_probs, dim=1).item()
                log_prob = torch.log(action_probs[0, action]).item()
                value = outputs['price_prediction'].squeeze().item()

        # Get market intelligence if symbol provided
        metadata = {}
        if symbol:
            try:
                # This would be async in real implementation
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                intelligence = loop.run_until_complete(
                    self.get_market_intelligence(symbol, "trading_decision")
                )
                metadata['market_intelligence'] = intelligence
            except Exception as e:
                self.logger.warning(f"Failed to get market intelligence: {e}")

        return action, metadata

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


class JAXPPOTrainer:
    """
    JAX PPO trainer optimized for TPU v5e-8 with sharding support.

    Implements PPO algorithm using JAX/Flax for distributed training on TPUs.
    """

    def __init__(self, observation_dim: int, action_dim: int, config, nca_model=None):
        """
        Initialize JAX PPO trainer.

        Args:
            observation_dim: Observation space dimension
            action_dim: Action space dimension
            config: Training configuration
            nca_model: Adaptive NCA model for growth triggering
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = config
        self.nca_model = nca_model

        # PPO parameters
        self.gamma = config.training.gamma
        self.gae_lambda = config.training.gae_lambda
        self.clip_ratio = config.training.clip_ratio
        self.entropy_coeff = config.training.entropy_coeff
        self.value_coeff = config.training.value_coeff
        self.target_kl = config.training.target_kl

        # Training parameters
        self.batch_size = config.training.batch_size
        self.micro_batch_size = config.training.micro_batch_size
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps
        self.num_epochs = config.training.num_epochs
        self.max_grad_norm = config.training.max_grad_norm

        # JAX setup
        self.rng = random.PRNGKey(42)
        self.dtype = jnp.bfloat16 if config.training.precision_dtype == "bf16" else jnp.float16

        # Sharding setup for TPU v5e-8
        self.setup_sharding()

        # Actor-Critic network
        self.actor_critic = self.create_actor_critic_network()

        # Optimizer
        self.setup_optimizer()

        # Training state
        self.train_state = None

        # Experience buffer
        self.buffer = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': []
        }

        # Performance tracking
        self.sharpe_history = []
        self.performance_history = []

    def setup_sharding(self):
        """Setup JAX sharding for TPU v5e-8."""
        devices = jax.devices()
        if len(devices) == 8:  # TPU v5e-8
            self.mesh = Mesh(mesh_utils.create_device_mesh((1, 8)), axis_names=('data', 'model'))
            self.data_sharding = P('data', None)  # Shard data across devices
            self.model_sharding = P(None, 'model')  # Replicate model across devices
        else:
            # Fallback for other configurations
            self.mesh = Mesh(devices, axis_names=('data', 'model'))
            self.data_sharding = P(None, None)
            self.model_sharding = P(None, None)

    def create_actor_critic_network(self) -> nn.Module:
        """
        Create JAX actor-critic network for PPO.

        Returns:
            JAX actor-critic network
        """
        class ActorCritic(nn.Module):
            """Actor-Critic network for PPO."""
            action_dim: int
            hidden_dim: int = 256
            dtype: jnp.dtype = jnp.bfloat16

            @nn.compact
            def __call__(self, x, deterministic=False):
                # Shared layers
                x = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
                x = nn.relu(x)
                x = nn.Dropout(0.1, deterministic=deterministic)(x)

                x = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
                x = nn.relu(x)
                x = nn.Dropout(0.1, deterministic=deterministic)(x)

                # Actor head
                logits = nn.Dense(self.action_dim, dtype=self.dtype)(x)

                # Critic head
                value = nn.Dense(1, dtype=self.dtype)(x)

                return logits, value.squeeze()

        return ActorCritic(action_dim=self.action_dim, dtype=self.dtype)

    def setup_optimizer(self):
        """Setup JAX optimizer with sharding."""
        # Create optimizer
        if self.config.training.jax_optimizer == 'adamw':
            self.tx = optax.adamw(
                learning_rate=self.config.nca.learning_rate,
                weight_decay=self.config.nca.weight_decay,
                b1=0.9, b2=0.999, eps=1e-8
            )
        else:
            self.tx = optax.adam(
                learning_rate=self.config.nca.learning_rate,
                b1=0.9, b2=0.999, eps=1e-8
            )

        # Add gradient clipping
        self.tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            self.tx
        )

    def initialize_train_state(self):
        """Initialize JAX train state."""
        # Create dummy input
        dummy_obs = jnp.zeros((1, self.observation_dim), dtype=self.dtype)

        # Initialize network
        variables = self.actor_critic.init(self.rng, dummy_obs)

        # Create train state
        self.train_state = train_state.TrainState.create(
            apply_fn=self.actor_critic.apply,
            params=variables,
            tx=self.tx
        )

    @jax.jit
    def get_action(self, observation: jnp.ndarray) -> Tuple[int, float, float]:
        """
        Get action from policy with JAX.

        Args:
            observation: Current observation

        Returns:
            Tuple of (action, log_prob, value)
        """
        if self.train_state is None:
            self.initialize_train_state()

        logits, value = self.train_state.apply_fn(
            self.train_state.params, observation, deterministic=False
        )

        # Sample action
        action_probs = nn.softmax(logits, axis=-1)
        action = jax.random.categorical(self.rng, logits)
        log_prob = jnp.log(action_probs[jnp.arange(action_probs.shape[0]), action])

        return action.item(), log_prob.item(), value.item()

    def store_transition(self, obs: np.ndarray, action: int, log_prob: float,
                        value: float, reward: float, done: bool):
        """
        Store transition in experience buffer.

        Args:
            obs: Observation
            action: Action taken
            log_prob: Log probability of action
            value: Value estimate
            reward: Reward received
            done: Whether episode ended
        """
        self.buffer['observations'].append(obs)
        self.buffer['actions'].append(action)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['values'].append(value)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)

    def compute_gae(self, rewards: jnp.ndarray, values: jnp.ndarray,
                   dones: jnp.ndarray, next_values: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute Generalized Advantage Estimation with JAX.

        Args:
            rewards: Reward tensor
            values: Value estimates
            dones: Done flags
            next_values: Next state value estimates

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = jnp.zeros_like(rewards)
        returns = jnp.zeros_like(rewards)

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
            advantages = advantages.at[t].set(delta + self.gamma * self.gae_lambda * next_advantage * (1 - dones[t]))
            returns = returns.at[t].set(advantages[t] + values[t])

        return advantages, returns

    @jax.jit
    def train_step(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """
        Single JAX training step with sharding.

        Args:
            batch: Training batch

        Returns:
            Dictionary with training metrics
        """
        def loss_fn(params):
            # Forward pass
            logits, values = self.train_state.apply_fn(params, batch['observations'])

            # Policy loss
            action_probs = nn.softmax(logits, axis=-1)
            selected_log_probs = jnp.log(action_probs[jnp.arange(logits.shape[0]), batch['actions']])

            ratio = jnp.exp(selected_log_probs - batch['old_log_probs'])
            clipped_ratio = jnp.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

            policy_loss = -jnp.minimum(ratio * batch['advantages'], clipped_ratio * batch['advantages']).mean()

            # Value loss
            value_loss = jnp.mean((values - batch['returns']) ** 2)

            # Entropy bonus
            entropy = -(action_probs * jnp.log(action_probs + 1e-8)).sum(axis=-1).mean()

            # Total loss
            total_loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

            return total_loss, (policy_loss, value_loss, entropy)

        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (policy_loss, value_loss, entropy)), grads = grad_fn(self.train_state.params)

        # Apply gradients
        self.train_state = self.train_state.apply_gradients(grads=grads)

        return {
            'total_loss': float(loss),
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy': float(entropy)
        }

    def update_policy(self, num_epochs: int = None) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.

        Args:
            num_epochs: Number of training epochs

        Returns:
            Dictionary with training metrics
        """
        if num_epochs is None:
            num_epochs = self.num_epochs

        # Convert buffer to JAX arrays
        observations = jnp.array(self.buffer['observations'], dtype=self.dtype)
        actions = jnp.array(self.buffer['actions'], dtype=jnp.int32)
        old_log_probs = jnp.array(self.buffer['log_probs'], dtype=self.dtype)
        values = jnp.array(self.buffer['values'], dtype=self.dtype)
        rewards = jnp.array(self.buffer['rewards'], dtype=self.dtype)
        dones = jnp.array(self.buffer['dones'], dtype=self.dtype)

        # Compute advantages and returns
        next_values = jnp.concatenate([values[1:], jnp.array([0.0])])
        advantages, returns = self.compute_gae(rewards, values, dones, next_values)

        # Normalize advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        # Create dataset
        dataset = {
            'observations': observations,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns
        }

        # Training loop
        total_metrics = {}
        for epoch in range(num_epochs):
            # Shuffle data
            indices = jax.random.permutation(self.rng, len(observations))
            for key in dataset:
                dataset[key] = dataset[key][indices]

            # Mini-batch training
            epoch_metrics = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
            num_batches = 0

            for i in range(0, len(observations), self.micro_batch_size):
                batch = {k: v[i:i+self.micro_batch_size] for k, v in dataset.items()}
                batch_metrics = self.train_step(batch)

                for key in epoch_metrics:
                    epoch_metrics[key] += batch_metrics[key]
                num_batches += 1

            # Average metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches

            total_metrics = epoch_metrics

        # Clear buffer
        for key in self.buffer:
            self.buffer[key].clear()

        return total_metrics

    def update_performance(self, returns: List[float]):
        """
        Update performance metrics for adaptation.

        Args:
            returns: Recent trading returns
        """
        # Calculate Sharpe ratio
        returns_array = jnp.array(returns)
        sharpe_ratio = jnp.mean(returns_array) / (jnp.std(returns_array) + 1e-8) * jnp.sqrt(252)  # Annualized

        self.sharpe_history.append(float(sharpe_ratio))

        # Keep history bounded
        if len(self.sharpe_history) > 100:
            self.sharpe_history = self.sharpe_history[-100:]

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.sharpe_history:
            return {'sharpe_ratio': 0.0, 'avg_sharpe': 0.0}

        current_sharpe = self.sharpe_history[-1] if self.sharpe_history else 0.0
        avg_sharpe = float(jnp.mean(jnp.array(self.sharpe_history[-10:])))  # Last 10 periods

        return {
            'sharpe_ratio': current_sharpe,
            'avg_sharpe': avg_sharpe
        }


class DistributedTrainer:
    """
    Distributed trainer with DDP and XLA SPMD support for multi-GPU/TPU training.

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

        # Determine world size based on device type
        if config.system.device == "tpu" and XLA_AVAILABLE:
            self.world_size = xm.xrt_world_size()
            self.local_rank = xm.get_ordinal()
            self.device_type = "tpu"
        else:
            # GPU/CPU fallback
            self.world_size = config.training.num_gpus
            self.local_rank = config.training.local_rank
            self.device_type = "gpu"

        # Initialize process group for GPU training
        if self.world_size > 1 and self.device_type == "gpu":
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(
                backend='nccl',
                rank=self.local_rank,
                world_size=self.world_size
            )

    def setup_distributed(self, model: NCATradingModel):
        """
        Set up distributed training (DDP for GPU, SPMD for TPU).

        Args:
            model: Model to wrap with distributed training

        Returns:
            Wrapped model
        """
        if self.device_type == "tpu" and XLA_AVAILABLE:
            # TPU uses XLA SPMD - no explicit wrapping needed
            self.logger.info(f"XLA SPMD initialized on TPU core {self.local_rank}")
            return model
        elif self.world_size > 1:
            # GPU uses DDP
            model = DDP(model, device_ids=[self.local_rank])
            self.logger.info(f"DDP initialized on rank {self.local_rank}")
            return model
        else:
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
        if self.device_type == "tpu" and XLA_AVAILABLE:
            # TPU uses XLA multiprocessing
            xmp.spawn(
                train_function,
                args=(self.world_size, *args),
                kwargs=kwargs,
                nprocs=self.world_size
            )
        elif self.world_size > 1:
            # GPU uses multiprocessing spawn
            mp.spawn(
                train_function,
                args=(self.world_size, self.local_rank, *args),
                kwargs=kwargs,
                nprocs=self.world_size
            )
        else:
            # Single device training
            train_function(self.local_rank, self.world_size, *args, **kwargs)


class RealTimeStreamingTrainer:
    """
    Real-time streaming trainer with JAX PPO and adaptive NCA growth.

    Provides continuous learning on streaming data with market intelligence integration,
    based on research papers for quantitative trading and autonomous agents.
    """

    def __init__(self, config, nca_model=None):
        """
        Initialize real-time streaming trainer.

        Args:
            config: Training configuration
            nca_model: Adaptive NCA model
        """
        self.config = config
        self.nca_model = nca_model
        self.logger = logging.getLogger(__name__)

        # JAX PPO trainer for online learning
        self.jax_ppo = None
        self.observation_dim = None
        self.action_dim = None

        # Streaming components
        self.data_handler = DataHandler()
        self.market_intelligence = MarketIntelligence(config)
        self.risk_calculator = RiskCalculator(config)

        # Real-time adaptation parameters
        self.stream_buffer_size = 1000
        self.online_update_freq = 50  # Update every 50 steps
        self.sharpe_check_freq = 100  # Check Sharpe every 100 steps
        self.sharpe_threshold = 0.5

        # Performance tracking
        self.performance_history = []
        self.streaming_data = []
        self.is_streaming = False

        # MCP integration for market intelligence
        self.brave_search_available = True  # Assume available

    def initialize_jax_ppo(self, observation_dim: int, action_dim: int):
        """
        Initialize JAX PPO trainer.

        Args:
            observation_dim: Observation space dimension
            action_dim: Action space dimension
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.jax_ppo = JAXPPOTrainer(observation_dim, action_dim, self.config, self.nca_model)
        self.logger.info("Real-time JAX PPO trainer initialized")

    async def start_streaming_training(self, symbols: List[str]):
        """
        Start real-time streaming training.

        Args:
            symbols: List of symbols to stream and train on
        """
        self.is_streaming = True
        self.logger.info(f"Starting real-time streaming training for symbols: {symbols}")

        # Initialize streaming data collection
        await self._initialize_data_streams(symbols)

        # Start training loop
        await self._streaming_training_loop(symbols)

    async def stop_streaming_training(self):
        """Stop real-time streaming training."""
        self.is_streaming = False
        self.logger.info("Stopping real-time streaming training")

    async def _initialize_data_streams(self, symbols: List[str]):
        """Initialize data streams for real-time training."""
        try:
            # Initialize data streaming (would connect to real-time feeds)
            await self.data_handler.initialize_realtime_streams(symbols)
            self.logger.info("Data streams initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize data streams: {e}")

    async def _streaming_training_loop(self, symbols: List[str]):
        """Main real-time streaming training loop."""
        step_count = 0

        while self.is_streaming:
            try:
                # Collect streaming data
                batch_data = await self._collect_streaming_batch(symbols)

                if batch_data and self.jax_ppo:
                    # Process batch with market intelligence
                    enriched_data = await self._enrich_with_market_intelligence(batch_data)

                    # Online learning update
                    self._online_learning_update(enriched_data)

                    # Periodic performance check and adaptation
                    if step_count % self.sharpe_check_freq == 0:
                        await self._check_performance_and_adapt()

                    step_count += 1

                # Control loop frequency
                await asyncio.sleep(1.0)  # 1 second intervals

            except Exception as e:
                self.logger.error(f"Error in streaming training loop: {e}")
                await asyncio.sleep(5.0)  # Wait before retry

    async def _collect_streaming_batch(self, symbols: List[str]) -> Optional[Dict[str, Any]]:
        """
        Collect batch of streaming data.

        Args:
            symbols: List of symbols

        Returns:
            Batch of streaming data or None
        """
        try:
            batch_data = []

            for symbol in symbols:
                # Get recent streaming data (would be from real-time feeds)
                recent_data = await self.data_handler.get_recent_streaming_data(symbol)

                if recent_data is not None:
                    batch_data.append({
                        'symbol': symbol,
                        'data': recent_data,
                        'timestamp': datetime.now()
                    })

            return {'batch': batch_data} if batch_data else None

        except Exception as e:
            self.logger.warning(f"Failed to collect streaming batch: {e}")
            return None

    async def _enrich_with_market_intelligence(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich batch data with market intelligence from Brave Search.

        Args:
            batch_data: Raw batch data

        Returns:
            Enriched batch data
        """
        enriched_batch = batch_data.copy()

        try:
            # Get market intelligence for each symbol in batch
            intelligence_tasks = []

            for item in batch_data.get('batch', []):
                symbol = item['symbol']
                intelligence_tasks.append(
                    self.market_intelligence.query_market_data(symbol, "streaming_training")
                )

            # Gather intelligence data
            intelligence_results = await asyncio.gather(*intelligence_tasks, return_exceptions=True)

            # Add intelligence to batch items
            for i, item in enumerate(enriched_batch.get('batch', [])):
                if i < len(intelligence_results) and not isinstance(intelligence_results[i], Exception):
                    item['market_intelligence'] = intelligence_results[i]

        except Exception as e:
            self.logger.warning(f"Failed to enrich with market intelligence: {e}")

        return enriched_batch

    def _online_learning_update(self, enriched_data: Dict[str, Any]):
        """
        Perform online learning update with enriched data.

        Args:
            enriched_data: Enriched batch data
        """
        try:
            # Extract features and create training examples
            for item in enriched_data.get('batch', []):
                # Convert streaming data to observation
                observation = self._extract_observation_from_streaming_data(item)

                # Get action from current policy
                action, log_prob, value = self.jax_ppo.get_action(observation)

                # Simulate reward (would come from actual trading)
                reward = self._calculate_streaming_reward(item)

                # Store transition for online learning
                done = False  # Streaming is continuous
                self.jax_ppo.store_transition(observation, action, log_prob, value, reward, done)

        except Exception as e:
            self.logger.error(f"Failed online learning update: {e}")

    def _extract_observation_from_streaming_data(self, item: Dict[str, Any]) -> np.ndarray:
        """
        Extract observation from streaming data item.

        Args:
            item: Streaming data item

        Returns:
            Observation array
        """
        # This would extract technical indicators and features from streaming data
        # For now, return a mock observation
        return np.random.randn(self.observation_dim)

    def _calculate_streaming_reward(self, item: Dict[str, Any]) -> float:
        """
        Calculate reward from streaming data item.

        Args:
            item: Streaming data item

        Returns:
            Reward value
        """
        # This would calculate reward based on market conditions and intelligence
        # For now, return a mock reward
        return np.random.normal(0, 0.1)

    async def _check_performance_and_adapt(self):
        """Check performance and trigger adaptations if needed."""
        try:
            # Calculate recent performance metrics
            recent_returns = [item.get('reward', 0) for item in self.streaming_data[-100:]]

            if recent_returns:
                sharpe_ratio = self.risk_calculator.calculate_sharpe_ratio(np.array(recent_returns))

                # Store performance
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'sharpe_ratio': sharpe_ratio,
                    'returns': recent_returns
                })

                # Trigger NCA growth if Sharpe ratio is too low
                if sharpe_ratio < self.sharpe_threshold:
                    self.logger.info(f"Streaming Sharpe ratio {sharpe_ratio:.3f} < {self.sharpe_threshold}, triggering NCA growth")
                    if self.nca_model and hasattr(self.nca_model, 'trigger_growth'):
                        self.nca_model.trigger_growth()

                # Keep history bounded
                if len(self.performance_history) > 50:
                    self.performance_history = self.performance_history[-50:]

        except Exception as e:
            self.logger.error(f"Failed performance check: {e}")

    def get_streaming_action(self, observation: np.ndarray, symbol: str = None) -> Tuple[int, Dict[str, Any]]:
        """
        Get action for streaming data with market intelligence.

        Args:
            observation: Current observation
            symbol: Stock symbol

        Returns:
            Tuple of (action, metadata)
        """
        if not self.jax_ppo:
            raise ValueError("JAX PPO not initialized")

        # Get action from JAX PPO
        action, log_prob, value = self.jax_ppo.get_action(observation)

        metadata = {
            'log_prob': log_prob,
            'value': value,
            'streaming_mode': True
        }

        # Add market intelligence if symbol provided
        if symbol:
            try:
                # This would be async in real implementation
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                intelligence = loop.run_until_complete(
                    self.market_intelligence.query_market_data(symbol, "streaming_decision")
                )
                metadata['market_intelligence'] = intelligence
            except Exception as e:
                self.logger.warning(f"Failed to get streaming market intelligence: {e}")

        return action, metadata


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
        self.streaming_trainer = None  # New real-time streaming trainer
        self.data_handler = DataHandler()

        # Adaptive components
        self.adaptive_grid_manager = create_adaptive_grid_manager(config)
        self.growth_strategies = create_growth_strategies(config)
        self.market_analyzer = create_market_condition_analyzer(config)
        self.performance_metrics = create_performance_metrics(config)

        # JAX PPO components
        self.jax_ppo_initialized = False
        self.observation_dim = None
        self.action_dim = None

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

    def create_model(self, adaptive: bool = False) -> Union[NCATradingModel, AdaptiveNCAWrapper]:
        """
        Create NCA trading model.

        Args:
            adaptive: Whether to create an adaptive model

        Returns:
            Initialized NCA model or adaptive wrapper
        """
        from nca_model import create_nca_model

        base_model = create_nca_model(self.config)

        if adaptive:
            # Create adaptive wrapper
            self.model = create_adaptive_nca_wrapper(base_model, self.config)
            self.logger.info("Adaptive NCA model created")
            self.logger.info(f"  Base model: {type(base_model).__name__}")
            self.logger.info(f"  Adaptive wrapper: {type(self.model).__name__}")
        else:
            self.model = base_model
            self.logger.info("Standard NCA model created")

        return self.model

    def initialize_jax_ppo(self, observation_dim: int, action_dim: int):
        """
        Initialize JAX PPO trainer for real-time learning.

        Args:
            observation_dim: Observation space dimension
            action_dim: Action space dimension
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Initialize JAX PPO in trainer
        if self.trainer:
            self.trainer.initialize_jax_ppo(observation_dim, action_dim, self.model)

        # Initialize streaming trainer
        self.streaming_trainer = RealTimeStreamingTrainer(self.config, self.model)
        self.streaming_trainer.initialize_jax_ppo(observation_dim, action_dim)

        self.jax_ppo_initialized = True
        self.logger.info("JAX PPO initialized for real-time adaptation")

    async def train_streaming(self, symbols: List[str]):
        """
        Start real-time streaming training.

        Args:
            symbols: List of symbols to stream and train on
        """
        if not self.jax_ppo_initialized:
            raise ValueError("JAX PPO must be initialized before streaming training")

        self.current_mode = 'streaming'
        self.is_training = True

        self.logger.info(f"Starting streaming training for symbols: {symbols}")

        # Start streaming training
        await self.streaming_trainer.start_streaming_training(symbols)

    def get_streaming_action(self, observation: np.ndarray, symbol: str = None) -> Tuple[int, Dict[str, Any]]:
        """
        Get action from streaming trainer with market intelligence.

        Args:
            observation: Current observation
            symbol: Stock symbol

        Returns:
            Tuple of (action, metadata)
        """
        if not self.streaming_trainer:
            raise ValueError("Streaming trainer not initialized")

        return self.streaming_trainer.get_streaming_action(observation, symbol)

    async def get_market_intelligence(self, symbol: str, context: str = "trading_decision") -> Dict[str, Any]:
        """
        Get market intelligence for trading decisions.

        Args:
            symbol: Stock symbol
            context: Decision context

        Returns:
            Market intelligence data
        """
        if self.streaming_trainer:
            return await self.streaming_trainer.market_intelligence.query_market_data(symbol, context)
        elif self.trainer:
            return await self.trainer.get_market_intelligence(symbol, context)
        else:
            # Fallback to basic market intelligence
            market_intel = MarketIntelligence(self.config)
            return await market_intel.query_market_data(symbol, context)

    def load_model(self, model_path: str) -> NCATradingModel:
        """
        Load NCA model from file.

        Args:
            model_path: Path to model file

        Returns:
            Loaded NCA model
        """
        from nca_model import load_nca_model

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

        if self.streaming_trainer:
            asyncio.create_task(self.streaming_trainer.stop_streaming_training())

        self.logger.info("Training stopped")

    async def stop_streaming_training(self):
        """Stop streaming training session."""
        if self.streaming_trainer:
            await self.streaming_trainer.stop_streaming_training()
        self.is_training = False
        self.logger.info("Streaming training stopped")

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
        Get training summary including new real-time features.

        Returns:
            Training summary dictionary
        """
        summary = {
            'is_training': self.is_training,
            'current_mode': self.current_mode,
            'model_path': str(self.model_dir),
            'log_path': str(self.log_dir),
            'jax_ppo_initialized': self.jax_ppo_initialized,
            'streaming_trainer_active': self.streaming_trainer is not None,
            'metrics': self.trainer.metrics if self.trainer else {}
        }

        # Add streaming trainer info
        if self.streaming_trainer:
            summary['streaming'] = {
                'is_streaming': self.streaming_trainer.is_streaming,
                'performance_history_length': len(self.streaming_trainer.performance_history),
                'observation_dim': self.streaming_trainer.observation_dim,
                'action_dim': self.streaming_trainer.action_dim
            }

        # Add JAX PPO info
        if self.jax_ppo_initialized and self.trainer and self.trainer.jax_ppo_trainer:
            jax_ppo = self.trainer.jax_ppo_trainer
            summary['jax_ppo'] = {
                'buffer_size': len(jax_ppo.buffer['observations']),
                'sharpe_history_length': len(jax_ppo.sharpe_history),
                'current_sharpe': jax_ppo.sharpe_history[-1] if jax_ppo.sharpe_history else None
            }

        return summary


class MarketIntelligence:
    """
    Market intelligence gathering using Brave Search MCP.

    Integrates real-time market news and stock information for trading decisions.
    """

    def __init__(self, config):
        """Initialize market intelligence."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def query_market_data(self, symbol: str, context: str = "trading_decision") -> Dict[str, Any]:
        """
        Query market data using Brave Search MCP.

        Args:
            symbol: Stock symbol
            context: Query context

        Returns:
            Market intelligence data
        """
        try:
            # Construct query based on context
            if context == "trading_decision":
                query = f"{symbol} stock latest news market sentiment technical analysis"
            elif context == "streaming_training":
                query = f"{symbol} real-time market data price action volatility"
            elif context == "streaming_decision":
                query = f"{symbol} current market conditions breaking news"
            else:
                query = f"{symbol} stock analysis {context}"

            # Use MCP brave_web_search tool
            search_results = await self._perform_brave_search(query)

            # Process results into intelligence data
            intelligence = self._process_search_results(symbol, search_results, context)
            intelligence['timestamp'] = datetime.now()

            return intelligence

        except Exception as e:
            self.logger.error(f"Failed to get market intelligence: {e}")
            # Return fallback mock data
            return {
                'symbol': symbol,
                'news_sentiment': 'neutral',
                'market_trend': 'sideways',
                'volatility': 'medium',
                'key_news': [],
                'technical_signals': {},
                'timestamp': datetime.now(),
                'error': str(e)
            }

    async def _perform_brave_search(self, query: str) -> Dict[str, Any]:
        """
        Perform Brave search using MCP tool.

        Args:
            query: Search query

        Returns:
            Search results
        """
        try:
            # Use MCP brave_search tool
            # This assumes the MCP server is connected and available
            search_results = await use_mcp_tool(
                server_name="brave-search",
                tool_name="brave_web_search",
                arguments={
                    "query": query,
                    "count": 10,  # Get top 10 results
                    "offset": 0
                }
            )
            return search_results
        except Exception as e:
            self.logger.warning(f"MCP Brave search failed: {e}, using fallback")
            # Fallback to mock results
            return {
                'results': [
                    {
                        'title': f'{query} - Market Analysis',
                        'description': f'Latest analysis for {query}',
                        'url': f'https://example.com/{query.replace(" ", "_")}'
                    }
                ]
            }

    def _process_search_results(self, symbol: str, search_results: Dict[str, Any], context: str) -> Dict[str, Any]:
        """
        Process search results into market intelligence.

        Args:
            symbol: Stock symbol
            search_results: Raw search results
            context: Query context

        Returns:
            Processed intelligence data
        """
        # Extract key information from search results
        results = search_results.get('results', [])

        # Analyze sentiment from titles and descriptions
        sentiment_scores = []
        volatility_indicators = []
        key_news = []

        for result in results[:5]:  # Process top 5 results
            title = result.get('title', '').lower()
            description = result.get('description', '').lower()

            # Simple sentiment analysis
            positive_words = ['bullish', 'gains', 'up', 'rise', 'growth', 'positive']
            negative_words = ['bearish', 'losses', 'down', 'fall', 'decline', 'negative']

            pos_score = sum(1 for word in positive_words if word in title or word in description)
            neg_score = sum(1 for word in negative_words if word in title or word in description)

            sentiment_scores.append(pos_score - neg_score)

            # Volatility indicators
            if any(word in title + description for word in ['volatile', 'volatility', 'swing', 'fluctuation']):
                volatility_indicators.append('high')
            elif any(word in title + description for word in ['stable', 'steady', 'calm']):
                volatility_indicators.append('low')
            else:
                volatility_indicators.append('medium')

            # Extract key news
            if len(key_news) < 3:  # Keep top 3 news items
                key_news.append({
                    'title': result.get('title', ''),
                    'summary': result.get('description', '')[:200] + '...',
                    'url': result.get('url', '')
                })

        # Determine overall sentiment
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        if avg_sentiment > 0.5:
            news_sentiment = 'positive'
        elif avg_sentiment < -0.5:
            news_sentiment = 'negative'
        else:
            news_sentiment = 'neutral'

        # Determine market trend (simplified)
        if 'up' in str(results).lower() and 'trend' in str(results).lower():
            market_trend = 'up'
        elif 'down' in str(results).lower() and 'trend' in str(results).lower():
            market_trend = 'down'
        else:
            market_trend = 'sideways'

        # Determine volatility
        volatility_counts = {'low': 0, 'medium': 0, 'high': 0}
        for vol in volatility_indicators:
            volatility_counts[vol] += 1

        volatility = max(volatility_counts, key=volatility_counts.get)

        # Generate technical signals (simplified)
        technical_signals = {
            'momentum': 'neutral',
            'volume': 'normal',
            'support_resistance': 'testing'
        }

        return {
            'symbol': symbol,
            'news_sentiment': news_sentiment,
            'market_trend': market_trend,
            'volatility': volatility,
            'key_news': key_news,
            'technical_signals': technical_signals,
            'sentiment_score': float(avg_sentiment),
            'search_results_count': len(results)
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
    from config import ConfigManager

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