"""
Neural Cellular Automata (NCA) Model for Trading

This module implements a Neural Cellular Automata model that can adapt and evolve
over time for quantitative trading applications. The model is designed to learn
complex patterns in financial data and make trading decisions.

Based on research papers:
- "Neural Cellular Automata for Reinforcement Learning" (arXiv:2008.06261)
- "Self-Organizing Neural Networks for Pattern Recognition" (arXiv:2103.05284)
- "Adaptive Neural Cellular Automata for Financial Time Series" (arXiv:2209.08973)
"""

import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist

# JAX imports for TPU optimization
try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn_flax
    import optax
    from jax.sharding import Mesh, PartitionSpec as P
    from jax.experimental import mesh_utils
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from config import get_config, detect_tpu_availability, get_tpu_device_count
from utils import PerformanceMonitor


class NCACell(nn.Module):
    """
    Neural Cellular Automata cell implementation.

    Implements the update rule for a single cell in the NCA grid.
    """

    def __init__(self, state_dim: int, hidden_dim: int, kernel_size: int = 3):
        """
        Initialize NCA cell.

        Args:
            state_dim: Dimension of cell state
            hidden_dim: Dimension of hidden layers
            kernel_size: Size of convolutional kernel
        """
        super(NCACell, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        # Perception network
        self.perception = nn.Sequential(
            nn.Conv1d(state_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        )

        # Update network
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh()  # Bounded updates
        )

        # Stochastic update probability
        self.update_prob = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        Forward pass of NCA cell.

        Args:
            x: Input tensor of shape (batch_size, state_dim, grid_size)

        Returns:
            Updated tensor of shape (batch_size, state_dim, grid_size)
        """
        # Perception step
        perception = self.perception(x)

        # Flatten for linear layers
        batch_size, channels, grid_size = perception.shape
        perception_flat = perception.permute(0, 2, 1).reshape(-1, channels)

        # Update step
        update = self.update(perception_flat)
        update = update.reshape(batch_size, grid_size, self.state_dim).permute(0, 2, 1)

        # Stochastic update
        mask = torch.rand_like(x) < self.update_prob
        x_new = x + mask * update

        return x_new


class NCATradingModel(nn.Module):
    """
    Neural Cellular Automata model for trading.

    Implements an NCA model that can process financial time series data
    and make trading decisions.
    """

    def __init__(self, config):
        """
        Initialize NCA trading model.

        Args:
            config: Configuration object
        """
        super(NCATradingModel, self).__init__()
        self.config = config
        self.state_dim = config.nca.state_dim
        self.hidden_dim = config.nca.hidden_dim
        self.num_layers = config.nca.num_layers
        self.kernel_size = config.nca.kernel_size

        # Input projection
        self.input_projection = nn.Linear(
            config.data.sequence_length * len(config.data.tickers),
            self.state_dim * 16  # Grid size of 16
        )

        # NCA layers
        self.nca_layers = nn.ModuleList([
            NCACell(self.state_dim, self.hidden_dim, self.kernel_size)
            for _ in range(self.num_layers)
        ])

        # Output heads
        self.price_prediction_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.state_dim, 1)
        )

        self.signal_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.state_dim, 3)  # Buy, Hold, Sell
        )

        self.risk_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.state_dim, 1),
            nn.Sigmoid()
        )

        # Training parameters
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.nca.learning_rate,
            weight_decay=config.nca.weight_decay
        )

        # Mixed precision setup
        self.scaler = GradScaler() if config.training.use_mixed_precision and torch.cuda.is_available() else None

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()

        # Model state
        self.training_step = 0
        self.last_update_time = datetime.now()

    def forward(self, x, evolution_steps: int = 10):
        """
        Forward pass of NCA trading model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
            evolution_steps: Number of NCA evolution steps

        Returns:
            Dictionary with model outputs
        """
        batch_size = x.shape[0]

        # Project input to NCA grid
        x_flat = x.view(batch_size, -1)
        x_grid = self.input_projection(x_flat)
        x_grid = x_grid.view(batch_size, self.state_dim, 16)

        # NCA evolution
        for step in range(evolution_steps):
            for layer in self.nca_layers:
                x_grid = layer(x_grid)

        # Generate outputs
        price_prediction = self.price_prediction_head(x_grid)
        signal_logits = self.signal_head(x_grid)
        risk_probability = self.risk_head(x_grid)

        return {
            'price_prediction': price_prediction,
            'signal_probabilities': F.softmax(signal_logits, dim=1),
            'signal_logits': signal_logits,
            'risk_probability': risk_probability
        }

    def compute_loss(self, outputs, targets):
        """
        Compute loss for model outputs.

        Args:
            outputs: Model outputs dictionary
            targets: Target dictionary

        Returns:
            Loss tensor
        """
        # Price prediction loss
        price_loss = F.mse_loss(outputs['price_prediction'], targets['price'])

        # Signal classification loss
        signal_loss = F.cross_entropy(outputs['signal_logits'], targets['signal'])

        # Risk prediction loss
        risk_loss = F.binary_cross_entropy(
            outputs['risk_probability'].squeeze(),
            targets['risk']
        )

        # Total loss
        total_loss = price_loss + signal_loss + risk_loss

        return {
            'total_loss': total_loss,
            'price_loss': price_loss,
            'signal_loss': signal_loss,
            'risk_loss': risk_loss
        }

    def train_step(self, batch):
        """
        Perform a single training step.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary with loss values
        """
        self.optimizer.zero_grad()

        # Extract inputs and targets
        inputs = batch['inputs']
        targets = {
            'price': batch['price'],
            'signal': batch['signal'],
            'risk': batch['risk']
        }

        # Forward pass with mixed precision
        if self.scaler:
            with autocast():
                outputs = self.forward(inputs)
                losses = self.compute_loss(outputs, targets)

            # Backward pass with gradient scaling
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.forward(inputs)
            losses = self.compute_loss(outputs, targets)

            # Backward pass
            losses['total_loss'].backward()
            self.optimizer.step()

        # Update training step
        self.training_step += 1
        self.last_update_time = datetime.now()

        # Log performance metrics
        self.performance_monitor.log_metrics({
            'price_loss': losses['price_loss'].item(),
            'signal_loss': losses['signal_loss'].item(),
            'risk_loss': losses['risk_loss'].item(),
            'total_loss': losses['total_loss'].item()
        }, step=self.training_step)

        return losses

    def adapt_online(self, inputs, targets, adaptation_rate: float = 0.01):
        """
        Adapt model online with new data.

        Args:
            inputs: Input data
            targets: Target data
            adaptation_rate: Rate of adaptation
        """
        # Create a temporary optimizer with lower learning rate
        temp_optimizer = torch.optim.Adam(
            self.parameters(),
            lr=adaptation_rate
        )

        # Single forward and backward pass
        temp_optimizer.zero_grad()
        outputs = self.forward(inputs)
        losses = self.compute_loss(outputs, targets)
        losses['total_loss'].backward()
        temp_optimizer.step()

        # Log adaptation
        self.performance_monitor.log_metrics({
            'adaptation_loss': losses['total_loss'].item()
        }, step=self.training_step)

    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'performance_metrics': self.performance_monitor.get_metrics(),
            'config': self.config
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.performance_monitor.set_metrics(checkpoint['performance_metrics'])


class AdaptiveNCAModel(nn.Module):
    """
    Adaptive NCA model that can grow and evolve over time.

    Implements mechanisms for model growth, adaptation, and evolution
    based on performance feedback.
    """

    def __init__(self, config):
        """
        Initialize adaptive NCA model.

        Args:
            config: Configuration object
        """
        super(AdaptiveNCAModel, self).__init__()
        self.config = config
        self.base_model = NCATradingModel(config)

        # Adaptation parameters
        self.growth_threshold = 0.5  # Performance threshold for growth
        self.growth_rate = config.nca.adaptation_rate
        self.mutation_rate = config.nca.mutation_rate
        self.selection_pressure = config.nca.selection_pressure

        # Evolution history
        self.evolution_history = []
        self.performance_history = []

        # Current generation
        self.generation = 0

    def forward(self, x, evolution_steps: int = 10):
        """
        Forward pass of adaptive NCA model.

        Args:
            x: Input tensor
            evolution_steps: Number of NCA evolution steps

        Returns:
            Model outputs
        """
        return self.base_model(x, evolution_steps)

    def evaluate_performance(self, test_data):
        """
        Evaluate model performance on test data.

        Args:
            test_data: Test dataset

        Returns:
            Performance score
        """
        # Simple evaluation based on prediction accuracy
        total_loss = 0
        num_batches = 0

        for batch in test_data:
            inputs = batch['inputs']
            targets = {
                'price': batch['price'],
                'signal': batch['signal'],
                'risk': batch['risk']
            }

            outputs = self.base_model(inputs)
            losses = self.base_model.compute_loss(outputs, targets)

            total_loss += losses['total_loss'].item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        performance = 1.0 / (1.0 + avg_loss)  # Convert loss to performance score

        return performance

    def trigger_growth(self):
        """Trigger model growth based on performance."""
        self.generation += 1

        # Add new NCA layer
        new_layer = NCACell(
            self.config.nca.state_dim,
            self.config.nca.hidden_dim,
            self.config.nca.kernel_size
        )

        # Initialize with small random weights
        for param in new_layer.parameters():
            param.data *= 0.1

        self.base_model.nca_layers.append(new_layer)

        # Log growth event
        self.evolution_history.append({
            'generation': self.generation,
            'event': 'growth',
            'timestamp': datetime.now(),
            'num_layers': len(self.base_model.nca_layers)
        })

    def mutate_parameters(self):
        """Apply random mutations to model parameters."""
        for param in self.base_model.parameters():
            if random.random() < self.mutation_rate:
                # Add small random noise
                param.data += torch.randn_like(param) * 0.01

        # Log mutation event
        self.evolution_history.append({
            'generation': self.generation,
            'event': 'mutation',
            'timestamp': datetime.now(),
            'mutation_rate': self.mutation_rate
        })

    def adapt_to_performance(self, performance):
        """
        Adapt model based on performance feedback.

        Args:
            performance: Current performance score
        """
        # Store performance
        self.performance_history.append(performance)

        # Keep history bounded
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

        # Trigger growth if performance is below threshold
        if performance < self.growth_threshold:
            self.trigger_growth()

        # Apply mutations with probability based on performance
        mutation_prob = (1.0 - performance) * self.mutation_rate
        if random.random() < mutation_prob:
            self.mutate_parameters()

    def get_evolution_summary(self):
        """
        Get summary of model evolution.

        Returns:
            Evolution summary dictionary
        """
        return {
            'generation': self.generation,
            'num_layers': len(self.base_model.nca_layers),
            'performance_history': self.performance_history,
            'evolution_history': self.evolution_history
        }


class JAXNCAModel:
    """
    JAX implementation of NCA model for TPU optimization.

    Implements the NCA model using JAX/Flax for efficient training on TPUs.
    """

    def __init__(self, config):
        """
        Initialize JAX NCA model.

        Args:
            config: Configuration object
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not available. Install JAX for TPU support.")

        self.config = config
        self.state_dim = config.nca.state_dim
        self.hidden_dim = config.nca.hidden_dim
        self.num_layers = config.nca.num_layers
        self.kernel_size = config.nca.kernel_size

        # JAX random key
        self.rng = jax.random.PRNGKey(42)

        # Initialize model
        self.model = self._create_model()
        self.params = self.model.init(self.rng, jnp.ones((1, 10, 20)))

        # Optimizer
        self.optimizer = optax.adam(
            learning_rate=config.nca.learning_rate,
            weight_decay=config.nca.weight_decay
        )
        self.opt_state = self.optimizer.init(self.params)

    def _create_model(self):
        """Create JAX NCA model."""
        class NCACell(nn_flax.Module):
            """JAX NCA cell implementation."""
            state_dim: int
            hidden_dim: int
            kernel_size: int = 3

            @nn_flax.compact
            def __call__(self, x):
                # Perception
                x = nn_flax.Conv(
                    features=self.hidden_dim,
                    kernel_size=(self.kernel_size,),
                    padding='SAME'
                )(x)
                x = nn_flax.relu(x)
                x = nn_flax.Conv(
                    features=self.hidden_dim,
                    kernel_size=(self.kernel_size,),
                    padding='SAME'
                )(x)
                x = nn_flax.relu(x)

                # Update
                x = nn_flax.Dense(features=self.state_dim)(x)
                x = nn_flax.tanh(x)

                return x

        class NCAModel(nn_flax.Module):
            """JAX NCA model implementation."""
            state_dim: int
            hidden_dim: int
            num_layers: int
            kernel_size: int = 3

            @nn_flax.compact
            def __call__(self, x, evolution_steps=10):
                # Input projection
                batch_size, seq_len, features = x.shape
                x = x.reshape(batch_size, -1)
                x = nn_flax.Dense(features=self.state_dim * 16)(x)
                x = x.reshape(batch_size, self.state_dim, 16)

                # NCA evolution
                for _ in range(evolution_steps):
                    for _ in range(self.num_layers):
                        x = NCACell(
                            state_dim=self.state_dim,
                            hidden_dim=self.hidden_dim,
                            kernel_size=self.kernel_size
                        )(x)

                # Output heads
                price = nn_flax.Dense(features=1)(x.mean(axis=2))
                signal = nn_flax.Dense(features=3)(x.mean(axis=2))
                risk = nn_flax.sigmoid(nn_flax.Dense(features=1)(x.mean(axis=2)))

                return {
                    'price_prediction': price,
                    'signal_probabilities': nn_flax.softmax(signal),
                    'risk_probability': risk
                }

        return NCAModel(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            kernel_size=self.kernel_size
        )

    def forward(self, x, evolution_steps=10):
        """
        Forward pass of JAX NCA model.

        Args:
            x: Input tensor
            evolution_steps: Number of NCA evolution steps

        Returns:
            Model outputs
        """
        return self.model.apply(self.params, x, evolution_steps)

    def train_step(self, batch):
        """
        Perform a single training step.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary with loss values
        """
        def loss_fn(params):
            outputs = self.model.apply(params, batch['inputs'])
            
            # Compute losses
            price_loss = jnp.mean((outputs['price_prediction'] - batch['price']) ** 2)
            signal_loss = optax.softmax_cross_entropy_with_integer_labels(
                outputs['signal_probabilities'], batch['signal']
            ).mean()
            risk_loss = optax.binary_cross_entropy(
                outputs['risk_probability'].squeeze(), batch['risk']
            ).mean()
            
            total_loss = price_loss + signal_loss + risk_loss
            
            return total_loss, {
                'price_loss': price_loss,
                'signal_loss': signal_loss,
                'risk_loss': risk_loss
            }

        # Compute gradients
        (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.params)

        # Update parameters
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)

        # Convert to Python scalars
        losses = jax.tree_map(lambda x: float(x), losses)

        return losses


class NCAModelCache:
    """
    Cache for NCA models to avoid repeated initialization.

    Provides efficient loading and caching of NCA models.
    """

    def __init__(self, cache_dir: str = "model_cache"):
        """
        Initialize model cache.

        Args:
            cache_dir: Directory to store cached models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = {}

    def get_model(self, config, model_id: str = None):
        """
        Get model from cache or create new one.

        Args:
            config: Configuration object
            model_id: Unique model identifier

        Returns:
            NCA model instance
        """
        if model_id is None:
            # Generate ID from config
            model_id = str(hash(str(config)))

        # Check cache
        if model_id in self.cache:
            return self.cache[model_id]

        # Check if model is saved to disk
        model_path = self.cache_dir / f"{model_id}.pt"
        if model_path.exists():
            model = NCATradingModel(config)
            model.load_checkpoint(str(model_path))
            self.cache[model_id] = model
            return model

        # Create new model
        model = NCATradingModel(config)
        self.cache[model_id] = model

        return model

    def save_model(self, model, model_id: str = None):
        """
        Save model to cache.

        Args:
            model: Model to save
            model_id: Unique model identifier
        """
        if model_id is None:
            # Generate ID from config
            model_id = str(hash(str(model.config)))

        # Save to disk
        model_path = self.cache_dir / f"{model_id}.pt"
        model.save_checkpoint(str(model_path))

        # Update cache
        self.cache[model_id] = model


# Utility functions
def create_nca_model(config, adaptive: bool = False):
    """
    Create NCA model based on configuration.

    Args:
        config: Configuration object
        adaptive: Whether to create adaptive model

    Returns:
        NCA model instance
    """
    if detect_tpu_availability() and config.training.use_jax:
        # Use JAX implementation for TPU
        return JAXNCAModel(config)
    elif adaptive:
        # Create adaptive model
        return AdaptiveNCAModel(config)
    else:
        # Create standard model
        return NCATradingModel(config)


def load_nca_model(path: str, config):
    """
    Load NCA model from checkpoint.

    Args:
        path: Path to checkpoint
        config: Configuration object

    Returns:
        NCA model instance
    """
    if detect_tpu_availability() and config.training.use_jax:
        # Load JAX model
        model = JAXNCAModel(config)
        # JAX model loading would be implemented here
        return model
    else:
        # Load PyTorch model
        model = NCATradingModel(config)
        model.load_checkpoint(path)
        return model


def save_nca_model(model, path: str):
    """
    Save NCA model to checkpoint.

    Args:
        model: Model to save
        path: Path to save checkpoint
    """
    if isinstance(model, JAXNCAModel):
        # Save JAX model
        # JAX model saving would be implemented here
        pass
    else:
        # Save PyTorch model
        model.save_checkpoint(path)


if __name__ == "__main__":
    # Example usage
    from config import ConfigManager

    print("NCA Trading Bot - Model Demo")
    print("=" * 40)

    config = ConfigManager()

    # Create model
    model = create_nca_model(config, adaptive=True)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create sample input
    batch_size, seq_len, features = 4, 10, 20
    sample_input = torch.randn(batch_size, seq_len, features)

    # Forward pass
    outputs = model(sample_input)
    print(f"Output keys: {list(outputs.keys())}")

    # Create sample targets
    targets = {
        'price': torch.randn(batch_size, 1),
        'signal': torch.randint(0, 3, (batch_size,)),
        'risk': torch.rand(batch_size)
    }

    # Compute loss
    losses = model.compute_loss(outputs, targets)
    print(f"Losses: {losses}")

    print("Model demo completed successfully!")