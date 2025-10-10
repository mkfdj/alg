"""
Adaptive Neural Cellular Automata implementation for trading
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
from jax import lax, config, vmap, jit, grad, value_and_grad
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Tuple, Dict, List, Optional, Any
import numpy as np

from .config import Config


class SobelPerception(nn.Module):
    """Sobel filter perception for NCA cells"""

    @nn.compact
    def __call__(self, grid):
        """
        Apply Sobel filters for gradient perception
        Args:
            grid: [batch, height, width, channels]
        Returns:
            perception: [batch, height, width, channels * 3]
        """
        batch, height, width, channels = grid.shape

        # Sobel kernels for gradient detection
        sobel_x = jnp.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]], dtype=jnp.float32)
        sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]], dtype=jnp.float32)

        # Add channel dimension
        sobel_x = sobel_x[None, None, :, :]  # [1, 1, 3, 3]
        sobel_y = sobel_y[None, None, :, :]  # [1, 1, 3, 3]

        # Reshape for convolution
        grid_reshaped = grid.reshape(-1, height, width, channels)

        # Apply Sobel filters to each channel
        grad_x = jnp.zeros_like(grid_reshaped)
        grad_y = jnp.zeros_like(grid_reshaped)

        for c in range(channels):
            channel = grid_reshaped[..., c:c+1]
            grad_x = grad_x.at[..., c].add(
                lax.conv_general_dilated(
                    channel,
                    sobel_x,
                    window_strides=[1, 1],
                    padding='SAME'
                ).squeeze(-1)
            )
            grad_y = grad_y.at[..., c].add(
                lax.conv_general_dilated(
                    channel,
                    sobel_y,
                    window_strides=[1, 1],
                    padding='SAME'
                ).squeeze(-1)
            )

        # Concatenate original state with gradients
        perception = jnp.concatenate([grid_reshaped, grad_x, grad_y], axis=-1)

        return perception.reshape(batch, height, width, channels * 3)


class UpdateRule(nn.Module):
    """NCA cell update rule MLP"""

    hidden_dim: int = 128
    channels: int = 16

    @nn.compact
    def __call__(self, perception):
        """
        Apply update rule to perception vector
        Args:
            perception: [batch, height, width, perception_dim]
        Returns:
            update: [batch, height, width, channels]
        """
        x = nn.Dense(features=self.hidden_dim)(perception)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)

        # Output update with zero initialization for stability
        update = nn.Dense(
            features=self.channels,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros
        )(x)

        return update


class AdaptiveNCA(nn.Module):
    """Adaptive Neural Cellular Automata for trading"""

    config: Config

    def setup(self):
        """Initialize NCA components"""
        self.perception = SobelPerception()
        self.update_rule = UpdateRule(
            hidden_dim=self.config.nca_hidden_dim,
            channels=self.config.nca_channels
        )

        # Growth parameters for adaptive architecture
        self.growth_threshold = self.config.nca_growth_threshold
        self.max_grid_size = self.config.nca_max_grid_size

    def __call__(self, grid, rng_key, training=True):
        """
        Evolve NCA grid for one step
        Args:
            grid: [batch, height, width, channels]
            rng_key: JAX random key
            training: Whether in training mode
        Returns:
            new_grid: [batch, height, width, channels]
        """
        # Perception step
        perception = self.perception(grid)

        # Update rule
        update = self.update_rule(perception)

        if training:
            # Stochastic update during training
            rng_key, subkey = jrand.split(rng_key)
            update_mask = jrand.bernoulli(subkey, p=0.5, shape=grid.shape)
            update = update * update_mask

        # Residual update
        new_grid = grid + update

        # Living cell masking
        alive_mask = self._get_alive_mask(new_grid)
        new_grid = new_grid * alive_mask

        return new_grid

    def _get_alive_mask(self, grid):
        """Create mask for living cells (alpha > 0.1)"""
        # Alpha channel is typically channel 3 (RGB + Alpha)
        alpha = grid[..., 3:4]

        # Check if any neighbor is alive
        alive = lax.max_pool(alpha, window_shape=(3, 3), strides=(1, 1), padding='SAME')
        alive_mask = (alive > 0.1).astype(grid.dtype)

        return alive_mask

    @jit
    def evolve(self, grid, rng_key, steps=96):
        """
        Evolve NCA grid for multiple steps
        Args:
            grid: [batch, height, width, channels]
            rng_key: JAX random key
            steps: Number of evolution steps
        Returns:
            final_grid: [batch, height, width, channels]
        """
        def step_fn(carry, _):
            grid, rng_key = carry
            rng_key, subkey = jrand.split(rng_key)
            new_grid = self(grid, subkey, training=False)
            return (new_grid, rng_key), None

        (final_grid, _), _ = lax.scan(step_fn, (grid, rng_key), None, length=steps)
        return final_grid

    def should_grow(self, grid, error_rate, data_complexity):
        """
        Determine if NCA should grow based on performance and data complexity
        Args:
            grid: Current NCA grid
            error_rate: Current prediction error rate
            data_complexity: Complexity measure of input data
        Returns:
            should_grow: Boolean indicating if grid should expand
        """
        batch, height, width, channels = grid.shape

        # Check if we haven't reached max size
        at_max_size = (height >= self.max_grid_size[0] or
                      width >= self.max_grid_size[1])

        # Check growth criteria
        error_threshold = self.growth_threshold
        complexity_threshold = 0.5

        should_grow = (error_rate > error_threshold and
                      data_complexity > complexity_threshold and
                      not at_max_size)

        return should_grow

    def expand_grid(self, grid):
        """
        Expand NCA grid by adding cells at boundaries
        Args:
            grid: [batch, height, width, channels]
        Returns:
            expanded_grid: [batch, height+2, width+2, channels]
        """
        # Pad grid with zeros at boundaries
        expanded_grid = jnp.pad(
            grid,
            ((0, 0), (1, 1), (1, 1), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return expanded_grid

    def initialize_grid(self, batch_size=1, seed=42):
        """
        Initialize NCA grid with single seed cell
        Args:
            batch_size: Number of grids to initialize
            seed: Random seed
        Returns:
            grid: [batch_size, height, width, channels]
        """
        rng_key = jrand.PRNGKey(seed)
        height, width = self.config.nca_grid_size
        channels = self.config.nca_channels

        # Initialize with zeros
        grid = jnp.zeros((batch_size, height, width, channels))

        # Add seed cell at center
        center_y, center_x = height // 2, width // 2
        grid = grid.at[:, center_y, center_x, :].set(1.0)  # All channels = 1.0

        return grid


class NCAEnsemble:
    """Ensemble of NCAs for robust predictions"""

    def __init__(self, config: Config):
        self.config = config
        self.num_models = config.rl_ensemble_size
        self.models = []
        self.states = []

    def initialize(self, rng_key):
        """Initialize ensemble of NCA models"""
        keys = jrand.split(rng_key, self.num_models)

        for i in range(self.num_models):
            model = AdaptiveNCA(self.config)

            # Initialize parameters
            dummy_grid = jnp.zeros((1, *self.config.nca_grid_size, self.config.nca_channels))
            params = model.init(keys[i], dummy_grid, keys[i], training=False)

            # Create optimizer
            tx = optax.adam(self.config.nca_learning_rate)
            state = train_state.TrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=tx
            )

            self.models.append(model)
            self.states.append(state)

    def predict(self, grids, rng_key):
        """
        Make ensemble prediction
        Args:
            grids: Input NCA grids [batch, height, width, channels]
            rng_key: Random key for evolution
        Returns:
            predictions: Ensemble predictions
        """
        keys = jrand.split(rng_key, self.num_models)
        predictions = []

        for i, (model, state) in enumerate(zip(self.models, self.states)):
            pred = model.apply(
                state.params,
                grids,
                keys[i],
                training=False
            )
            predictions.append(pred)

        # Stack and average predictions
        predictions = jnp.stack(predictions, axis=0)  # [ensemble, batch, ...]
        ensemble_pred = jnp.mean(predictions, axis=0)

        return ensemble_pred, predictions

    def update_states(self, new_states):
        """Update ensemble training states"""
        self.states = new_states


def create_nca_loss_function(target_pattern, alpha_weight=1.0, rgb_weight=1.0):
    """
    Create loss function for NCA training
    Args:
        target_pattern: Target pattern to grow
        alpha_weight: Weight for alpha channel loss
        rgb_weight: Weight for RGB channel loss
    Returns:
        loss_function: Loss function for training
    """
    def loss_fn(params, model, grid, rng_key):
        """Compute loss between evolved grid and target pattern"""
        # Evolve grid
        evolved_grid = model.apply(params, grid, rng_key, training=False)

        # Compute MSE loss for each channel
        alpha_loss = jnp.mean(
            (evolved_grid[..., 3:4] - target_pattern[..., 3:4]) ** 2
        )
        rgb_loss = jnp.mean(
            (evolved_grid[..., :3] - target_pattern[..., :3]) ** 2
        )

        total_loss = alpha_weight * alpha_loss + rgb_weight * rgb_loss

        return total_loss, {
            "alpha_loss": alpha_loss,
            "rgb_loss": rgb_loss,
            "total_loss": total_loss
        }

    return loss_fn


def create_sample_pool(initial_grid, pool_size=1024):
    """
    Create sample pool for NCA training
    Args:
        initial_grid: Initial NCA grid
        pool_size: Size of sample pool
    Returns:
        sample_pool: Array of grids for training
    """
    pool = jnp.tile(initial_grid[0:1], (pool_size, 1, 1, 1))
    return pool


def apply_damage(grid, damage_prob=0.1, max_damage_radius=5):
    """
    Apply random damage to NCA grid for robustness training
    Args:
        grid: Input NCA grid
        damage_prob: Probability of applying damage
        max_damage_radius: Maximum radius of damage
    Returns:
        damaged_grid: Grid with applied damage
    """
    batch, height, width, channels = grid.shape
    rng_key = jrand.PRNGKey(0)

    # Randomly decide to apply damage
    should_damage = jrand.bernoulli(rng_key, p=damage_prob)

    if should_damage:
        # Random center for damage
        rng_key, *keys = jrand.split(rng_key, 4)
        center_y = jrand.randint(keys[0], (), 0, height)
        center_x = jrand.randint(keys[1], (), 0, width)
        radius = jrand.randint(keys[2], (), 1, max_damage_radius + 1)

        # Create damage mask
        y_coords, x_coords = jnp.meshgrid(
            jnp.arange(height),
            jnp.arange(width),
            indexing='ij'
        )

        distance = jnp.sqrt(
            (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2
        )
        damage_mask = (distance <= radius).astype(jnp.float32)
        damage_mask = damage_mask[..., None]  # Add channel dimension

        # Apply damage (set to zeros)
        damaged_grid = grid * (1 - damage_mask)
    else:
        damaged_grid = grid

    return damaged_grid


# TPU-specific utilities
def shard_grid(grid, mesh):
    """
    Shard grid across TPU devices
    Args:
        grid: Input grid [batch, height, width, channels]
        mesh: JAX device mesh
    Returns:
        sharded_grid: Sharded grid across devices
    """
    sharding = NamedSharding(mesh, P('tp', None))
    sharded_grid = jax.device_put(grid, sharding)
    return sharded_grid


def create_tpu_mesh(device_count=8):
    """
    Create TPU device mesh for distributed computing
    Args:
        device_count: Number of TPU devices
    Returns:
        mesh: Device mesh for sharding
    """
    devices = jax.devices()[:device_count]
    mesh = Mesh(jax.device_mesh((device_count,)), ('tp',))
    return mesh


# Training utilities
@jit
def train_nca_step(state, model, batch, rng_key, loss_fn):
    """
    Single training step for NCA
    Args:
        state: Training state
        model: NCA model
        batch: Training batch
        rng_key: Random key
        loss_fn: Loss function
    Returns:
        new_state: Updated training state
        metrics: Training metrics
    """
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params, model, batch, rng_key)

    new_state = state.apply_gradients(grads=grads)

    metrics = {
        "loss": loss,
        **aux
    }

    return new_state, metrics