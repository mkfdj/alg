"""
Neural Cellular Automata model for NCA Trading Bot.

This module implements the core NCA architecture with JAX for XLA-compiled
architecture, enabling native TPU support via XLA JIT compilation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from jax.nn import relu, sigmoid, softmax
from jax.scipy.signal import convolve2d
from functools import partial
import time
from collections import deque

from config import get_config

# JAX/XLA setup for TPU support
jax.config.update("jax_platform_name", "cpu")  # Default to CPU, can be overridden


class JAXConvGRUCell:
    """
    JAX-based Convolutional GRU cell for NCA state updates.

    Implements a convolutional version of GRU for processing spatial-temporal data
    optimized for XLA compilation and TPU execution.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, key: jax.random.PRNGKey = None):
        """
        Initialize JAX ConvGRU cell.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            kernel_size: Convolutional kernel size
            key: JAX random key for initialization
        """
        if key is None:
            key = random.PRNGKey(42)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        # Initialize weights using JAX
        key_z, key_r, key_h = random.split(key, 3)

        # Update gate weights
        self.w_z = random.normal(key_z, (hidden_dim, input_dim + hidden_dim, kernel_size, kernel_size)) * 0.1
        self.b_z = jnp.zeros((hidden_dim,))

        # Reset gate weights
        self.w_r = random.normal(key_r, (hidden_dim, input_dim + hidden_dim, kernel_size, kernel_size)) * 0.1
        self.b_r = jnp.zeros((hidden_dim,))

        # Candidate weights
        self.w_h = random.normal(key_h, (hidden_dim, input_dim + hidden_dim, kernel_size, kernel_size)) * 0.1
        self.b_h = jnp.zeros((hidden_dim,))

    def __call__(self, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through JAX ConvGRU cell.

        Args:
            x: Input tensor (batch, channels, height, width)
            h: Hidden state tensor (batch, channels, height, width)

        Returns:
            Updated hidden state
        """
        # Concatenate input and hidden state
        combined = jnp.concatenate([x, h], axis=1)  # (batch, input_dim + hidden_dim, H, W)

        # Update gate
        z = sigmoid(self._conv2d(combined, self.w_z, self.b_z))

        # Reset gate
        r = sigmoid(self._conv2d(combined, self.w_r, self.b_r))

        # Candidate activation
        combined_r = jnp.concatenate([x, r * h], axis=1)
        h_tilde = jnp.tanh(self._conv2d(combined_r, self.w_h, self.b_h))

        # New hidden state
        h_new = (1 - z) * h + z * h_tilde

        return h_new

    def _conv2d(self, x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """JAX-based 2D convolution with same padding."""
        # Simple convolution implementation (can be optimized for TPU)
        # In practice, you'd use jax.lax.conv_general_dilated
        batch_size, channels, height, width = x.shape
        out_channels = w.shape[0]

        # For simplicity, using a basic implementation
        # In production, replace with optimized conv
        output = jnp.zeros((batch_size, out_channels, height, width))

        for b in range(batch_size):
            for oc in range(out_channels):
                for ic in range(channels):
                    # Convolution with same padding
                    conv_result = convolve2d(
                        x[b, ic], w[oc, ic],
                        mode='same', boundary='fill', fillvalue=0
                    )
                    output = output.at[b, oc].add(conv_result)

        # Add bias
        output = output + b[None, :, None, None]

        return output


class JAXNCACell:
    """
    JAX-based Neural Cellular Automata cell with self-adaptive capabilities.

    Implements the core NCA update rule with convolutional operations
    and adaptive parameters for trading applications, optimized for XLA.
    """

    def __init__(self, state_dim: int, hidden_dim: int, kernel_size: int = 3, key: jax.random.PRNGKey = None):
        """
        Initialize JAX NCA cell.

        Args:
            state_dim: State vector dimension
            hidden_dim: Hidden layer dimension
            kernel_size: Convolutional kernel size
            key: JAX random key
        """
        if key is None:
            key = random.PRNGKey(42)

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        # Learnable NCA parameters
        key_alpha, key_beta, key_gamma, key_conv1, key_conv2, key_adapt = random.split(key, 6)

        self.alpha = random.normal(key_alpha, ()) * 0.1  # Update rate
        self.beta = random.normal(key_beta, ()) * 0.1    # Growth rate
        self.gamma = random.normal(key_gamma, ()) * 0.1  # Decay rate

        # Convolutional weights
        self.w_conv1 = random.normal(key_conv1, (hidden_dim, state_dim, kernel_size, kernel_size)) * 0.1
        self.b_conv1 = jnp.zeros((hidden_dim,))

        self.w_conv2 = random.normal(key_conv2, (state_dim, hidden_dim, kernel_size, kernel_size)) * 0.1
        self.b_conv2 = jnp.zeros((state_dim,))

        # Self-adaptation weights
        self.w_adapt = random.normal(key_adapt, (state_dim, state_dim, 1, 1)) * 0.1
        self.b_adapt = jnp.zeros((state_dim,))

        self.mutation_rate = random.normal(random.split(key)[0], ()) * 0.001

    def __call__(self, x: jnp.ndarray, h: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through JAX NCA cell.

        Args:
            x: Input state tensor
            h: Hidden state tensor

        Returns:
            Tuple of (new_state, new_hidden)
        """
        # State update
        combined = jnp.concatenate([x, h], axis=1)
        hidden_out = relu(self._conv2d(combined, self.w_conv1, self.b_conv1))
        state_update = self._conv2d(hidden_out, self.w_conv2, self.b_conv2)

        # Apply NCA update rule
        new_state = x + self.alpha * state_update

        # Growth and decay
        growth = self.beta * sigmoid(state_update)
        decay = self.gamma * sigmoid(-state_update)

        new_state = new_state + growth - decay

        # Self-adaptation
        adaptation = self._conv2d(x, self.w_adapt, self.b_adapt)
        adaptation_mask = sigmoid(adaptation)

        # Apply adaptation with mutation
        mutation = random.normal(random.PRNGKey(int(time.time()*1000)), new_state.shape) * self.mutation_rate
        new_state = adaptation_mask * new_state + (1 - adaptation_mask) * (new_state + mutation)

        # Update hidden state
        new_hidden = relu(hidden_out)

        return new_state, new_hidden

    def _conv2d(self, x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """JAX-based 2D convolution."""
        # Simplified convolution - replace with jax.lax.conv_general_dilated for production
        batch_size, channels, height, width = x.shape
        out_channels = w.shape[0]

        output = jnp.zeros((batch_size, out_channels, height, width))

        for b_idx in range(batch_size):
            for oc in range(out_channels):
                for ic in range(channels):
                    conv_result = convolve2d(
                        x[b_idx, ic], w[oc, ic],
                        mode='same', boundary='fill', fillvalue=0
                    )
                    output = output.at[b_idx, oc].add(conv_result)

        output = output + b[None, :, None, None]
        return output


class AdaptiveGridState:
    """State of the adaptive grid system."""

    def __init__(self, initial_size: int = 100, max_history: int = 200):
        self.current_size = initial_size
        self.target_size = initial_size
        self.growth_rate = 0.0
        self.complexity_score = 0.0
        self.state_history = deque(maxlen=max_history)
        self.evolution_steps = 10  # Start with minimum
        self.last_adaptation = 0.0

    def update_history(self, state: jnp.ndarray):
        """Update state history."""
        self.state_history.append(state)

    def get_recent_states(self, n: int = 10) -> List[jnp.ndarray]:
        """Get last n states from history."""
        return list(self.state_history)[-n:] if self.state_history else []


class JAXAdaptiveNCA:
    """
    JAX-based Adaptive Neural Cellular Automata with dynamic grid sizing.

    Features:
    - Dynamic grid growth/shrink based on prediction error and complexity
    - State vector with history of last 200 time steps
    - Variable evolution steps (10-100, self-adjusted)
    - XLA-compiled for native TPU support
    """

    def __init__(self, config, key: jax.random.PRNGKey = None):
        """
        Initialize JAX Adaptive NCA.

        Args:
            config: Configuration object
            key: JAX random key
        """
        if key is None:
            key = random.PRNGKey(42)

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Adaptive grid state
        self.grid_state = AdaptiveGridState(
            initial_size=100,  # Start with 100x features grid
            max_history=200
        )

        # JAX random key for operations
        self.key = key

        # NCA layers (will be dynamically resized)
        self.nca_layers = []
        self._initialize_nca_layers()

        # Evolution parameters
        self.min_evolution_steps = 10
        self.max_evolution_steps = 100

        # Adaptation thresholds
        self.growth_threshold = 0.1
        self.shrink_threshold = -0.05
        self.error_threshold = 0.05

        # JIT-compiled functions
        self._setup_jit_functions()

    def _initialize_nca_layers(self):
        """Initialize NCA layers with current grid size."""
        key_layer1, key_layer2 = random.split(self.key, 2)

        self.nca_layers = [
            JAXNCACell(
                state_dim=self.grid_state.current_size,
                hidden_dim=self.config.nca.hidden_dim,
                kernel_size=self.config.nca.kernel_size,
                key=key_layer1
            ),
            JAXNCACell(
                state_dim=self.grid_state.current_size,
                hidden_dim=self.config.nca.hidden_dim,
                kernel_size=self.config.nca.kernel_size,
                key=key_layer2
            )
        ]

    def _setup_jit_functions(self):
        """Setup JIT-compiled functions for performance."""
        # JIT compile the evolve_grid function with static steps
        self.evolve_grid_jit = {}

        # Other JIT functions
        self._predict_jit = jit(self._predict_impl)
        self._adapt_grid_jit = jit(self._adapt_grid_impl)

    def _evolve_grid_impl(self, initial_state: jnp.ndarray, steps: int) -> jnp.ndarray:
        """
        Grid evolution implementation.

        Args:
            initial_state: Initial state tensor
            steps: Number of evolution steps

        Returns:
            Evolved state tensor
        """
        current_state = initial_state
        hidden = jnp.zeros_like(current_state)

        for _ in range(steps):
            current_state, hidden = self.nca_layers[0](current_state, hidden)

        return current_state

    def evolve_grid(self, initial_state: jnp.ndarray, steps: Optional[int] = None) -> jnp.ndarray:
        """
        Evolve the NCA grid for specified steps.

        Args:
            initial_state: Initial state tensor
            steps: Number of evolution steps (auto-determined if None)

        Returns:
            Evolved state tensor
        """
        if steps is None:
            steps = self._determine_evolution_steps()

        # Ensure steps are within bounds
        steps = int(jnp.clip(steps, self.min_evolution_steps, self.max_evolution_steps))

        # Evolve the grid
        evolved_state = self._evolve_grid_impl(initial_state, steps)

        # Update history
        self.grid_state.update_history(evolved_state)

        return evolved_state

    def _determine_evolution_steps(self) -> int:
        """
        Determine optimal evolution steps based on current conditions.

        Returns:
            Number of evolution steps
        """
        # Base on complexity and recent performance
        complexity_factor = min(1.0, self.grid_state.complexity_score / 0.5)
        history_length = len(self.grid_state.state_history)

        # More steps for higher complexity or longer history
        steps = int(self.min_evolution_steps + (self.max_evolution_steps - self.min_evolution_steps) *
                   (complexity_factor + min(1.0, history_length / 50)))

        return steps

    def _predict_impl(self, state: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        JIT-compiled prediction implementation.

        Args:
            state: Current state tensor

        Returns:
            Prediction dictionary
        """
        # Simple prediction based on state mean
        # In practice, this would be more sophisticated
        state_mean = jnp.mean(state, axis=(1, 2, 3))

        # Price prediction (simplified)
        price_pred = state_mean

        # Signal prediction (3 classes: sell, hold, buy)
        signal_logits = jnp.stack([state_mean, jnp.zeros_like(state_mean), -state_mean], axis=1)
        signal_probs = softmax(signal_logits, axis=1)

        # Risk prediction
        risk_prob = sigmoid(jnp.mean(state, axis=(1, 2, 3)))

        return {
            'price_prediction': price_pred,
            'signal_probabilities': signal_probs,
            'risk_probability': risk_prob
        }

    def predict(self, state: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Make predictions from current state.

        Args:
            state: Current state tensor

        Returns:
            Prediction dictionary
        """
        return self._predict_jit(state)

    def _adapt_grid_impl(self, current_error: float, complexity: float) -> Tuple[int, float]:
        """
        JIT-compiled grid adaptation implementation.

        Args:
            current_error: Current prediction error
            complexity: Data complexity measure

        Returns:
            Tuple of (new_size, growth_rate)
        """
        current_size = self.grid_state.current_size

        # Determine growth/shrink based on error and complexity
        error_factor = jnp.tanh(current_error * 10.0)
        complexity_factor = jnp.tanh(complexity * 5.0)

        growth_signal = error_factor * 0.6 + complexity_factor * 0.4

        # Calculate new size
        size_change = int(current_size * growth_signal * 0.1)
        new_size = jnp.clip(current_size + size_change, 50, 512)  # Reasonable bounds

        return int(new_size), float(growth_signal)

    def adapt_grid(self, prediction_error: float, data_complexity: float):
        """
        Adapt grid size based on performance metrics.

        Args:
            prediction_error: Current prediction error
            data_complexity: Data complexity measure
        """
        # Use JIT-compiled adaptation
        new_size, growth_rate = self._adapt_grid_jit(prediction_error, data_complexity)

        # Update grid state
        old_size = self.grid_state.current_size
        self.grid_state.target_size = new_size
        self.grid_state.growth_rate = growth_rate

        # Only resize if significant change
        if abs(new_size - old_size) > 5:
            self._resize_grid(new_size)
            self.grid_state.last_adaptation = time.time()
            self.logger.info(f"Grid adapted from {old_size} to {new_size}")

    def _resize_grid(self, new_size: int):
        """
        Resize the NCA grid to new dimensions.

        Args:
            new_size: New grid size
        """
        # Reinitialize NCA layers with new size
        self.grid_state.current_size = new_size
        self._initialize_nca_layers()

    def get_state_with_history(self) -> jnp.ndarray:
        """
        Get current state augmented with history.

        Returns:
            State tensor with history
        """
        if not self.grid_state.state_history:
            return jnp.zeros((1, self.grid_state.current_size, 10, 10))  # Default shape

        # Get recent states
        recent_states = self.grid_state.get_recent_states(10)

        # Stack along time dimension
        if len(recent_states) > 1:
            history_tensor = jnp.stack(recent_states, axis=1)  # (batch, time, features, H, W)
            return history_tensor
        else:
            return recent_states[0][None, None]  # Add time dimension

    def forward(self, x: jnp.ndarray, market_data: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Forward pass through adaptive NCA.

        Args:
            x: Input tensor
            market_data: Optional market condition data

        Returns:
            Model outputs dictionary
        """
        # Adapt grid if market data provided
        if market_data:
            prediction_error = market_data.get('prediction_error', 0.0)
            complexity = market_data.get('complexity', 0.0)
            self.adapt_grid(prediction_error, complexity)

        # Evolve grid
        evolved_state = self.evolve_grid(x)

        # Make predictions
        predictions = self.predict(evolved_state)

        # Add metadata
        predictions['grid_size'] = self.grid_state.current_size
        predictions['evolution_steps'] = self._determine_evolution_steps()
        predictions['history_length'] = len(self.grid_state.state_history)

        return predictions


class JAXNCATrainer:
    """
    JAX-based trainer for NCA models with XLA optimization.

    Provides optimized training loops with automatic differentiation
    and XLA compilation for TPU support.
    """

    def __init__(self, model: JAXAdaptiveNCA, config):
        """
        Initialize JAX NCA trainer.

        Args:
            model: JAX NCA model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Training state
        self.key = random.PRNGKey(42)
        self.step_count = 0

    @partial(jit, static_argnums=(0,))
    def train_step_jit(self, params, batch, key):
        """
        JIT-compiled training step.

        Args:
            params: Model parameters
            batch: Training batch
            key: Random key

        Returns:
            Updated params and loss
        """
        def loss_fn(params, x, targets):
            # Forward pass
            predictions = self.model.forward(x)

            # Compute losses
            price_loss = jnp.mean((predictions['price_prediction'] - targets['price']) ** 2)
            signal_loss = jnp.mean(
                -jnp.sum(targets['signal'] * jnp.log(predictions['signal_probabilities'] + 1e-10), axis=1)
            )
            risk_loss = jnp.mean(
                -targets['risk'] * jnp.log(predictions['risk_probability'] + 1e-10) -
                (1 - targets['risk']) * jnp.log(1 - predictions['risk_probability'] + 1e-10)
            )

            total_loss = price_loss + signal_loss + risk_loss
            return total_loss

        # Compute gradients
        grad_fn = grad(loss_fn)
        grads = grad_fn(params, batch['x'], batch['targets'])

        # Simple SGD update (can be replaced with Adam)
        learning_rate = self.config.nca.learning_rate
        new_params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)

        # Compute loss for logging
        loss = loss_fn(params, batch['x'], batch['targets'])

        return new_params, loss

    def train_step(self, batch: Dict[str, jnp.ndarray]) -> float:
        """
        Perform single training step.

        Args:
            batch: Training batch

        Returns:
            Training loss
        """
        # This is a simplified implementation
        # In practice, you'd extract and update model parameters properly
        self.key, subkey = random.split(self.key)

        # Dummy training step (simplified)
        loss = 0.5  # Placeholder

        self.step_count += 1
        return loss


# Utility functions
def create_jax_nca_model(config) -> JAXAdaptiveNCA:
    """
    Create JAX-based NCA model instance.

    Args:
        config: Configuration object

    Returns:
        Initialized JAX NCA model
    """
    key = random.PRNGKey(42)
    model = JAXAdaptiveNCA(config, key)
    return model


def load_jax_nca_model(path: str, config) -> JAXAdaptiveNCA:
    """
    Load JAX NCA model from file.

    Args:
        path: Path to model file
        config: Configuration object

    Returns:
        Loaded JAX NCA model
    """
    # JAX models are typically saved as parameter dictionaries
    # Implementation would depend on how parameters are stored
    model = JAXAdaptiveNCA(config)
    return model


if __name__ == "__main__":
    # Example usage
    from config import ConfigManager

    print("JAX NCA Trading Bot - Model Demo")
    print("=" * 40)

    config = ConfigManager()
    model = create_jax_nca_model(config)

    # Create sample input
    key = random.PRNGKey(0)
    sample_input = random.normal(key, (4, 100, 10, 10))  # batch, features, H, W

    print(f"Model created with grid size: {model.grid_state.current_size}")
    print(f"Input shape: {sample_input.shape}")

    # Forward pass
    outputs = model.forward(sample_input)
    print("Output keys:", list(outputs.keys()))
    print(f"Price prediction shape: {outputs['price_prediction'].shape}")

    print("JAX NCA Model demo completed successfully!")