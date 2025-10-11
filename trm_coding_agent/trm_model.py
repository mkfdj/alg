"""
Tiny Recursive Model (TRM) implementation for coding tasks.

Based on "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)
adapted for code generation with binary thinking logic.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze, unfreeze
from flax.training.train_state import TrainState
import optax
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .config import TRMConfig


class MLPBlock(nn.Module):
    """MLP block with optional binary activation."""

    hidden_size: int
    intermediate_size: int
    dropout_rate: float = 0.1
    use_binary_activation: bool = False
    binary_threshold: float = 0.5

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # First linear layer
        x = nn.Dense(self.intermediate_size)(x)

        # Binary activation or standard activation
        if self.use_binary_activation:
            x = jnp.where(x > self.binary_threshold, 1.0, 0.0)
        else:
            x = nn.gelu(x)

        # Dropout
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)

        # Second linear layer
        x = nn.Dense(self.hidden_size)(x)

        return x


class TinyRecursiveNetwork(nn.Module):
    """Core Tiny Recursive Network following TRM architecture."""

    config: TRMConfig

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size
        )

        self.latent_projection = nn.Dense(self.config.latent_dim)
        self.output_projection = nn.Dense(self.config.vocab_size)

        # Single tiny network for both reasoning and output generation
        self.core_network = MLPBlock(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            dropout_rate=self.config.dropout_rate,
            use_binary_activation=self.config.use_binary_activations,
            binary_threshold=self.config.binary_threshold
        )

        # Layer normalization
        self.input_norm = nn.LayerNorm()
        self.latent_norm = nn.LayerNorm()
        self.output_norm = nn.LayerNorm()

        # Dropout
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        y_init: Optional[jnp.ndarray] = None,
        z_init: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
        return_all_states: bool = False
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Forward pass of TRM.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            y_init: Initial output state [batch, seq_len, hidden_size]
            z_init: Initial latent state [batch, seq_len, latent_dim]
            deterministic: Whether to use deterministic dropout
            return_all_states: Whether to return all intermediate states

        Returns:
            output_logits: Final output logits [batch, seq_len, vocab_size]
            states: Dictionary containing intermediate states
        """
        batch_size, seq_len = input_ids.shape

        # Embed input
        x = self.embedding(input_ids)
        x = self.input_norm(x)
        x = self.dropout(x, deterministic=deterministic)

        # Initialize states
        if y_init is None:
            y_init = jnp.zeros((batch_size, seq_len, self.config.hidden_size))
        if z_init is None:
            z_init = jnp.zeros((batch_size, seq_len, self.config.latent_dim))

        y = y_init
        z = z_init

        # Storage for intermediate states
        if return_all_states:
            y_states = []
            z_states = []
            binary_decisions = []

        # TRM recursion loop
        for step in range(self.config.recursion_depth):
            # Determine if this step should use gradients
            use_gradients = (step == self.config.recursion_depth - 1) or (
                step >= self.config.recursion_depth - self.config.gradient_free_steps
            )

            # Project latent state
            z_proj = self.latent_projection(z)

            # Combine inputs: x + y + z
            combined_input = x + y + z_proj

            # Apply core network
            if use_gradients:
                # Step with gradients
                network_output = self.core_network(combined_input, deterministic=deterministic)
            else:
                # Gradient-free step
                with jax.lax.stop_gradient():
                    network_output = self.core_network(combined_input, deterministic=deterministic)

            # Update latent state z
            z_new = network_output

            # Binarize latent state if enabled
            if self.config.binary_binarization:
                binary_z = jnp.where(z_new > self.config.binary_threshold, 1.0, 0.0)
                z = binary_z
            else:
                z = z_new

            # Normalize latent state
            z = self.latent_norm(z)

            # Update output state y using latent state
            y_input = y + z
            y_new = self.core_network(y_input, deterministic=deterministic)

            # Binarize output state if enabled
            if self.config.binary_binarization:
                binary_y = jnp.where(y_new > self.config.binary_threshold, 1.0, 0.0)
                y = binary_y
            else:
                y = y_new

            # Normalize output state
            y = self.output_norm(y)

            # Store intermediate states
            if return_all_states:
                y_states.append(y)
                z_states.append(z)
                binary_decisions.append(jnp.sum(binary_z) / (batch_size * seq_len * self.config.latent_dim))

        # Final output projection
        logits = self.output_projection(y)
        logits = self.output_norm(logits)

        # Prepare output states
        states = {
            'final_y': y,
            'final_z': z,
            'embeddings': x
        }

        if return_all_states:
            states.update({
                'y_states': jnp.stack(y_states),
                'z_states': jnp.stack(z_states),
                'binary_decisions': jnp.array(binary_decisions)
            })

        return logits, states


class TRMModel:
    """TRM Model wrapper with training utilities."""

    def __init__(self, config: TRMConfig):
        self.config = config
        self.network = TinyRecursiveNetwork(config)
        self.key = jax.random.PRNGKey(config.seed)

    def init(self, input_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Initialize model parameters."""
        batch_size, seq_len = input_shape
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        return self.network.init(
            {'params': self.key},
            input_ids=input_ids,
            deterministic=True
        )

    def create_train_state(
        self,
        learning_rate: float = None,
        weight_decay: float = None
    ) -> TrainState:
        """Create training state with optimizer."""
        if learning_rate is None:
            learning_rate = self.config.learning_rate
        if weight_decay is None:
            weight_decay = self.config.weight_decay

        # Initialize parameters
        input_shape = (self.config.batch_size, self.config.max_sequence_length)
        params = self.init(input_shape)['params']

        # Create optimizer with weight decay
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            b1=self.config.adam_beta1,
            b2=self.config.adam_beta2
        )

        # Gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optimizer
        )

        # Create train state
        train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=optimizer
        )

        return train_state

    @jax.jit
    def forward(
        self,
        params: Dict[str, Any],
        input_ids: jnp.ndarray,
        y_init: Optional[jnp.ndarray] = None,
        z_init: Optional[jnp.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Forward pass of the model."""
        return self.network.apply(
            {'params': params},
            input_ids=input_ids,
            y_init=y_init,
            z_init=z_init,
            deterministic=deterministic
        )

    @jax.jit
    def forward_with_states(
        self,
        params: Dict[str, Any],
        input_ids: jnp.ndarray,
        y_init: Optional[jnp.ndarray] = None,
        z_init: Optional[jnp.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Forward pass with intermediate states for analysis."""
        return self.network.apply(
            {'params': params},
            input_ids=input_ids,
            y_init=y_init,
            z_init=z_init,
            deterministic=deterministic,
            return_all_states=True
        )

    def generate(
        self,
        params: Dict[str, Any],
        input_ids: jnp.ndarray,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        deterministic: bool = False
    ) -> jnp.ndarray:
        """
        Generate code using TRM with recursive refinement.

        Args:
            params: Model parameters
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            deterministic: Whether to use deterministic generation

        Returns:
            Generated token IDs
        """
        batch_size, input_len = input_ids.shape

        # Initialize generation
        current_ids = input_ids
        generated_ids = []

        # Initial forward pass
        logits, states = self.forward(
            params,
            current_ids,
            deterministic=deterministic
        )

        # Get initial prediction
        next_token_logits = logits[:, -1, :] / temperature

        if top_k > 0:
            # Top-k sampling
            top_k_logits, top_k_indices = jax.lax.top_k(
                next_token_logits, top_k
            )
            next_token_logits = jnp.full_like(
                next_token_logits, -jnp.inf
            ).at[jax.lax.top_indices(top_k_indices, top_k)].set(top_k_logits)

        next_token_probs = jax.nn.softmax(next_token_logits, axis=-1)
        next_token = jax.random.categorical(self.key, next_token_probs)

        generated_ids.append(next_token)

        # Generate tokens recursively
        for _ in range(max_length - input_len):
            # Update current sequence
            current_ids = jnp.concatenate([current_ids, next_token[:, None]], axis=1)

            # Forward pass with previous states
            y_init = states['final_y']
            z_init = states['final_z']

            logits, states = self.forward(
                params,
                current_ids,
                y_init=y_init,
                z_init=z_init,
                deterministic=deterministic
            )

            # Get next token
            next_token_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_k_logits, top_k_indices = jax.lax.top_k(
                    next_token_logits, top_k
                )
                next_token_logits = jnp.full_like(
                    next_token_logits, -jnp.inf
                ).at[jax.lax.top_indices(top_k_indices, top_k)].set(top_k_logits)

            next_token_probs = jax.nn.softmax(next_token_logits, axis=-1)
            next_token = jax.random.categorical(self.key, next_token_probs)

            generated_ids.append(next_token)

            # Check for end token (if tokenizer has one)
            if jnp.all(next_token == 0):  # Assuming 0 is pad token
                break

        # Combine input and generated tokens
        if generated_ids:
            generated_sequence = jnp.concatenate(generated_ids, axis=1)
            full_sequence = jnp.concatenate([input_ids, generated_sequence], axis=1)
        else:
            full_sequence = input_ids

        return full_sequence

    def compute_loss(
        self,
        params: Dict[str, Any],
        batch: Dict[str, jnp.ndarray],
        deterministic: bool = False
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute loss for training.

        Args:
            params: Model parameters
            batch: Training batch containing input_ids, labels
            deterministic: Whether to use deterministic forward pass

        Returns:
            loss: Computed loss
            metrics: Dictionary of metrics
        """
        input_ids = batch['input_ids']
        labels = batch.get('labels', input_ids)
        attention_mask = batch.get('attention_mask', jnp.ones_like(input_ids))

        # Forward pass
        logits, states = self.forward(
            params,
            input_ids=input_ids,
            deterministic=deterministic
        )

        # Shift labels for next token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        # Compute cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels
        )

        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask[:, 1:]  # Shift for loss computation
            loss = loss * mask
            loss = loss.sum() / mask.sum()

        # Compute metrics
        metrics = {
            'loss': loss,
            'perplexity': jnp.exp(loss),
            'binary_decision_ratio': jnp.mean(states['final_z'] > self.config.binary_threshold),
            'final_norm_y': jnp.linalg.norm(states['final_y']),
            'final_norm_z': jnp.linalg.norm(states['final_z'])
        }

        return loss, metrics

    @jax.jit
    def train_step(
        self,
        state: TrainState,
        batch: Dict[str, jnp.ndarray]
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """Training step with gradient computation."""
        def loss_fn(params):
            return self.compute_loss(params, batch, deterministic=False)

        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)

        # Update state
        new_state = state.apply_gradients(grads=grads)

        # Add gradient norm to metrics
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_leaves(grads)))
        metrics['grad_norm'] = grad_norm

        return new_state, metrics

    @jax.jit
    def eval_step(
        self,
        params: Dict[str, Any],
        batch: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Evaluation step."""
        loss, metrics = self.compute_loss(params, batch, deterministic=True)
        return metrics


# Utility functions for TRM
def create_trm_model(config: TRMConfig) -> TRMModel:
    """Create TRM model instance."""
    return TRMModel(config)


def trm_loss_function(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Compute TRM-specific loss with stable max loss if enabled.

    Args:
        logits: Model logits [batch, seq_len, vocab_size]
        targets: Target token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]

    Returns:
        Computed loss
    """
    # Shift for next token prediction
    shift_logits = logits[:, :-1, :]
    shift_targets = targets[:, 1:]

    # Standard cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        shift_logits, shift_targets
    )

    # Apply attention mask
    if attention_mask is not None:
        mask = attention_mask[:, 1:]
        loss = loss * mask
        loss = loss.sum() / mask.sum()
    else:
        loss = loss.mean()

    # Apply stable max loss if needed (for training stability)
    # This is a simplified version - implement full stable max loss as needed
    return loss


if __name__ == "__main__":
    # Example usage
    from .config import get_config

    config = get_config("debug")
    model = create_trm_model(config)

    # Create dummy input
    batch_size, seq_len = config.batch_size, 128
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    # Initialize model
    params = model.init((batch_size, seq_len))

    # Forward pass
    logits, states = model.forward(params['params'], input_ids)

    print(f"Model initialized successfully!")
    print(f"Output shape: {logits.shape}")
    print(f"Final latent state shape: {states['final_z'].shape}")
    print(f"Number of parameters: {sum(x.size for x in jax.tree_leaves(params['params']))}")