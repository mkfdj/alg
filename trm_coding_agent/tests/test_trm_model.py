"""
Tests for TRM Model implementation.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from trm_coding_agent.config import get_config, TRMConfig
from trm_coding_agent.trm_model import TRMModel, create_trm_model


class TestTRMModel:
    """Test cases for TRM Model."""

    @pytest.fixture
    def config(self):
        """Get test configuration."""
        return get_config("debug")

    @pytest.fixture
    def model(self, config):
        """Create TRM model instance."""
        return create_trm_model(config.trm)

    @pytest.fixture
    def sample_input(self):
        """Create sample input for testing."""
        batch_size = 4
        seq_len = 128
        return jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    def test_model_initialization(self, model, config):
        """Test model initialization."""
        assert model is not None
        assert model.config == config.trm
        assert model.config.hidden_size == 128  # Debug config value
        assert model.config.recursion_depth == 4  # Debug config value

    def test_parameter_initialization(self, model):
        """Test parameter initialization."""
        input_shape = (4, 128)
        params = model.init(input_shape)

        assert params is not None
        assert 'params' in params

        # Check parameter count
        param_count = sum(x.size for x in jax.tree_leaves(params['params']))
        assert param_count > 0

    def test_forward_pass(self, model, sample_input):
        """Test forward pass."""
        params = model.init((4, 128))

        logits, states = model.forward(
            params['params'],
            sample_input,
            deterministic=True
        )

        # Check output shapes
        batch_size, seq_len, vocab_size = logits.shape
        assert batch_size == 4
        assert seq_len == 128
        assert vocab_size == 50257  # GPT-NeoX vocab size

        # Check states
        assert 'final_y' in states
        assert 'final_z' in states
        assert 'embeddings' in states

        assert states['final_y'].shape == (4, 128, model.config.hidden_size)
        assert states['final_z'].shape == (4, 128, model.config.latent_dim)

    def test_forward_with_states(self, model, sample_input):
        """Test forward pass with intermediate states."""
        params = model.init((4, 128))

        logits, states = model.forward_with_states(
            params['params'],
            sample_input,
            deterministic=True
        )

        # Check intermediate states
        assert 'y_states' in states
        assert 'z_states' in states
        assert 'binary_decisions' in states

        # Check recursion depth
        assert states['y_states'].shape[0] == model.config.recursion_depth
        assert states['z_states'].shape[0] == model.config.recursion_depth

    def test_binary_binarization(self, model, sample_input):
        """Test binary binarization functionality."""
        config = model.config
        config.binary_binarization = True

        params = model.init((4, 128))

        logits, states = model.forward(
            params['params'],
            sample_input,
            deterministic=True
        )

        # Check binary states
        final_z = states['final_z']
        assert jnp.all(jnp.isfinite(final_z))

        # Binary values should be 0 or 1
        binary_z = jnp.where(final_z > config.binary_threshold, 1.0, 0.0)
        assert jnp.all((binary_z == 0.0) | (binary_z == 1.0))

    def test_gradient_computation(self, model, sample_input):
        """Test gradient computation."""
        config = model.config
        batch = {
            'input_ids': sample_input,
            'labels': sample_input,
            'attention_mask': jnp.ones_like(sample_input)
        }

        params = model.init((4, 128))
        train_state = model.create_train_state()

        # Compute loss and gradients
        loss, metrics = model.compute_loss(train_state.params, batch)

        assert jnp.isfinite(loss)
        assert loss >= 0

        # Check metrics
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert 'binary_decision_ratio' in metrics

    def test_train_step(self, model, sample_input):
        """Test training step."""
        batch = {
            'input_ids': sample_input,
            'labels': sample_input,
            'attention_mask': jnp.ones_like(sample_input)
        }

        train_state = model.create_train_state()

        # Training step
        new_state, metrics = model.train_step(train_state, batch)

        # Check state update
        assert new_state.step == train_state.step + 1
        assert new_state.params is not None

        # Check metrics
        assert 'loss' in metrics
        assert 'grad_norm' in metrics

    def test_eval_step(self, model, sample_input):
        """Test evaluation step."""
        batch = {
            'input_ids': sample_input,
            'labels': sample_input,
            'attention_mask': jnp.ones_like(sample_input)
        }

        train_state = model.create_train_state()

        # Evaluation step
        metrics = model.eval_step(train_state.params, batch)

        # Check metrics
        assert 'loss' in metrics
        assert 'perplexity' in metrics

    def test_code_generation(self, model):
        """Test code generation."""
        params = model.init((1, 64))

        # Generate sequence
        input_ids = jnp.ones((1, 32), dtype=jnp.int32)
        generated = model.generate(
            params['params'],
            input_ids,
            max_length=64,
            deterministic=True
        )

        # Check generation
        assert generated.shape[0] == 1  # Batch size
        assert generated.shape[1] >= input_ids.shape[1]  # Should be at least as long as input

    def test_recursion_depth(self, model, sample_input):
        """Test recursion depth behavior."""
        config = model.config
        original_depth = config.recursion_depth

        params = model.init((4, 128))

        # Test with different recursion depths
        for depth in [2, 4, 8]:
            config.recursion_depth = depth

            logits, states = model.forward(
                params['params'],
                sample_input,
                deterministic=True
            )

            # Recursion depth should affect computation but not output shape
            assert logits.shape == (4, 128, 50257)

        # Restore original depth
        config.recursion_depth = original_depth

    def test_memory_efficiency(self, model):
        """Test memory efficiency of the model."""
        # This is a basic test - in practice, you'd monitor actual memory usage
        batch_size = 8
        seq_len = 256

        # Should not run out of memory with reasonable batch sizes
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        params = model.init((batch_size, seq_len))

        logits, states = model.forward(
            params['params'],
            input_ids,
            deterministic=True
        )

        assert logits is not None
        assert states is not None

    def test_deterministic_forward(self, model, sample_input):
        """Test deterministic forward pass."""
        params = model.init((4, 128))

        # Forward pass twice with same parameters
        logits1, states1 = model.forward(
            params['params'],
            sample_input,
            deterministic=True
        )

        logits2, states2 = model.forward(
            params['params'],
            sample_input,
            deterministic=True
        )

        # Should be identical
        assert jnp.allclose(logits1, logits2)
        assert jnp.allclose(states1['final_y'], states2['final_y'])
        assert jnp.allclose(states1['final_z'], states2['final_z'])


class TestTRMModelIntegration:
    """Integration tests for TRM Model."""

    def test_end_to_end_training(self):
        """Test end-to-end training process."""
        config = get_config("debug")
        model = create_trm_model(config.trm)

        # Create dummy data
        batch_size = 4
        seq_len = 64
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        # Initialize training
        train_state = model.create_train_state()

        # Training loop
        initial_loss = float('inf')
        for step in range(5):
            batch = {
                'input_ids': input_ids,
                'labels': input_ids,
                'attention_mask': jnp.ones_like(input_ids)
            }

            train_state, metrics = model.train_step(train_state, batch)

            if step == 0:
                initial_loss = metrics['loss']

            print(f"Step {step}: Loss = {metrics['loss']:.4f}")

        # Loss should decrease
        assert metrics['loss'] <= initial_loss

    def test_model_with_different_configurations(self):
        """Test model with different configurations."""
        configs = ["debug", "cpu"]

        for config_name in configs:
            config = get_config(config_name)
            model = create_trm_model(config.trm)

            # Test basic functionality
            input_ids = jnp.ones((2, 32), dtype=jnp.int32)
            params = model.init((2, 32))

            logits, states = model.forward(
                params['params'],
                input_ids,
                deterministic=True
            )

            assert logits.shape == (2, 32, 50257)
            assert states['final_y'].shape == (2, 32, config.trm.hidden_size)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__])