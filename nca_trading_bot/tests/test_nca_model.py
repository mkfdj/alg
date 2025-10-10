"""
Test NCA model implementation
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from nca_trading_bot.config import Config
from nca_trading_bot.nca_model import AdaptiveNCA, NCAEnsemble, create_sample_pool, apply_damage


class TestAdaptiveNCA:
    """Test AdaptiveNCA class"""

    def test_nca_initialization(self):
        """Test NCA model initialization"""
        config = Config()
        model = AdaptiveNCA(config)

        assert model.config == config
        assert model.perception is not None
        assert model.update_rule is not None

    def test_nca_forward_pass(self):
        """Test NCA forward pass"""
        config = Config()
        model = AdaptiveNCA(config)

        # Create dummy input
        batch_size = 2
        height, width = config.nca_grid_size
        channels = config.nca_channels

        grid = jnp.ones((batch_size, height, width, channels))
        rng_key = jax.random.PRNGKey(42)

        # Initialize parameters
        params = model.init(rng_key, grid, rng_key, training=False)

        # Forward pass
        output = model.apply(params, grid, rng_key, training=False)

        assert output.shape == grid.shape
        assert not jnp.allclose(output, grid)  # Output should be different

    def test_nca_evolution(self):
        """Test NCA evolution over multiple steps"""
        config = Config()
        model = AdaptiveNCA(config)

        # Create dummy input
        height, width = config.nca_grid_size
        channels = config.nca_channels

        grid = jnp.ones((1, height, width, channels))
        rng_key = jax.random.PRNGKey(42)

        # Initialize parameters
        params = model.init(rng_key, grid, rng_key, training=False)

        # Evolution
        evolved_grid = model.apply(params, grid, rng_key, training=False)
        final_grid = model.evolve(evolved_grid, rng_key, steps=10)

        assert final_grid.shape == grid.shape
        assert not jnp.allclose(final_grid, grid)

    def test_grid_initialization(self):
        """Test grid initialization with seed cell"""
        config = Config()
        model = AdaptiveNCA(config)

        grid = model.initialize_grid(batch_size=2, seed=42)

        height, width = config.nca_grid_size
        channels = config.nca_channels

        assert grid.shape == (2, height, width, channels)

        # Check that center cell is initialized
        center_y, center_x = height // 2, width // 2
        assert jnp.all(grid[:, center_y, center_x, :] == 1.0)

        # Check that other cells are zero
        mask = jnp.ones((height, width), dtype=bool)
        mask = mask.at[center_y, center_x].set(False)
        assert jnp.all(jnp.all(grid[:, mask, :] == 0.0, axis=-1))

    def test_growth_decision(self):
        """Test NCA growth decision logic"""
        config = Config()
        model = AdaptiveNCA(config)

        height, width = config.nca_grid_size
        channels = config.nca_channels

        grid = jnp.ones((1, height, width, channels))

        # Low error rate should not trigger growth
        should_grow = model.should_grow(grid, error_rate=0.05, data_complexity=0.3)
        assert not should_grow

        # High error rate and high complexity should trigger growth
        should_grow = model.should_grow(grid, error_rate=0.2, data_complexity=0.8)
        assert should_grow

    def test_grid_expansion(self):
        """Test grid expansion functionality"""
        config = Config()
        model = AdaptiveNCA(config)

        height, width = config.nca_grid_size
        channels = config.nca_channels

        grid = jnp.ones((1, height, width, channels))
        expanded_grid = model.expand_grid(grid)

        assert expanded_grid.shape == (1, height + 2, width + 2, channels)


class TestNCAEnsemble:
    """Test NCA Ensemble class"""

    def test_ensemble_initialization(self):
        """Test ensemble initialization"""
        config = Config()
        config.rl_ensemble_size = 3

        ensemble = NCAEnsemble(config)

        assert ensemble.num_models == 3
        assert len(ensemble.models) == 0
        assert len(ensemble.states) == 0

    def test_ensemble_setup(self):
        """Test ensemble setup with models"""
        config = Config()
        config.rl_ensemble_size = 3

        ensemble = NCAEnsemble(config)
        rng_key = jax.random.PRNGKey(42)

        ensemble.initialize(rng_key)

        assert len(ensemble.models) == 3
        assert len(ensemble.states) == 3

        for state in ensemble.states:
            assert state.params is not None
            assert state.tx is not None

    def test_ensemble_prediction(self):
        """Test ensemble prediction"""
        config = Config()
        config.rl_ensemble_size = 3

        ensemble = NCAEnsemble(config)
        rng_key = jax.random.PRNGKey(42)

        ensemble.initialize(rng_key)

        # Create dummy input
        height, width = config.nca_grid_size
        channels = config.nca_channels

        grid = jnp.ones((1, height, width, channels))

        # Make prediction
        ensemble_pred, individual_preds = ensemble.predict(grid, rng_key)

        assert ensemble_pred.shape == grid.shape
        assert individual_preds.shape == (3, 1, height, width, channels)

        # Ensemble prediction should be average of individual predictions
        expected_ensemble = jnp.mean(individual_preds, axis=0)
        assert jnp.allclose(ensemble_pred, expected_ensemble)


class TestUtilityFunctions:
    """Test utility functions"""

    def test_sample_pool_creation(self):
        """Test sample pool creation"""
        config = Config()
        model = AdaptiveNCA(config)

        grid = model.initialize_grid(batch_size=1, seed=42)
        pool = create_sample_pool(grid, pool_size=100)

        assert pool.shape == (100, *config.nca_grid_size, config.nca_channels)

        # All samples should be identical to the original grid
        assert jnp.all(jnp.all(pool == grid[0], axis=(-1, -2, -3)))

    def test_damage_application(self):
        """Test damage application"""
        config = Config()
        model = AdaptiveNCA(config)

        grid = model.initialize_grid(batch_size=1, seed=42)

        # Apply damage
        damaged_grid = apply_damage(grid, damage_prob=1.0, max_damage_radius=3)

        # Grid should be different after damage
        assert not jnp.allclose(damaged_grid, grid)

        # Some cells should be zero
        assert jnp.any(damaged_grid == 0.0)

    def test_damage_probability_zero(self):
        """Test no damage when probability is zero"""
        config = Config()
        model = AdaptiveNCA(config)

        grid = model.initialize_grid(batch_size=1, seed=42)

        # Apply damage with zero probability
        damaged_grid = apply_damage(grid, damage_prob=0.0, max_damage_radius=3)

        # Grid should be identical
        assert jnp.allclose(damaged_grid, grid)


class TestNCAIntegration:
    """Test NCA integration with trading data"""

    def test_nca_with_trading_data(self):
        """Test NCA processing of trading data"""
        config = Config()
        model = AdaptiveNCA(config)

        # Create synthetic trading data
        sequence_length = 200
        n_features = 10

        sequence = np.random.randn(sequence_length, n_features)

        # Convert to NCA grid
        from nca_trading_bot.data_handler import DataHandler
        data_handler = DataHandler(config)

        grid = data_handler.create_nca_grid(sequence)

        assert grid.shape == (*config.nca_grid_size, config.nca_channels)

        # Test NCA evolution
        rng_key = jax.random.PRNGKey(42)
        params = model.init(rng_key, grid[None, ...], rng_key, training=False)

        evolved_grid = model.apply(params, grid[None, ...], rng_key, training=False)

        assert evolved_grid.shape == (1, *config.nca_grid_size, config.nca_channels)

    def test_target_pattern_creation(self):
        """Test target pattern creation for NCA training"""
        config = Config()
        from nca_trading_bot.data_handler import DataHandler

        data_handler = DataHandler(config)

        # Create synthetic returns
        returns = np.array([0.01, -0.02, 0.03])  # Positive, negative, positive

        target_pattern = data_handler.create_target_pattern(returns)

        assert target_pattern.shape == (*config.nca_grid_size, config.nca_channels)

        # Check RGB channels (positive returns should be green)
        assert target_pattern[0, 0, 1] > 0  # Green channel
        assert target_pattern[0, 0, 0] == 0  # Red channel

        # Check alpha channel
        assert target_pattern[0, 0, 3] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])