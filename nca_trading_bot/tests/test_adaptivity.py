"""
Tests for the adaptivity module.

This module contains comprehensive tests for all adaptive NCA components:
- AdaptiveGridManager
- GrowthStrategies
- AdaptiveNCAWrapper
- MarketConditionAnalyzer
- PerformanceMetrics
"""

import unittest
import torch
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from config import ConfigManager
from adaptivity import (
    AdaptiveGridManager,
    GrowthStrategies,
    AdaptiveNCAWrapper,
    MarketConditionAnalyzer,
    PerformanceMetrics,
    create_adaptive_grid_manager,
    create_growth_strategies,
    create_adaptive_nca_wrapper,
    create_market_condition_analyzer,
    create_performance_metrics,
    AdaptationTrigger,
    GridState,
    AdaptationMetrics
)


class TestAdaptiveGridManager(unittest.TestCase):
    """Test cases for AdaptiveGridManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.grid_manager = AdaptiveGridManager(self.config)

    def test_initialization(self):
        """Test AdaptiveGridManager initialization."""
        self.assertEqual(self.grid_manager.grid_state.current_size, self.config.nca.state_dim)
        self.assertEqual(self.grid_manager.grid_state.target_size, self.config.nca.state_dim)
        self.assertEqual(self.grid_manager.min_grid_size, 32)
        self.assertEqual(self.grid_manager.max_grid_size, 512)

    def test_analyze_grid_state(self):
        """Test grid state analysis."""
        # Create mock NCA state
        nca_state = torch.randn(1, 64, 10, 10)
        market_data = {
            'prediction_error': 0.05,
            'volatility': 0.2,
            'sharpe_ratio': 1.5
        }

        grid_state = self.grid_manager.analyze_grid_state(nca_state, market_data)

        self.assertIsInstance(grid_state, GridState)
        self.assertEqual(grid_state.current_size, self.config.nca.state_dim)
        self.assertGreaterEqual(grid_state.complexity_score, 0.0)

    def test_should_adapt(self):
        """Test adaptation decision logic."""
        current_metrics = {'prediction_error': 0.15, 'volatility': 0.4}
        target_metrics = {'max_prediction_error': 0.1, 'volatility_threshold': 0.3}

        should_adapt, reason = self.grid_manager.should_adapt(current_metrics, target_metrics)

        self.assertTrue(should_adapt)
        self.assertIn(reason, [trigger.value for trigger in AdaptationTrigger])

    def test_adapt_grid(self):
        """Test grid adaptation."""
        # Mock model
        mock_model = Mock()
        mock_model.state_dict.return_value = {'weight': torch.randn(64, 32, 3, 3)}

        # Set target size different from current
        self.grid_manager.grid_state.target_size = 128

        adapted_model, adaptation_info = self.grid_manager.adapt_grid(
            mock_model, AdaptationTrigger.PREDICTION_ERROR.value
        )

        self.assertTrue(adaptation_info['adapted'])
        self.assertEqual(adaptation_info['old_size'], 64)
        self.assertEqual(adaptation_info['new_size'], 128)

    def test_get_adaptation_metrics(self):
        """Test adaptation metrics retrieval."""
        metrics = self.grid_manager.get_adaptation_metrics()

        if 'no_adaptations' in metrics:
            self.assertTrue(metrics['no_adaptations'])
        else:
            self.assertIn('adaptation_frequency', metrics)
            self.assertIn('current_grid_size', metrics)


class TestGrowthStrategies(unittest.TestCase):
    """Test cases for GrowthStrategies."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.growth_strategies = GrowthStrategies(self.config)

    def test_initialization(self):
        """Test GrowthStrategies initialization."""
        self.assertEqual(self.growth_strategies.mutation_rate, 0.01)
        self.assertEqual(self.growth_strategies.crossover_rate, 0.7)
        self.assertEqual(self.growth_strategies.population_size, 50)

    def test_gradient_growth(self):
        """Test gradient-based growth strategy."""
        # Mock model
        mock_model = Mock()
        mock_model.state_dict.return_value = {'weight': torch.randn(64, 32, 3, 3)}

        # Apply gradient growth
        grown_model = self.growth_strategies.apply_gradient_growth(
            mock_model, target_performance=0.8
        )

        self.assertIsNotNone(grown_model)

    def test_evolutionary_growth(self):
        """Test evolutionary growth strategy."""
        # Mock model
        mock_model = Mock()
        mock_model.state_dict.return_value = {'weight': torch.randn(64, 32, 3, 3)}

        # Mock fitness function
        def fitness_func(model):
            return 0.7

        # Apply evolutionary growth
        evolved_model = self.growth_strategies.apply_evolutionary_growth(
            mock_model, fitness_func, num_generations=5
        )

        self.assertIsNotNone(evolved_model)

    def test_self_assembly_growth(self):
        """Test self-assembly growth strategy."""
        # Mock model
        mock_model = Mock()
        mock_model.state_dict.return_value = {'weight': torch.randn(64, 32, 3, 3)}

        # Create assembly rules
        assembly_rules = [
            (torch.randn(10), torch.randn(10)),
            (torch.randn(10), torch.randn(10))
        ]

        # Apply self-assembly growth
        assembled_model = self.growth_strategies.apply_self_assembly_growth(
            mock_model, assembly_rules
        )

        self.assertIsNotNone(assembled_model)


class TestAdaptiveNCAWrapper(unittest.TestCase):
    """Test cases for AdaptiveNCAWrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()

        # Mock base model
        self.mock_base_model = Mock()
        self.mock_base_model.state_dict.return_value = {'weight': torch.randn(64, 32, 3, 3)}
        self.mock_base_model.parameters.return_value = [torch.randn(10, 10)]

        self.adaptive_wrapper = AdaptiveNCAWrapper(self.mock_base_model, self.config)

    def test_initialization(self):
        """Test AdaptiveNCAWrapper initialization."""
        self.assertEqual(self.adaptive_wrapper.initial_state_dim, 100)
        self.assertEqual(self.adaptive_wrapper.max_state_dim, 512)
        self.assertEqual(self.adaptive_wrapper.history_length, 200)
        self.assertEqual(self.adaptive_wrapper.evolution_steps, 10)

    def test_forward_pass(self):
        """Test forward pass through adaptive wrapper."""
        # Mock input
        x = torch.randn(4, 10, 20)
        market_data = {
            'prediction_error': 0.05,
            'volatility': 0.2,
            'sharpe_ratio': 1.5
        }

        # Mock base model forward
        self.mock_base_model.return_value = {
            'price_prediction': torch.randn(4),
            'signal_probabilities': torch.randn(4, 3),
            'risk_probability': torch.randn(4)
        }

        outputs = self.adaptive_wrapper.forward(x, market_data)

        self.assertIn('price_prediction', outputs)
        self.assertIn('signal_probabilities', outputs)
        self.assertIn('adaptation_info', outputs)

    def test_adaptation_signal_analysis(self):
        """Test adaptation signal analysis."""
        market_data = {
            'prediction_error': 0.15,
            'volatility': 0.4,
            'sharpe_ratio': 0.8
        }

        adaptation_signal = self.adaptive_wrapper._analyze_adaptation_needs(market_data)

        self.assertIsInstance(adaptation_signal, float)

    def test_state_dimension_adaptation(self):
        """Test state dimension adaptation."""
        # Test growth
        self.adaptive_wrapper._adapt_state_dimension(0.2)
        self.assertEqual(self.adaptive_wrapper.current_state_dim, 100)

        # Test shrinking
        self.adaptive_wrapper._adapt_state_dimension(-0.1)
        self.assertEqual(self.adaptive_wrapper.current_state_dim, 100)

    def test_evolution_steps_adaptation(self):
        """Test evolution steps adaptation."""
        # Test increase
        self.adaptive_wrapper._adapt_evolution_steps(0.3)
        self.assertEqual(self.adaptive_wrapper.evolution_steps, 15)

        # Test decrease
        self.adaptive_wrapper._adapt_evolution_steps(-0.2)
        self.assertEqual(self.adaptive_wrapper.evolution_steps, 10)


class TestMarketConditionAnalyzer(unittest.TestCase):
    """Test cases for MarketConditionAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.market_analyzer = MarketConditionAnalyzer(self.config)

    def test_initialization(self):
        """Test MarketConditionAnalyzer initialization."""
        self.assertEqual(self.market_analyzer.volatility_window, 20)
        self.assertEqual(self.market_analyzer.volume_window, 50)
        self.assertEqual(self.market_analyzer.volatility_threshold, 0.25)

    def test_analyze_market_conditions(self):
        """Test market condition analysis."""
        market_data = {
            'prices': {'values': np.random.randn(100).tolist()},
            'volumes': {'values': np.random.randn(100).tolist()},
            'asset_prices': {'matrix': np.random.randn(5, 100).tolist()}
        }

        analysis_results = self.market_analyzer.analyze_market_conditions(market_data)

        self.assertIsInstance(analysis_results, dict)
        self.assertIn('volatility', analysis_results)

    def test_volatility_computation(self):
        """Test volatility computation."""
        price_series = jnp.array(np.random.randn(100))
        volatility = self.market_analyzer._jax_compute_volatility(price_series, 20)

        self.assertIsInstance(volatility, float)
        self.assertGreaterEqual(volatility, 0.0)

    def test_volume_pattern_analysis(self):
        """Test volume pattern analysis."""
        volume_series = jnp.array(np.random.randn(100))
        price_series = jnp.array(np.random.randn(100))

        analysis = self.market_analyzer._jax_analyze_volume_patterns(
            volume_series, price_series, 50
        )

        self.assertIsInstance(analysis, dict)
        self.assertIn('volume_trend', analysis)
        self.assertIn('divergence', analysis)

    def test_correlation_structure_analysis(self):
        """Test correlation structure analysis."""
        price_matrix = jnp.array(np.random.randn(5, 100))
        complexity = self.market_analyzer._jax_compute_correlation_structure(price_matrix, 30)

        self.assertIsInstance(complexity, float)
        self.assertGreaterEqual(complexity, 0.0)

    def test_momentum_analysis(self):
        """Test momentum analysis."""
        price_series = jnp.array(np.random.randn(100))
        momentum_analysis = self.market_analyzer._jax_analyze_momentum(price_series, 14)

        self.assertIsInstance(momentum_analysis, dict)
        self.assertIn('momentum', momentum_analysis)
        self.assertIn('rsi', momentum_analysis)

    def test_adaptation_triggering(self):
        """Test adaptation trigger logic."""
        analysis_results = {
            'volatility': 0.4,  # Above threshold
            'divergence': 2.0,  # High divergence
            'correlation_complexity': 0.8,  # High complexity
            'momentum': 0.15  # Above threshold
        }

        should_trigger, reason = self.market_analyzer.should_trigger_adaptation(analysis_results)

        self.assertTrue(should_trigger)
        self.assertIn(reason, [
            AdaptationTrigger.VOLATILITY_SPIKE.value,
            "volume_divergence",
            AdaptationTrigger.MARKET_REGIME_CHANGE.value,
            "momentum_shift"
        ])

    def test_adaptation_recommendations(self):
        """Test adaptation recommendations."""
        analysis_results = {
            'volatility': 0.4,
            'volume_trend': 0.2,
            'correlation_complexity': 0.8
        }

        recommendations = self.market_analyzer.get_adaptation_recommendations(analysis_results)

        self.assertIsInstance(recommendations, dict)
        self.assertIn('grow', recommendations)
        self.assertIn('shrink', recommendations)
        self.assertIn('reasons', recommendations)
        self.assertIn('confidence', recommendations)


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for PerformanceMetrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.performance_metrics = PerformanceMetrics(self.config)

    def test_initialization(self):
        """Test PerformanceMetrics initialization."""
        current_metrics = self.performance_metrics.get_current_metrics()
        self.assertIsInstance(current_metrics, AdaptationMetrics)
        self.assertEqual(current_metrics.growth_efficiency, 0.0)

    def test_update_metrics(self):
        """Test metrics update."""
        adaptation_info = {
            'adapted': True,
            'old_size': 64,
            'new_size': 128
        }

        performance_data = {
            'before': 0.7,
            'after': 0.8,
            'computational_cost': 1.2,
            'prediction_improvement': 0.1,
            'memory_usage': 1.0
        }

        self.performance_metrics.update_metrics(adaptation_info, performance_data)

        current_metrics = self.performance_metrics.get_current_metrics()
        self.assertGreater(current_metrics.growth_efficiency, 0.0)

    def test_aggregated_metrics(self):
        """Test aggregated metrics computation."""
        # Add some test data
        for i in range(10):
            adaptation_info = {'adapted': True, 'old_size': 64, 'new_size': 128}
            performance_data = {
                'before': 0.7,
                'after': 0.8 + i * 0.01,
                'computational_cost': 1.0 + i * 0.1,
                'prediction_improvement': 0.1 + i * 0.01,
                'memory_usage': 1.0
            }
            self.performance_metrics.update_metrics(adaptation_info, performance_data)

        aggregated_metrics = self.performance_metrics.get_aggregated_metrics()

        self.assertIsInstance(aggregated_metrics, dict)
        self.assertIn('avg_growth_efficiency', aggregated_metrics)
        self.assertIn('total_prediction_improvement', aggregated_metrics)

    def test_efficiency_trends(self):
        """Test efficiency trends retrieval."""
        # Add some test data
        for i in range(5):
            adaptation_info = {'adapted': True, 'old_size': 64, 'new_size': 128}
            performance_data = {
                'before': 0.7,
                'after': 0.8,
                'computational_cost': 1.0,
                'prediction_improvement': 0.1,
                'memory_usage': 1.0
            }
            self.performance_metrics.update_metrics(adaptation_info, performance_data)

        trends = self.performance_metrics.get_efficiency_trends()

        self.assertIsInstance(trends, dict)
        self.assertIn('growth_efficiency', trends)
        self.assertIn('computational_cost', trends)
        self.assertEqual(len(trends['growth_efficiency']), 5)

    def test_reset_metrics(self):
        """Test metrics reset."""
        # Add some data
        adaptation_info = {'adapted': True, 'old_size': 64, 'new_size': 128}
        performance_data = {'before': 0.7, 'after': 0.8, 'computational_cost': 1.0}
        self.performance_metrics.update_metrics(adaptation_info, performance_data)

        # Reset
        self.performance_metrics.reset_metrics()

        # Check if reset
        current_metrics = self.performance_metrics.get_current_metrics()
        self.assertEqual(current_metrics.growth_efficiency, 0.0)
        self.assertEqual(current_metrics.computational_cost, 0.0)


class TestFactoryFunctions(unittest.TestCase):
    """Test cases for factory functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()

    def test_create_adaptive_grid_manager(self):
        """Test AdaptiveGridManager factory function."""
        manager = create_adaptive_grid_manager(self.config)
        self.assertIsInstance(manager, AdaptiveGridManager)

    def test_create_growth_strategies(self):
        """Test GrowthStrategies factory function."""
        strategies = create_growth_strategies(self.config)
        self.assertIsInstance(strategies, GrowthStrategies)

    def test_create_adaptive_nca_wrapper(self):
        """Test AdaptiveNCAWrapper factory function."""
        mock_model = Mock()
        wrapper = create_adaptive_nca_wrapper(mock_model, self.config)
        self.assertIsInstance(wrapper, AdaptiveNCAWrapper)

    def test_create_market_condition_analyzer(self):
        """Test MarketConditionAnalyzer factory function."""
        analyzer = create_market_condition_analyzer(self.config)
        self.assertIsInstance(analyzer, MarketConditionAnalyzer)

    def test_create_performance_metrics(self):
        """Test PerformanceMetrics factory function."""
        metrics = create_performance_metrics(self.config)
        self.assertIsInstance(metrics, PerformanceMetrics)


class TestJAXOptimizations(unittest.TestCase):
    """Test cases for JAX optimizations."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.grid_manager = AdaptiveGridManager(self.config)
        self.market_analyzer = MarketConditionAnalyzer(self.config)
        self.performance_metrics = PerformanceMetrics(self.config)

    def test_jax_volatility_computation(self):
        """Test JAX volatility computation."""
        price_series = jnp.array(np.random.randn(100))
        volatility = self.market_analyzer._jax_compute_volatility(price_series, 20)

        self.assertIsInstance(volatility, float)
        self.assertGreaterEqual(volatility, 0.0)

    def test_jax_efficiency_computation(self):
        """Test JAX efficiency computation."""
        efficiency = self.performance_metrics._jax_compute_efficiency(0.7, 0.8, 0.1)

        self.assertIsInstance(efficiency, float)
        self.assertGreater(efficiency, 0.0)

    def test_jax_cost_benefit_analysis(self):
        """Test JAX cost-benefit analysis."""
        costs = jnp.array([1.0, 1.2, 0.8])
        benefits = jnp.array([0.1, 0.15, 0.12])

        analysis = self.performance_metrics._jax_analyze_cost_benefit(costs, benefits)

        self.assertIsInstance(analysis, dict)
        self.assertIn('total_cost', analysis)
        self.assertIn('roi', analysis)


class TestIntegration(unittest.TestCase):
    """Integration tests for adaptive components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()

        # Create all adaptive components
        self.grid_manager = create_adaptive_grid_manager(self.config)
        self.growth_strategies = create_growth_strategies(self.config)
        self.market_analyzer = create_market_condition_analyzer(self.config)
        self.performance_metrics = create_performance_metrics(self.config)

    def test_full_adaptation_workflow(self):
        """Test complete adaptation workflow."""
        # Mock model
        mock_model = Mock()
        mock_model.state_dict.return_value = {'weight': torch.randn(64, 32, 3, 3)}

        # Analyze market conditions
        market_data = {
            'prices': {'values': np.random.randn(100).tolist()},
            'volumes': {'values': np.random.randn(100).tolist()},
            'asset_prices': {'matrix': np.random.randn(5, 100).tolist()},
            'prediction_error': 0.15,
            'sharpe_ratio': 0.8,
            'volatility': 0.4
        }

        analysis_results = self.market_analyzer.analyze_market_conditions(market_data)

        # Check if adaptation should be triggered
        should_adapt, reason = self.market_analyzer.should_trigger_adaptation(analysis_results)

        if should_adapt:
            # Analyze grid state
            nca_state = torch.randn(1, 64, 10, 10)
            grid_state = self.grid_manager.analyze_grid_state(nca_state, market_data)

            # Adapt grid
            adapted_model, adaptation_info = self.grid_manager.adapt_grid(
                mock_model, reason
            )

            # Update performance metrics
            performance_data = {
                'before': 0.7,
                'after': 0.8,
                'computational_cost': 1.2,
                'prediction_improvement': 0.1,
                'memory_usage': 1.0
            }

            self.performance_metrics.update_metrics(adaptation_info, performance_data)

            # Verify workflow completed
            self.assertTrue(adaptation_info['adapted'])
            self.assertGreater(self.performance_metrics.get_current_metrics().growth_efficiency, 0.0)

    def test_adaptive_model_integration(self):
        """Test adaptive model integration."""
        # Mock base model
        mock_base_model = Mock()
        mock_base_model.state_dict.return_value = {'weight': torch.randn(64, 32, 3, 3)}
        mock_base_model.parameters.return_value = [torch.randn(10, 10)]
        mock_base_model.return_value = {
            'price_prediction': torch.randn(4),
            'signal_probabilities': torch.randn(4, 3),
            'risk_probability': torch.randn(4)
        }

        # Create adaptive wrapper
        adaptive_model = AdaptiveNCAWrapper(mock_base_model, self.config)

        # Test forward pass with market data
        x = torch.randn(4, 10, 20)
        market_data = {
            'prediction_error': 0.05,
            'volatility': 0.2,
            'sharpe_ratio': 1.5
        }

        outputs = adaptive_model.forward(x, market_data)

        # Verify outputs
        self.assertIn('price_prediction', outputs)
        self.assertIn('signal_probabilities', outputs)
        self.assertIn('adaptation_info', outputs)
        self.assertIn('current_state_dim', outputs['adaptation_info'])


if __name__ == '__main__':
    unittest.main()