"""
Adaptive Neural Cellular Automata module for NCA Trading Bot.

This module implements dynamic growth logic for Neural Cellular Automata,
including adaptive grid management, growth strategies, market condition analysis,
and performance tracking with JAX/XLA optimizations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
import time
from collections import deque

from config import get_config

# TPU/XLA imports
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False
    xm = None
    pl = None


class AdaptationTrigger(Enum):
    """Triggers for NCA adaptation."""
    VOLATILITY_SPIKE = "volatility_spike"
    PREDICTION_ERROR = "prediction_error"
    SHARPE_RATIO_DROP = "sharpe_ratio_drop"
    MARKET_REGIME_CHANGE = "market_regime_change"
    DATA_COMPLEXITY_INCREASE = "data_complexity_increase"
    MANUAL_TRIGGER = "manual_trigger"


@dataclass
class GridState:
    """Current state of the adaptive grid."""
    current_size: int
    target_size: int
    growth_rate: float
    shrink_rate: float
    complexity_score: float
    adaptation_history: List[Dict] = field(default_factory=list)
    last_adaptation: float = 0.0  # timestamp


@dataclass
class AdaptationMetrics:
    """Metrics for tracking adaptation performance."""
    growth_efficiency: float = 0.0
    computational_cost: float = 0.0
    prediction_improvement: float = 0.0
    memory_usage: float = 0.0
    adaptation_frequency: float = 0.0
    convergence_time: float = 0.0


class AdaptiveGridManager:
    """
    Manages dynamic grid growth/shrinking for Neural Cellular Automata.

    This class handles the adaptive resizing of NCA grids based on:
    - Prediction error thresholds
    - Data complexity metrics
    - Sharpe ratio performance
    - Market volatility conditions
    """

    def __init__(self, config):
        """
        Initialize AdaptiveGridManager.

        Args:
            config: Configuration object with adaptation parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Grid state management
        self.grid_state = GridState(
            current_size=config.nca.state_dim,
            target_size=config.nca.state_dim,
            growth_rate=config.nca.adaptation_rate,
            shrink_rate=config.nca.adaptation_rate * 0.5,
            complexity_score=0.0
        )

        # Adaptation parameters
        self.min_grid_size = 32
        self.max_grid_size = 512
        self.growth_threshold = 0.1
        self.shrink_threshold = -0.05
        self.stability_period = 100  # steps before considering adaptation
        self.adaptation_cooldown = 50  # steps after adaptation

        # Performance tracking
        self.performance_history = deque(maxlen=200)
        self.adaptation_triggers = []

        # JAX/XLA optimizations
        self._setup_jax_optimizations()

    def _setup_jax_optimizations(self):
        """Setup JAX-compatible functions for performance."""
        # JAX-compiled grid analysis functions
        self._jax_analyze_complexity = jit(self._jax_analyze_complexity_impl)
        self._jax_compute_growth_rate = jit(self._jax_compute_growth_rate_impl)
        self._jax_evaluate_adaptation = jit(self._jax_evaluate_adaptation_impl)

    def _jax_analyze_complexity_impl(self, state_tensor: jnp.ndarray) -> float:
        """
        JAX implementation of complexity analysis.

        Args:
            state_tensor: Current NCA state tensor

        Returns:
            Complexity score
        """
        # Compute entropy-based complexity measure
        flat_state = state_tensor.flatten()
        probs = jnp.abs(flat_state) / jnp.sum(jnp.abs(flat_state))
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-10))
        return float(entropy)

    def _jax_compute_growth_rate_impl(self, error_rate: float, volatility: float,
                                    sharpe_ratio: float) -> float:
        """
        JAX implementation of growth rate computation.

        Args:
            error_rate: Current prediction error rate
            volatility: Market volatility measure
            sharpe_ratio: Current Sharpe ratio

        Returns:
            Computed growth rate
        """
        # Adaptive growth rate based on multiple factors
        error_factor = jnp.tanh(error_rate * 10.0)
        volatility_factor = jnp.tanh(volatility * 5.0)
        sharpe_factor = jnp.tanh(sharpe_ratio * 2.0)

        growth_rate = error_factor * 0.5 + volatility_factor * 0.3 + sharpe_factor * 0.2
        return float(jnp.clip(growth_rate, -0.1, 0.5))

    def _jax_evaluate_adaptation_impl(self, current_metrics: Dict[str, float],
                                    target_metrics: Dict[str, float]) -> bool:
        """
        JAX implementation of adaptation evaluation.

        Args:
            current_metrics: Current performance metrics
            target_metrics: Target performance metrics

        Returns:
            Whether adaptation is needed
        """
        # Compute adaptation necessity score
        metric_diffs = []
        for key in current_metrics.keys():
            if key in target_metrics:
                diff = abs(current_metrics[key] - target_metrics[key])
                metric_diffs.append(diff)

        avg_diff = jnp.mean(jnp.array(metric_diffs))
        return bool(avg_diff > 0.1)

    def analyze_grid_state(self, nca_state: torch.Tensor,
                          market_data: Dict[str, float]) -> GridState:
        """
        Analyze current grid state and determine if adaptation is needed.

        Args:
            nca_state: Current NCA state tensor
            market_data: Market condition data

        Returns:
            Updated grid state
        """
        # Convert to JAX for analysis if using JAX backend
        if hasattr(nca_state, 'device') and nca_state.device.type == 'cpu':
            # Use JAX for CPU-based analysis
            state_np = nca_state.detach().cpu().numpy()
            complexity_score = self._jax_analyze_complexity(jnp.array(state_np))
        else:
            # Use PyTorch for GPU/TPU analysis
            complexity_score = self._analyze_complexity_torch(nca_state)

        # Update complexity score
        self.grid_state.complexity_score = complexity_score

        # Compute growth rate based on market conditions
        error_rate = market_data.get('prediction_error', 0.0)
        volatility = market_data.get('volatility', 0.0)
        sharpe_ratio = market_data.get('sharpe_ratio', 0.0)

        growth_rate = self._jax_compute_growth_rate(error_rate, volatility, sharpe_ratio)

        # Update target size based on growth rate
        current_size = self.grid_state.current_size
        size_change = int(current_size * growth_rate)

        # Apply bounds and constraints
        new_target_size = np.clip(
            current_size + size_change,
            self.min_grid_size,
            self.max_grid_size
        )

        self.grid_state.target_size = new_target_size
        self.grid_state.growth_rate = abs(growth_rate)

        return self.grid_state

    def _analyze_complexity_torch(self, state_tensor: torch.Tensor) -> float:
        """
        PyTorch implementation of complexity analysis.

        Args:
            state_tensor: Current NCA state tensor

        Returns:
            Complexity score
        """
        # Compute entropy-based complexity measure
        flat_state = state_tensor.flatten()
        probs = torch.abs(flat_state) / torch.sum(torch.abs(flat_state))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return entropy.item()

    def should_adapt(self, current_metrics: Dict[str, float],
                    target_metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Determine if grid adaptation is needed.

        Args:
            current_metrics: Current performance metrics
            target_metrics: Target performance metrics

        Returns:
            Tuple of (should_adapt, reason)
        """
        # Check if we're in cooldown period
        if len(self.adaptation_triggers) > 0:
            last_trigger_time = self.adaptation_triggers[-1].get('timestamp', 0)
            if time.time() - last_trigger_time < self.adaptation_cooldown:
                return False, "cooldown"

        # Use JAX evaluation
        needs_adaptation = self._jax_evaluate_adaptation(current_metrics, target_metrics)

        if not needs_adaptation:
            return False, "no_change_needed"

        # Determine specific trigger
        trigger_reason = self._determine_trigger_reason(current_metrics, target_metrics)

        return True, trigger_reason

    def _determine_trigger_reason(self, current_metrics: Dict[str, float],
                                target_metrics: Dict[str, float]) -> str:
        """
        Determine the specific reason for adaptation trigger.

        Args:
            current_metrics: Current performance metrics
            target_metrics: Target performance metrics

        Returns:
            Trigger reason string
        """
        # Check various trigger conditions
        if (current_metrics.get('prediction_error', 0) >
            target_metrics.get('max_prediction_error', 0.1)):
            return AdaptationTrigger.PREDICTION_ERROR.value

        if (current_metrics.get('volatility', 0) >
            target_metrics.get('volatility_threshold', 0.3)):
            return AdaptationTrigger.VOLATILITY_SPIKE.value

        if (current_metrics.get('sharpe_ratio', 0) <
            target_metrics.get('min_sharpe_ratio', 0.5)):
            return AdaptationTrigger.SHARPE_RATIO_DROP.value

        return AdaptationTrigger.MARKET_REGIME_CHANGE.value

    def adapt_grid(self, current_model: nn.Module,
                  adaptation_reason: str) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Perform grid adaptation on the model.

        Args:
            current_model: Current NCA model
            adaptation_reason: Reason for adaptation

        Returns:
            Tuple of (adapted_model, adaptation_info)
        """
        old_size = self.grid_state.current_size
        new_size = self.grid_state.target_size

        if old_size == new_size:
            return current_model, {"adapted": False, "reason": "no_size_change"}

        # Create new model with adapted size
        adapted_model = self._resize_model(current_model, old_size, new_size)

        # Update grid state
        self.grid_state.current_size = new_size
        self.grid_state.last_adaptation = time.time()

        # Record adaptation
        adaptation_record = {
            'timestamp': time.time(),
            'old_size': old_size,
            'new_size': new_size,
            'reason': adaptation_reason,
            'growth_rate': self.grid_state.growth_rate
        }
        self.adaptation_triggers.append(adaptation_record)

        adaptation_info = {
            "adapted": True,
            "old_size": old_size,
            "new_size": new_size,
            "reason": adaptation_reason,
            "adaptation_record": adaptation_record
        }

        self.logger.info(f"Grid adapted from {old_size} to {new_size} due to {adaptation_reason}")
        return adapted_model, adaptation_info

    def _resize_model(self, model: nn.Module, old_size: int, new_size: int) -> nn.Module:
        """
        Resize model parameters to new grid size.

        Args:
            model: Current model
            old_size: Current grid size
            new_size: Target grid size

        Returns:
            Model with resized parameters
        """
        # Create new model instance with target size
        new_model = type(model)(self._create_resized_config(old_size, new_size))

        # Transfer learned parameters with appropriate resizing
        new_state_dict = new_model.state_dict()

        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in new_state_dict:
                    if 'weight' in name and len(param.shape) >= 2:
                        # Resize convolutional weights
                        new_param = self._resize_conv_weight(param, new_size)
                    elif 'bias' in name:
                        # Resize bias terms
                        new_param = self._resize_bias(param, new_size)
                    else:
                        # Keep other parameters as-is or interpolate
                        new_param = self._interpolate_parameter(param, old_size, new_size)

                    new_state_dict[name] = new_param

        new_model.load_state_dict(new_state_dict)
        return new_model

    def _create_resized_config(self, old_size: int, new_size: int) -> Any:
        """
        Create configuration with resized dimensions.

        Args:
            old_size: Current size
            new_size: Target size

        Returns:
            Updated configuration
        """
        # Create a copy of the config and update dimensions
        config = self.config
        config.nca.state_dim = new_size
        return config

    def _resize_conv_weight(self, weight: torch.Tensor, new_size: int) -> torch.Tensor:
        """
        Resize convolutional weight tensor.

        Args:
            weight: Weight tensor to resize
            new_size: Target size

        Returns:
            Resized weight tensor
        """
        if len(weight.shape) == 4:  # Conv2d weight
            out_channels, in_channels, k_h, k_w = weight.shape
            # Resize input channels
            if in_channels == self.grid_state.current_size:
                weight_resized = F.interpolate(
                    weight, size=(new_size, k_h, k_w),
                    mode='trilinear', align_corners=False
                )
                return weight_resized
        return weight

    def _resize_bias(self, bias: torch.Tensor, new_size: int) -> torch.Tensor:
        """
        Resize bias tensor.

        Args:
            bias: Bias tensor to resize
            new_size: Target size

        Returns:
            Resized bias tensor
        """
        if bias.shape[0] == self.grid_state.current_size:
            # Interpolate bias values
            bias_np = bias.detach().cpu().numpy()
            x_old = np.linspace(0, 1, len(bias_np))
            x_new = np.linspace(0, 1, new_size)
            bias_new = np.interp(x_new, x_old, bias_np)
            return torch.tensor(bias_new, dtype=bias.dtype, device=bias.device)
        return bias

    def _interpolate_parameter(self, param: torch.Tensor, old_size: int,
                             new_size: int) -> torch.Tensor:
        """
        Interpolate parameter tensor to new size.

        Args:
            param: Parameter tensor
            old_size: Current size
            new_size: Target size

        Returns:
            Interpolated parameter tensor
        """
        if len(param.shape) == 1 and param.shape[0] == old_size:
            # 1D parameter - interpolate
            param_np = param.detach().cpu().numpy()
            x_old = np.linspace(0, 1, len(param_np))
            x_new = np.linspace(0, 1, new_size)
            param_new = np.interp(x_new, x_old, param_np)
            return torch.tensor(param_new, dtype=param.dtype, device=param.device)
        return param

    def get_adaptation_metrics(self) -> Dict[str, float]:
        """
        Get current adaptation performance metrics.

        Returns:
            Dictionary with adaptation metrics
        """
        if not self.adaptation_triggers:
            return {"no_adaptations": True}

        recent_adaptations = [
            trigger for trigger in self.adaptation_triggers
            if time.time() - trigger['timestamp'] < 3600  # Last hour
        ]

        if not recent_adaptations:
            return {"no_recent_adaptations": True}

        # Calculate metrics
        adaptation_frequency = len(recent_adaptations) / max(1, len(self.performance_history))
        avg_size_change = np.mean([
            abs(trigger['new_size'] - trigger['old_size'])
            for trigger in recent_adaptations
        ])

        return {
            "adaptation_frequency": adaptation_frequency,
            "avg_size_change": avg_size_change,
            "total_adaptations": len(self.adaptation_triggers),
            "current_grid_size": self.grid_state.current_size,
            "target_grid_size": self.grid_state.target_size,
            "complexity_score": self.grid_state.complexity_score
        }


class GrowthStrategy(Enum):
    """Different strategies for NCA growth."""
    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    SELF_ASSEMBLY = "self_assembly"
    HYBRID = "hybrid"


class GrowthStrategies:
    """
    Different strategies for growing Neural Cellular Automata.

    Implements various approaches from research papers:
    - Gradient-based growth
    - Evolutionary algorithms
    - Self-assembly mechanisms
    - Hardware configuration optimization
    """

    def __init__(self, config):
        """
        Initialize GrowthStrategies.

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Strategy parameters
        self.mutation_rate = 0.01
        self.crossover_rate = 0.7
        self.elite_fraction = 0.1
        self.population_size = 50

        # JAX optimizations
        self._setup_jax_growth_functions()

    def _setup_jax_growth_functions(self):
        """Setup JAX-optimized growth functions."""
        self._jax_gradient_growth = jit(self._jax_gradient_growth_impl)
        self._jax_evolutionary_growth = jit(self._jax_evolutionary_growth_impl)
        self._jax_self_assembly = jit(self._jax_self_assembly_impl)

    def _jax_gradient_growth_impl(self, current_state: jnp.ndarray,
                                target_state: jnp.ndarray,
                                learning_rate: float) -> jnp.ndarray:
        """
        JAX implementation of gradient-based growth.

        Args:
            current_state: Current NCA state
            target_state: Target state
            learning_rate: Learning rate for growth

        Returns:
            Grown state
        """
        # Compute gradient-based growth direction
        error = target_state - current_state
        growth_direction = jnp.tanh(error * learning_rate)

        # Apply growth with momentum
        new_state = current_state + growth_direction * 0.1
        return new_state

    def _jax_evolutionary_growth_impl(self, population: jnp.ndarray,
                                    fitness_scores: jnp.ndarray,
                                    mutation_rate: float) -> jnp.ndarray:
        """
        JAX implementation of evolutionary growth.

        Args:
            population: Population of NCA states
            fitness_scores: Fitness scores for each individual
            mutation_rate: Mutation rate

        Returns:
            Evolved population
        """
        # Select elite individuals
        elite_size = max(1, int(len(population) * self.elite_fraction))
        elite_indices = jnp.argsort(fitness_scores)[-elite_size:]

        # Create new population through crossover and mutation
        new_population = population[elite_indices]

        # Add mutated offspring
        for _ in range(len(population) - elite_size):
            parent1_idx = jax.random.choice(jnp.arange(len(elite_indices)))
            parent2_idx = jax.random.choice(jnp.arange(len(elite_indices)))

            parent1 = population[elite_indices[parent1_idx]]
            parent2 = population[elite_indices[parent2_idx]]

            # Crossover
            crossover_point = jax.random.randint(0, parent1.size)
            offspring = jnp.concatenate([
                parent1[:crossover_point],
                parent2[crossover_point:]
            ])

            # Mutation
            mutation_mask = jax.random.bernoulli(jax.random.PRNGKey(0), mutation_rate, offspring.shape)
            mutation = jax.random.normal(jax.random.PRNGKey(1), offspring.shape) * 0.1
            offspring = jnp.where(mutation_mask, offspring + mutation, offspring)

            new_population = jnp.vstack([new_population, offspring])

        return new_population

    def _jax_self_assembly_impl(self, current_state: jnp.ndarray,
                              assembly_rules: jnp.ndarray) -> jnp.ndarray:
        """
        JAX implementation of self-assembly growth.

        Args:
            current_state: Current NCA state
            assembly_rules: Assembly rules for growth

        Returns:
            Self-assembled state
        """
        # Apply assembly rules to current state
        # This implements pattern-based growth rules
        assembled_state = current_state

        for rule in assembly_rules:
            # Apply each assembly rule
            pattern, replacement = rule
            # Pattern matching and replacement logic
            matches = jnp.all(current_state == pattern, axis=-1)
            assembled_state = jnp.where(matches, replacement, assembled_state)

        return assembled_state

    def apply_gradient_growth(self, current_model: nn.Module,
                            target_performance: float,
                            learning_rate: float = None) -> nn.Module:
        """
        Apply gradient-based growth strategy.

        Args:
            current_model: Current NCA model
            target_performance: Target performance level
            learning_rate: Learning rate for growth

        Returns:
            Model with gradient-based growth applied
        """
        if learning_rate is None:
            learning_rate = self.config.nca.learning_rate

        # Get current model state
        current_state = self._get_model_state(current_model)

        # Compute target state based on performance requirements
        target_state = self._compute_target_state(current_state, target_performance)

        # Apply JAX-optimized gradient growth
        state_np = current_state.detach().cpu().numpy()
        target_np = target_state.detach().cpu().numpy()

        grown_state_np = self._jax_gradient_growth(
            jnp.array(state_np),
            jnp.array(target_np),
            learning_rate
        )

        grown_state = torch.tensor(grown_state_np, dtype=current_state.dtype,
                                  device=current_state.device)

        # Update model with grown state
        updated_model = self._update_model_state(current_model, grown_state)
        return updated_model

    def apply_evolutionary_growth(self, current_model: nn.Module,
                                fitness_function: Callable,
                                num_generations: int = 10) -> nn.Module:
        """
        Apply evolutionary growth strategy.

        Args:
            current_model: Current NCA model
            fitness_function: Function to evaluate model fitness
            num_generations: Number of evolutionary generations

        Returns:
            Model with evolutionary growth applied
        """
        # Initialize population
        population = self._initialize_population(current_model)

        for generation in range(num_generations):
            # Evaluate fitness
            fitness_scores = jnp.array([
                fitness_function(self._population_member_to_model(member))
                for member in population
            ])

            # Evolve population using JAX
            population_np = np.array([member.detach().cpu().numpy() for member in population])
            fitness_np = np.array(fitness_scores)

            evolved_population_np = self._jax_evolutionary_growth(
                jnp.array(population_np),
                jnp.array(fitness_np),
                self.mutation_rate
            )

            # Convert back to tensors
            population = [
                torch.tensor(member, dtype=population[0].dtype,
                           device=population[0].device)
                for member in evolved_population_np
            ]

        # Return best individual
        best_idx = jnp.argmax(fitness_scores)
        best_state = population[best_idx]
        return self._update_model_state(current_model, best_state)

    def apply_self_assembly_growth(self, current_model: nn.Module,
                                 assembly_rules: List[Tuple]) -> nn.Module:
        """
        Apply self-assembly growth strategy.

        Args:
            current_model: Current NCA model
            assembly_rules: List of (pattern, replacement) tuples

        Returns:
            Model with self-assembly growth applied
        """
        # Get current model state
        current_state = self._get_model_state(current_model)

        # Convert assembly rules to JAX arrays
        assembly_rules_jax = [
            (jnp.array(pattern), jnp.array(replacement))
            for pattern, replacement in assembly_rules
        ]

        # Apply JAX-optimized self-assembly
        state_np = current_state.detach().cpu().numpy()
        assembled_state_np = self._jax_self_assembly(
            jnp.array(state_np),
            assembly_rules_jax
        )

        assembled_state = torch.tensor(assembled_state_np, dtype=current_state.dtype,
                                     device=current_state.device)

        # Update model with assembled state
        updated_model = self._update_model_state(current_model, assembled_state)
        return updated_model

    def _get_model_state(self, model: nn.Module) -> torch.Tensor:
        """
        Extract model state for growth operations.

        Args:
            model: NCA model

        Returns:
            Model state tensor
        """
        # Extract relevant parameters for growth
        state_dict = model.state_dict()
        # Combine key parameters into state representation
        state_tensors = []
        for name, param in state_dict.items():
            if 'weight' in name and len(param.shape) >= 2:
                state_tensors.append(param.flatten())

        if state_tensors:
            return torch.cat(state_tensors)
        else:
            return torch.zeros(100)  # Default state

    def _compute_target_state(self, current_state: torch.Tensor,
                            target_performance: float) -> torch.Tensor:
        """
        Compute target state based on performance requirements.

        Args:
            current_state: Current model state
            target_performance: Target performance level

        Returns:
            Target state tensor
        """
        # Scale current state based on performance requirements
        performance_factor = target_performance / max(0.1, target_performance)
        target_state = current_state * performance_factor

        # Add some exploration for growth
        exploration = torch.randn_like(current_state) * 0.1
        target_state = target_state + exploration

        return target_state

    def _initialize_population(self, base_model: nn.Module) -> List[torch.Tensor]:
        """
        Initialize population for evolutionary growth.

        Args:
            base_model: Base model to create population from

        Returns:
            List of model state tensors
        """
        base_state = self._get_model_state(base_model)
        population = []

        for _ in range(self.population_size):
            # Create variation of base state
            variation = torch.randn_like(base_state) * 0.1
            individual_state = base_state + variation
            population.append(individual_state)

        return population

    def _population_member_to_model(self, state_tensor: torch.Tensor) -> nn.Module:
        """
        Convert population member back to model.

        Args:
            state_tensor: State tensor from population

        Returns:
            Model instance
        """
        # Create new model and update its state
        model = type(self.config.nca_model)(self.config)
        updated_model = self._update_model_state(model, state_tensor)
        return updated_model

    def _update_model_state(self, model: nn.Module, new_state: torch.Tensor) -> nn.Module:
        """
        Update model with new state.

        Args:
            model: Model to update
            new_state: New state tensor

        Returns:
            Updated model
        """
        # This is a simplified implementation
        # In practice, you'd need to carefully map state back to model parameters
        return model


class AdaptiveNCAWrapper:
    """
    Wrapper class that combines existing NCA with adaptive capabilities.

    Features:
    - Dynamic state vector sizing (start with 100x features, grow as needed)
    - History tracking (last 200 time steps)
    - Self-adjusted evolution steps (10-100 steps)
    - Real-time adaptation triggers
    """

    def __init__(self, base_nca_model: nn.Module, config):
        """
        Initialize AdaptiveNCAWrapper.

        Args:
            base_nca_model: Base NCA model to wrap
            config: Configuration object
        """
        self.base_model = base_nca_model
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Adaptive parameters
        self.initial_state_dim = 100
        self.max_state_dim = 512
        self.history_length = 200
        self.min_evolution_steps = 10
        self.max_evolution_steps = 100

        # State management
        self.current_state_dim = self.initial_state_dim
        self.evolution_steps = self.min_evolution_steps
        self.state_history = deque(maxlen=self.history_length)

        # Adaptation components
        self.grid_manager = AdaptiveGridManager(config)
        self.growth_strategies = GrowthStrategies(config)

        # Performance tracking
        self.adaptation_metrics = AdaptationMetrics()

        # JAX/XLA setup
        self._setup_jax_optimizations()

    def _setup_jax_optimizations(self):
        """Setup JAX optimizations for adaptive operations."""
        self._jax_adapt_state = jit(self._jax_adapt_state_impl)
        self._jax_update_history = jit(self._jax_update_history_impl)
        self._jax_compute_adaptation = jit(self._jax_compute_adaptation_impl)

    def _jax_adapt_state_impl(self, current_state: jnp.ndarray,
                            adaptation_signal: float) -> jnp.ndarray:
        """
        JAX implementation of state adaptation.

        Args:
            current_state: Current state
            adaptation_signal: Signal for adaptation

        Returns:
            Adapted state
        """
        # Apply adaptive transformation
        adaptation_factor = jnp.tanh(adaptation_signal)
        adapted_state = current_state * (1.0 + adaptation_factor * 0.1)
        return adapted_state

    def _jax_update_history_impl(self, history: jnp.ndarray,
                               new_state: jnp.ndarray) -> jnp.ndarray:
        """
        JAX implementation of history update.

        Args:
            history: Current history
            new_state: New state to add

        Returns:
            Updated history
        """
        # Shift history and add new state
        updated_history = jnp.roll(history, 1, axis=0)
        updated_history = updated_history.at[0].set(new_state)
        return updated_history

    def _jax_compute_adaptation_impl(self, performance_metrics: Dict[str, float],
                                   market_conditions: Dict[str, float]) -> float:
        """
        JAX implementation of adaptation computation.

        Args:
            performance_metrics: Current performance metrics
            market_conditions: Market condition data

        Returns:
            Adaptation signal
        """
        # Compute adaptation based on multiple factors
        perf_factor = performance_metrics.get('prediction_error', 0.0)
        volatility_factor = market_conditions.get('volatility', 0.0)
        complexity_factor = market_conditions.get('complexity', 0.0)

        adaptation_signal = (perf_factor * 0.4 +
                           volatility_factor * 0.3 +
                           complexity_factor * 0.3)
        return float(adaptation_signal)

    def forward(self, x: torch.Tensor, market_data: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through adaptive NCA.

        Args:
            x: Input tensor
            market_data: Market condition data

        Returns:
            Dictionary containing model outputs
        """
        # Analyze market conditions for adaptation
        if market_data:
            adaptation_signal = self._analyze_adaptation_needs(market_data)
            self._apply_adaptation(adaptation_signal)

        # Update state history
        current_state = self._get_current_state()
        self._update_state_history(current_state)

        # Apply adaptive evolution steps
        adaptive_steps = self._compute_adaptive_steps(market_data)

        # Forward through base model with adaptive parameters
        outputs = self.base_model(x, evolution_steps=adaptive_steps)

        # Add adaptation metadata
        outputs['adaptation_info'] = {
            'current_state_dim': self.current_state_dim,
            'evolution_steps': adaptive_steps,
            'history_length': len(self.state_history),
            'adaptation_signal': adaptation_signal if market_data else 0.0
        }

        return outputs

    def _analyze_adaptation_needs(self, market_data: Dict[str, float]) -> float:
        """
        Analyze market conditions to determine adaptation needs.

        Args:
            market_data: Market condition data

        Returns:
            Adaptation signal strength
        """
        # Use JAX for efficient computation
        performance_metrics = {
            'prediction_error': market_data.get('prediction_error', 0.0),
            'sharpe_ratio': market_data.get('sharpe_ratio', 1.0),
            'volatility': market_data.get('volatility', 0.0)
        }

        adaptation_signal = self._jax_compute_adaptation(performance_metrics, market_data)
        return adaptation_signal

    def _apply_adaptation(self, adaptation_signal: float):
        """
        Apply adaptation to the model.

        Args:
            adaptation_signal: Signal for adaptation
        """
        # Adapt state dimension
        if abs(adaptation_signal) > 0.1:
            self._adapt_state_dimension(adaptation_signal)

        # Adapt evolution steps
        self._adapt_evolution_steps(adaptation_signal)

    def _adapt_state_dimension(self, adaptation_signal: float):
        """
        Adapt the state vector dimension.

        Args:
            adaptation_signal: Signal for adaptation
        """
        # Determine new state dimension
        current_dim = self.current_state_dim
        adaptation_factor = 1.0 + adaptation_signal * 0.1

        new_dim = int(current_dim * adaptation_factor)
        new_dim = np.clip(new_dim, self.initial_state_dim, self.max_state_dim)

        if new_dim != current_dim:
            # Resize the base model
            self.base_model = self._resize_base_model(new_dim)
            self.current_state_dim = new_dim

            self.logger.info(f"Adapted state dimension from {current_dim} to {new_dim}")

    def _adapt_evolution_steps(self, adaptation_signal: float):
        """
        Adapt the number of evolution steps.

        Args:
            adaptation_signal: Signal for adaptation
        """
        # Adjust evolution steps based on adaptation signal
        if adaptation_signal > 0.2:
            # Increase evolution steps for complex conditions
            self.evolution_steps = min(self.evolution_steps + 5, self.max_evolution_steps)
        elif adaptation_signal < -0.1:
            # Decrease evolution steps for stable conditions
            self.evolution_steps = max(self.evolution_steps - 2, self.min_evolution_steps)

    def _get_current_state(self) -> torch.Tensor:
        """
        Get current model state.

        Returns:
            Current state tensor
        """
        # Extract state from base model
        return self._get_model_state(self.base_model)

    def _update_state_history(self, state: torch.Tensor):
        """
        Update state history.

        Args:
            state: New state to add to history
        """
        # Convert to JAX for efficient processing
        if hasattr(state, 'device') and state.device.type == 'cpu':
            state_np = state.detach().cpu().numpy()
            history_np = np.array([s.detach().cpu().numpy() for s in self.state_history])

            updated_history_np = self._jax_update_history(jnp.array(history_np), jnp.array(state_np))
            self.state_history.clear()

            for hist_state in updated_history_np:
                self.state_history.append(torch.tensor(hist_state, device=state.device, dtype=state.dtype))
        else:
            self.state_history.append(state)

    def _compute_adaptive_steps(self, market_data: Dict[str, float] = None) -> int:
        """
        Compute adaptive number of evolution steps.

        Args:
            market_data: Market condition data

        Returns:
            Number of evolution steps
        """
        if not market_data:
            return self.evolution_steps

        # Adjust based on market volatility
        volatility = market_data.get('volatility', 0.0)
        if volatility > 0.3:
            return min(self.evolution_steps + 10, self.max_evolution_steps)
        elif volatility < 0.1:
            return max(self.evolution_steps - 5, self.min_evolution_steps)

        return self.evolution_steps

    def _resize_base_model(self, new_dim: int) -> nn.Module:
        """
        Resize the base model to new state dimension.

        Args:
            new_dim: New state dimension

        Returns:
            Resized model
        """
        # This would implement the model resizing logic
        # For now, return the base model (simplified implementation)
        return self.base_model

    def _get_model_state(self, model: nn.Module) -> torch.Tensor:
        """
        Extract state from model.

        Args:
            model: Model to extract state from

        Returns:
            State tensor
        """
        # Simplified state extraction
        # In practice, this would extract meaningful state representation
        return torch.zeros(self.current_state_dim)

    def _jax_compute_adaptation(self, performance_metrics: Dict[str, float],
                              market_conditions: Dict[str, float]) -> float:
        """
        JAX implementation of adaptation computation.

        Args:
            performance_metrics: Performance metrics
            market_conditions: Market conditions

        Returns:
            Adaptation signal
        """
        # Simplified adaptation computation
        return 0.0


class MarketConditionAnalyzer:
    """
    Analyze market conditions to determine when to grow/shrink NCA.

    Analyzes:
    - Volatility-based triggers
    - Volume pattern analysis
    - Price momentum indicators
    - Correlation structure changes
    """

    def __init__(self, config):
        """
        Initialize MarketConditionAnalyzer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Analysis parameters
        self.volatility_window = 20
        self.volume_window = 50
        self.correlation_window = 30
        self.momentum_window = 14

        # Thresholds
        self.volatility_threshold = 0.25
        self.volume_threshold = 1.5
        self.correlation_threshold = 0.7
        self.momentum_threshold = 0.1

        # JAX optimizations
        self._setup_jax_analysis()

    def _setup_jax_analysis(self):
        """Setup JAX-optimized analysis functions."""
        self._jax_compute_volatility = jit(self._jax_compute_volatility_impl)
        self._jax_analyze_volume_patterns = jit(self._jax_analyze_volume_patterns_impl)
        self._jax_compute_correlation_structure = jit(self._jax_compute_correlation_structure_impl)
        self._jax_analyze_momentum = jit(self._jax_analyze_momentum_impl)

    def _jax_compute_volatility_impl(self, price_series: jnp.ndarray,
                                   window: int) -> float:
        """
        JAX implementation of volatility computation.

        Args:
            price_series: Price time series
            window: Rolling window size

        Returns:
            Volatility measure
        """
        returns = jnp.diff(jnp.log(price_series))
        rolling_vol = jnp.sqrt(jnp.convolve(returns**2, jnp.ones(window)/window, mode='valid'))
        return float(jnp.mean(rolling_vol))

    def _jax_analyze_volume_patterns_impl(self, volume_series: jnp.ndarray,
                                        price_series: jnp.ndarray,
                                        window: int) -> Dict[str, float]:
        """
        JAX implementation of volume pattern analysis.

        Args:
            volume_series: Volume time series
            price_series: Price time series
            window: Analysis window

        Returns:
            Volume pattern metrics
        """
        # Compute volume-price divergence
        volume_ma = jnp.convolve(volume_series, jnp.ones(window)/window, mode='valid')
        price_ma = jnp.convolve(price_series, jnp.ones(window)/window, mode='valid')

        volume_trend = jnp.polyfit(jnp.arange(len(volume_ma)), volume_ma, 1)[0]
        price_trend = jnp.polyfit(jnp.arange(len(price_ma)), price_ma, 1)[0]

        divergence = volume_trend * price_trend

        return {
            'volume_trend': float(volume_trend),
            'price_trend': float(price_trend),
            'divergence': float(divergence)
        }

    def _jax_compute_correlation_structure_impl(self, price_matrix: jnp.ndarray,
                                              window: int) -> float:
        """
        JAX implementation of correlation structure analysis.

        Args:
            price_matrix: Price matrix (assets x time)
            window: Analysis window

        Returns:
            Correlation structure complexity
        """
        # Compute rolling correlation matrix
        num_assets, num_periods = price_matrix.shape

        if num_periods < window:
            return 0.0

        # Compute correlation for each window
        correlations = []
        for i in range(num_periods - window + 1):
            window_data = price_matrix[:, i:i+window]
            corr_matrix = jnp.corrcoef(window_data)
            # Average absolute correlation
            avg_corr = jnp.mean(jnp.abs(corr_matrix))
            correlations.append(avg_corr)

        return float(jnp.mean(jnp.array(correlations)))

    def _jax_analyze_momentum_impl(self, price_series: jnp.ndarray,
                                 window: int) -> Dict[str, float]:
        """
        JAX implementation of momentum analysis.

        Args:
            price_series: Price time series
            window: Analysis window

        Returns:
            Momentum metrics
        """
        # Compute momentum indicators
        sma = jnp.convolve(price_series, jnp.ones(window)/window, mode='valid')
        momentum = price_series[window-1:] - sma

        # RSI-like momentum measure
        gains = jnp.maximum(momentum, 0)
        losses = jnp.abs(jnp.minimum(momentum, 0))

        avg_gain = jnp.mean(gains)
        avg_loss = jnp.mean(losses)

        rsi = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-10))))

        return {
            'momentum': float(jnp.mean(momentum)),
            'rsi': float(rsi),
            'momentum_strength': float(jnp.mean(jnp.abs(momentum)))
        }

    def analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze current market conditions.

        Args:
            market_data: Market data dictionary

        Returns:
            Analysis results
        """
        analysis_results = {}

        # Extract time series data
        price_data = market_data.get('prices', {})
        volume_data = market_data.get('volumes', {})
        asset_prices = market_data.get('asset_prices', {})

        # Volatility analysis
        if isinstance(price_data, dict) and 'values' in price_data:
            prices = jnp.array(price_data['values'])
            volatility = self._jax_compute_volatility(prices, self.volatility_window)
            analysis_results['volatility'] = volatility

        # Volume pattern analysis
        if isinstance(volume_data, dict) and 'values' in volume_data:
            volumes = jnp.array(volume_data['values'])
            if isinstance(price_data, dict) and 'values' in price_data:
                prices = jnp.array(price_data['values'])
                volume_analysis = self._jax_analyze_volume_patterns(volumes, prices, self.volume_window)
                analysis_results.update(volume_analysis)

        # Correlation structure analysis
        if isinstance(asset_prices, dict) and 'matrix' in asset_prices:
            price_matrix = jnp.array(asset_prices['matrix'])
            correlation_complexity = self._jax_compute_correlation_structure(
                price_matrix, self.correlation_window
            )
            analysis_results['correlation_complexity'] = correlation_complexity

        # Momentum analysis
        if isinstance(price_data, dict) and 'values' in price_data:
            prices = jnp.array(price_data['values'])
            momentum_analysis = self._jax_analyze_momentum(prices, self.momentum_window)
            analysis_results.update(momentum_analysis)

        return analysis_results

    def should_trigger_adaptation(self, analysis_results: Dict[str, float]) -> Tuple[bool, str]:
        """
        Determine if market conditions warrant adaptation.

        Args:
            analysis_results: Market analysis results

        Returns:
            Tuple of (should_trigger, reason)
        """
        # Check volatility trigger
        volatility = analysis_results.get('volatility', 0.0)
        if volatility > self.volatility_threshold:
            return True, AdaptationTrigger.VOLATILITY_SPIKE.value

        # Check volume divergence trigger
        divergence = analysis_results.get('divergence', 0.0)
        if abs(divergence) > self.volume_threshold:
            return True, "volume_divergence"

        # Check correlation structure trigger
        correlation_complexity = analysis_results.get('correlation_complexity', 0.0)
        if correlation_complexity > self.correlation_threshold:
            return True, AdaptationTrigger.MARKET_REGIME_CHANGE.value

        # Check momentum trigger
        momentum = analysis_results.get('momentum', 0.0)
        if abs(momentum) > self.momentum_threshold:
            return True, "momentum_shift"

        return False, "no_trigger"

    def get_adaptation_recommendations(self, analysis_results: Dict[str, float]) -> Dict[str, Any]:
        """
        Get specific adaptation recommendations.

        Args:
            analysis_results: Market analysis results

        Returns:
            Adaptation recommendations
        """
        recommendations = {
            'grow': False,
            'shrink': False,
            'reasons': [],
            'confidence': 0.0
        }

        # Volatility-based recommendations
        volatility = analysis_results.get('volatility', 0.0)
        if volatility > self.volatility_threshold * 1.5:
            recommendations['grow'] = True
            recommendations['reasons'].append('high_volatility')
        elif volatility < self.volatility_threshold * 0.5:
            recommendations['shrink'] = True
            recommendations['reasons'].append('low_volatility')

        # Volume-based recommendations
        volume_trend = analysis_results.get('volume_trend', 0.0)
        if volume_trend > 0.1:
            recommendations['grow'] = True
            recommendations['reasons'].append('increasing_volume')

        # Correlation-based recommendations
        correlation_complexity = analysis_results.get('correlation_complexity', 0.0)
        if correlation_complexity > self.correlation_threshold:
            recommendations['grow'] = True
            recommendations['reasons'].append('complex_correlation_structure')

        # Calculate confidence
        recommendations['confidence'] = len(recommendations['reasons']) * 0.25

        return recommendations


class PerformanceMetrics:
    """
    Track adaptation performance metrics.

    Monitors:
    - Growth efficiency metrics
    - Computational cost tracking
    - Prediction accuracy improvements
    - Memory usage optimization
    """

    def __init__(self, config):
        """
        Initialize PerformanceMetrics.

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Metrics storage
        self.growth_efficiency_history = deque(maxlen=1000)
        self.computational_cost_history = deque(maxlen=1000)
        self.prediction_improvement_history = deque(maxlen=1000)
        self.memory_usage_history = deque(maxlen=1000)

        # Current metrics
        self.current_metrics = AdaptationMetrics()

        # JAX optimizations
        self._setup_jax_metrics()

    def _setup_jax_metrics(self):
        """Setup JAX-optimized metrics computation."""
        self._jax_compute_efficiency = jit(self._jax_compute_efficiency_impl)
        self._jax_analyze_cost_benefit = jit(self._jax_analyze_cost_benefit_impl)
        self._jax_compute_memory_efficiency = jit(self._jax_compute_memory_efficiency_impl)

    def _jax_compute_efficiency_impl(self, performance_before: float,
                                   performance_after: float,
                                   adaptation_cost: float) -> float:
        """
        JAX implementation of efficiency computation.

        Args:
            performance_before: Performance before adaptation
            performance_after: Performance after adaptation
            adaptation_cost: Cost of adaptation

        Returns:
            Efficiency metric
        """
        improvement = performance_after - performance_before
        efficiency = improvement / (adaptation_cost + 1e-10)
        return float(efficiency)

    def _jax_analyze_cost_benefit_impl(self, costs: jnp.ndarray,
                                     benefits: jnp.ndarray) -> Dict[str, float]:
        """
        JAX implementation of cost-benefit analysis.

        Args:
            costs: Array of adaptation costs
            benefits: Array of adaptation benefits

        Returns:
            Cost-benefit analysis results
        """
        total_cost = jnp.sum(costs)
        total_benefit = jnp.sum(benefits)
        avg_cost = jnp.mean(costs)
        avg_benefit = jnp.mean(benefits)
        roi = total_benefit / (total_cost + 1e-10)

        return {
            'total_cost': float(total_cost),
            'total_benefit': float(total_benefit),
            'avg_cost': float(avg_cost),
            'avg_benefit': float(avg_benefit),
            'roi': float(roi)
        }

    def _jax_compute_memory_efficiency_impl(self, memory_before: float,
                                          memory_after: float,
                                          performance_improvement: float) -> float:
        """
        JAX implementation of memory efficiency computation.

        Args:
            memory_before: Memory usage before adaptation
            memory_after: Memory usage after adaptation
            performance_improvement: Performance improvement

        Returns:
            Memory efficiency metric
        """
        memory_change = memory_after - memory_before
        efficiency = performance_improvement / (abs(memory_change) + 1e-10)
        return float(efficiency)

    def update_metrics(self, adaptation_info: Dict[str, Any],
                      performance_data: Dict[str, float]):
        """
        Update performance metrics after adaptation.

        Args:
            adaptation_info: Information about the adaptation
            performance_data: Current performance data
        """
        # Extract adaptation details
        adapted = adaptation_info.get('adapted', False)
        if not adapted:
            return

        # Compute growth efficiency
        if 'old_size' in adaptation_info and 'new_size' in adaptation_info:
            old_size = adaptation_info['old_size']
            new_size = adaptation_info['new_size']

            # Get performance before and after
            performance_before = performance_data.get('before', 0.0)
            performance_after = performance_data.get('after', 0.0)

            # Compute adaptation cost (simplified)
            adaptation_cost = abs(new_size - old_size) * 0.01

            efficiency = self._jax_compute_efficiency(
                performance_before, performance_after, adaptation_cost
            )

            self.current_metrics.growth_efficiency = efficiency
            self.growth_efficiency_history.append(efficiency)

        # Update computational cost
        computational_cost = performance_data.get('computational_cost', 1.0)
        self.current_metrics.computational_cost = computational_cost
        self.computational_cost_history.append(computational_cost)

        # Update prediction improvement
        prediction_improvement = performance_data.get('prediction_improvement', 0.0)
        self.current_metrics.prediction_improvement = prediction_improvement
        self.prediction_improvement_history.append(prediction_improvement)

        # Update memory usage
        memory_usage = performance_data.get('memory_usage', 1.0)
        self.current_metrics.memory_usage = memory_usage
        self.memory_usage_history.append(memory_usage)

    def get_current_metrics(self) -> AdaptationMetrics:
        """
        Get current adaptation metrics.

        Returns:
            Current metrics
        """
        return self.current_metrics

    def get_aggregated_metrics(self) -> Dict[str, float]:
        """
        Get aggregated performance metrics.

        Returns:
            Aggregated metrics
        """
        if not self.growth_efficiency_history:
            return {"no_data": True}

        # Compute statistics
        efficiency_array = jnp.array(list(self.growth_efficiency_history))
        cost_array = jnp.array(list(self.computational_cost_history))
        improvement_array = jnp.array(list(self.prediction_improvement_history))

        # Cost-benefit analysis
        cost_benefit = self._jax_analyze_cost_benefit(cost_array, improvement_array)

        return {
            "avg_growth_efficiency": float(jnp.mean(efficiency_array)),
            "std_growth_efficiency": float(jnp.std(efficiency_array)),
            "avg_computational_cost": float(jnp.mean(cost_array)),
            "total_prediction_improvement": float(jnp.sum(improvement_array)),
            "adaptation_frequency": len(self.growth_efficiency_history) / max(1, len(self.computational_cost_history)),
            **cost_benefit
        }

    def get_efficiency_trends(self) -> Dict[str, List[float]]:
        """
        Get efficiency trends over time.

        Returns:
            Efficiency trends
        """
        return {
            "growth_efficiency": list(self.growth_efficiency_history),
            "computational_cost": list(self.computational_cost_history),
            "prediction_improvement": list(self.prediction_improvement_history),
            "memory_usage": list(self.memory_usage_history)
        }

    def reset_metrics(self):
        """Reset all performance metrics."""
        self.growth_efficiency_history.clear()
        self.computational_cost_history.clear()
        self.prediction_improvement_history.clear()
        self.memory_usage_history.clear()
        self.current_metrics = AdaptationMetrics()


# Factory functions for creating adaptive components
def create_adaptive_grid_manager(config) -> AdaptiveGridManager:
    """Create AdaptiveGridManager instance."""
    return AdaptiveGridManager(config)


def create_growth_strategies(config) -> GrowthStrategies:
    """Create GrowthStrategies instance."""
    return GrowthStrategies(config)


def create_adaptive_nca_wrapper(base_model: nn.Module, config) -> AdaptiveNCAWrapper:
    """Create AdaptiveNCAWrapper instance."""
    return AdaptiveNCAWrapper(base_model, config)


def create_market_condition_analyzer(config) -> MarketConditionAnalyzer:
    """Create MarketConditionAnalyzer instance."""
    return MarketConditionAnalyzer(config)


def create_performance_metrics(config) -> PerformanceMetrics:
    """Create PerformanceMetrics instance."""
    return PerformanceMetrics(config)


if __name__ == "__main__":
    # Example usage and testing
    from config import ConfigManager

    print("Adaptive NCA Module Demo")
    print("=" * 40)

    config = ConfigManager()

    # Create adaptive components
    grid_manager = create_adaptive_grid_manager(config)
    growth_strategies = create_growth_strategies(config)
    market_analyzer = create_market_condition_analyzer(config)
    performance_metrics = create_performance_metrics(config)

    print("Adaptive components created successfully!")
    print(f"Grid Manager: {type(grid_manager).__name__}")
    print(f"Growth Strategies: {type(growth_strategies).__name__}")
    print(f"Market Analyzer: {type(market_analyzer).__name__}")
    print(f"Performance Metrics: {type(performance_metrics).__name__}")

    # Test market analysis
    sample_market_data = {
        'prices': {'values': np.random.randn(100).tolist()},
        'volumes': {'values': np.random.randn(100).tolist()},
        'asset_prices': {'matrix': np.random.randn(5, 100).tolist()}
    }

    analysis = market_analyzer.analyze_market_conditions(sample_market_data)
    print(f"Market analysis completed: {len(analysis)} metrics computed")

    print("Adaptive NCA module demo completed successfully!")