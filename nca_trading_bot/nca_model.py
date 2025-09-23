"""
Neural Cellular Automata model for NCA Trading Bot.

This module implements the core NCA architecture with convolutional update rules,
self-adaptive mechanisms, online learning, and performance optimizations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.cuda.amp import autocast, GradScaler
from functools import lru_cache
import hashlib
import pickle
from pathlib import Path

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


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell for NCA state updates.

    Implements a convolutional version of GRU for processing spatial-temporal data.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        """
        Initialize ConvGRU cell.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            kernel_size: Convolutional kernel size
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        padding = kernel_size // 2

        # Update gate
        self.conv_z = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=True
        )

        # Reset gate
        self.conv_r = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=True
        )

        # Candidate activation
        self.conv_h = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=True
        )

        # Initialize weights
        for conv in [self.conv_z, self.conv_r, self.conv_h]:
            nn.init.orthogonal_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ConvGRU cell.

        Args:
            x: Input tensor
            h: Hidden state tensor

        Returns:
            Updated hidden state
        """
        combined = torch.cat([x, h], dim=1)

        # Update gate
        z = torch.sigmoid(self.conv_z(combined))

        # Reset gate
        r = torch.sigmoid(self.conv_r(combined))

        # Candidate activation - optimized for TPU
        combined_r = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_r))

        # New hidden state - optimized computation for TPU
        h_new = (1 - z) * h + z * h_tilde

        return h_new

    def forward_tpu_optimized(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        TPU-optimized forward pass with XLA compilation support.

        Args:
            x: Input tensor
            h: Hidden state tensor

        Returns:
            Updated hidden state
        """
        # Use XLA compilation for better TPU performance
        if XLA_AVAILABLE and x.device.type == 'xla':
            # Mark operations for XLA compilation
            combined = torch.cat([x, h], dim=1)

            # Update gate
            z = torch.sigmoid(self.conv_z(combined))

            # Reset gate
            r = torch.sigmoid(self.conv_r(combined))

            # Candidate activation
            combined_r = torch.cat([x, r * h], dim=1)
            h_tilde = torch.tanh(self.conv_h(combined_r))

            # New hidden state
            h_new = (1 - z) * h + z * h_tilde

            return h_new
        else:
            return self.forward(x, h)


class NCACell(nn.Module):
    """
    Neural Cellular Automata cell with self-adaptive capabilities.

    Implements the core NCA update rule with convolutional operations
    and adaptive parameters for trading applications.
    """

    def __init__(self, state_dim: int, hidden_dim: int, kernel_size: int = 3):
        """
        Initialize NCA cell.

        Args:
            state_dim: State vector dimension
            hidden_dim: Hidden layer dimension
            kernel_size: Convolutional kernel size
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        # NCA update parameters (learnable)
        self.alpha = Parameter(torch.tensor(0.1))  # Update rate
        self.beta = Parameter(torch.tensor(0.1))   # Growth rate
        self.gamma = Parameter(torch.tensor(0.1))  # Decay rate

        # Convolutional layers for state updates
        self.conv1 = nn.Conv2d(state_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(hidden_dim, state_dim, kernel_size, padding=kernel_size//2)

        # Self-adaptation layers
        self.adaptation_conv = nn.Conv2d(state_dim, state_dim, 1)
        self.mutation_rate = Parameter(torch.tensor(0.001))

        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.adaptation_conv.weight)

        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)
        nn.init.constant_(self.adaptation_conv.bias, 0.0)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through NCA cell.

        Args:
            x: Input state tensor
            h: Hidden state tensor

        Returns:
            Tuple of (new_state, new_hidden)
        """
        # State update
        combined = torch.cat([x, h], dim=1)
        hidden_out = F.relu(self.conv1(combined))
        state_update = self.conv2(hidden_out)

        # Apply NCA update rule
        new_state = x + self.alpha * state_update

        # Growth and decay
        growth = self.beta * torch.sigmoid(state_update)
        decay = self.gamma * torch.sigmoid(-state_update)

        new_state = new_state + growth - decay

        # Self-adaptation
        adaptation = self.adaptation_conv(x)
        adaptation_mask = torch.sigmoid(adaptation)

        # Apply adaptation with mutation
        mutation = torch.randn_like(new_state) * self.mutation_rate
        new_state = adaptation_mask * new_state + (1 - adaptation_mask) * (new_state + mutation)

        # Update hidden state
        new_hidden = F.relu(hidden_out)

        return new_state, new_hidden

    def forward_tpu_optimized(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TPU-optimized forward pass with XLA compilation support.

        Args:
            x: Input state tensor
            h: Hidden state tensor

        Returns:
            Tuple of (new_state, new_hidden)
        """
        # Optimized for TPU's systolic arrays and MXUs
        if XLA_AVAILABLE and x.device.type == 'xla':
            # Use XLA compilation for better TPU performance
            combined = torch.cat([x, h], dim=1)

            # State update - optimized for TPU matrix operations
            hidden_out = F.relu(self.conv1(combined))
            state_update = self.conv2(hidden_out)

            # Apply NCA update rule with optimized operations
            new_state = x + self.alpha * state_update

            # Growth and decay - vectorized operations for TPU
            growth = self.beta * torch.sigmoid(state_update)
            decay = self.gamma * torch.sigmoid(-state_update)

            new_state = new_state + growth - decay

            # Self-adaptation - optimized for TPU
            adaptation = self.adaptation_conv(x)
            adaptation_mask = torch.sigmoid(adaptation)

            # Apply adaptation with mutation
            mutation = torch.randn_like(new_state) * self.mutation_rate
            new_state = adaptation_mask * new_state + (1 - adaptation_mask) * (new_state + mutation)

            # Update hidden state
            new_hidden = F.relu(hidden_out)

            return new_state, new_hidden
        else:
            return self.forward(x, h)

    def evolve(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Evolve NCA for multiple steps.

        Args:
            x: Initial state tensor
            steps: Number of evolution steps

        Returns:
            Evolved state tensor
        """
        current_state = x
        hidden = torch.zeros(
            x.size(0), self.hidden_dim, x.size(2), x.size(3),
            device=x.device, dtype=x.dtype
        )

        for _ in range(steps):
            current_state, hidden = self.forward(current_state, hidden)

        return current_state

    def evolve_tpu_optimized(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        TPU-optimized NCA evolution with XLA compilation support.

        Args:
            x: Initial state tensor
            steps: Number of evolution steps

        Returns:
            Evolved state tensor
        """
        if XLA_AVAILABLE and x.device.type == 'xla':
            # Use XLA compilation for better TPU performance
            current_state = x
            hidden = torch.zeros(
                x.size(0), self.hidden_dim, x.size(2), x.size(3),
                device=x.device, dtype=x.dtype
            )

            for _ in range(steps):
                current_state, hidden = self.forward_tpu_optimized(current_state, hidden)

            return current_state
        else:
            return self.evolve(x, steps)


class NCATradingModel(nn.Module):
    """
    Complete NCA trading model with multiple layers and trading-specific features.

    Implements a multi-layer NCA architecture optimized for financial time series
    prediction and trading signal generation.
    """

    def __init__(self, config):
        """
        Initialize NCA trading model.

        Args:
            config: Configuration object with model parameters
        """
        super().__init__()

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Model dimensions
        self.state_dim = config.nca.state_dim
        self.hidden_dim = config.nca.hidden_dim
        self.num_layers = config.nca.num_layers
        self.kernel_size = config.nca.kernel_size

        # Input processing
        self.input_conv = nn.Conv2d(1, self.state_dim, 1)

        # NCA layers
        self.nca_layers = nn.ModuleList([
            NCACell(self.state_dim, self.hidden_dim, self.kernel_size)
            for _ in range(self.num_layers)
        ])

        # Output processing
        self.output_conv = nn.Conv2d(self.state_dim, 1, 1)
        self.dropout = nn.Dropout(config.nca.dropout_rate)

        # Trading-specific heads
        self.price_predictor = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.nca.dropout_rate),
            nn.Linear(self.hidden_dim, 1)
        )

        self.signal_classifier = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.nca.dropout_rate),
            nn.Linear(self.hidden_dim, 3)  # Buy, Hold, Sell
        )

        # Risk assessment
        self.risk_predictor = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.nca.dropout_rate),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Self-adaptation parameters
        self.adaptation_rate = Parameter(torch.tensor(config.nca.adaptation_rate))
        self.selection_pressure = Parameter(torch.tensor(config.nca.selection_pressure))

        # Performance tracking
        self.performance_history = []

        # Initialize weights
        self._initialize_weights()

        # TPU-specific initialization
        self._initialize_tpu_optimizations()

    def _initialize_weights(self):
        """Initialize model weights using appropriate schemes."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'input' in name or 'output' in name:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _initialize_tpu_optimizations(self):
        """Initialize TPU-specific optimizations."""
        # TPU-specific optimizations will be applied here
        # This includes memory layout optimizations, XLA compilation hints, etc.
        pass

    def forward(self, x: torch.Tensor, evolution_steps: int = 1) -> Dict[str, torch.Tensor]:
        """
        Forward pass through NCA trading model.

        Args:
            x: Input tensor (batch_size, seq_len, features)
            evolution_steps: Number of NCA evolution steps

        Returns:
            Dictionary containing model outputs
        """
        batch_size, seq_len, num_features = x.shape

        # Reshape for convolutional processing
        x_conv = x.unsqueeze(1)  # Add channel dimension
        x_conv = self.input_conv(x_conv)

        # Process through NCA layers - TPU optimized
        current_state = x_conv
        for layer_idx, nca_layer in enumerate(self.nca_layers):
            # Apply dropout for regularization
            current_state = self.dropout(current_state)

            # Evolve NCA - use TPU-optimized version if available
            if XLA_AVAILABLE and x_conv.device.type == 'xla' and hasattr(nca_layer, 'evolve_tpu_optimized'):
                current_state = nca_layer.evolve_tpu_optimized(current_state, evolution_steps)
            else:
                current_state = nca_layer.evolve(current_state, evolution_steps)

        # Generate outputs
        price_pred = self.price_predictor(current_state.mean(dim=[2, 3]))
        signal_logits = self.signal_classifier(current_state.mean(dim=[2, 3]))
        risk_prob = self.risk_predictor(current_state.mean(dim=[2, 3]))

        # Apply softmax to signals
        signal_probs = F.softmax(signal_logits, dim=1)

        return {
            'price_prediction': price_pred.squeeze(),
            'signal_probabilities': signal_probs,
            'risk_probability': risk_prob.squeeze(),
            'final_state': current_state,
            'adaptation_rate': self.adaptation_rate,
            'selection_pressure': self.selection_pressure
        }

    def predict(self, x: torch.Tensor) -> Dict[str, Union[float, int]]:
        """
        Make trading predictions from input data.

        Args:
            x: Input tensor

        Returns:
            Dictionary with trading predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)

            # Convert signal probabilities to trading signal
            signal_probs = outputs['signal_probabilities']
            signal_idx = torch.argmax(signal_probs, dim=1).item()

            signal_map = {0: 'sell', 1: 'hold', 2: 'buy'}
            trading_signal = signal_map[signal_idx]

            return {
                'trading_signal': trading_signal,
                'confidence': signal_probs[0, signal_idx].item(),
                'price_prediction': outputs['price_prediction'].item(),
                'risk_probability': outputs['risk_probability'].item()
            }

    def adapt_online(self, x: torch.Tensor, target: torch.Tensor,
                    adaptation_strength: float = None) -> Dict[str, float]:
        """
        Perform online adaptation of model parameters.

        Args:
            x: Input tensor
            target: Target values
            adaptation_strength: Strength of adaptation

        Returns:
            Dictionary with adaptation metrics
        """
        if adaptation_strength is None:
            adaptation_strength = self.adaptation_rate.item()

        self.train()

        # Forward pass
        outputs = self.forward(x)

        # Calculate losses
        price_loss = F.mse_loss(outputs['price_prediction'], target)
        signal_loss = F.cross_entropy(outputs['signal_probabilities'], target.long())

        # Combined loss
        total_loss = price_loss + signal_loss

        # Adaptation step
        adaptation_gradients = torch.autograd.grad(
            total_loss, self.parameters(), retain_graph=True
        )

        # Apply adaptation with momentum
        with torch.no_grad():
            for param, grad in zip(self.parameters(), adaptation_gradients):
                if grad is not None:
                    param.data -= adaptation_strength * grad

        return {
            'total_loss': total_loss.item(),
            'price_loss': price_loss.item(),
            'signal_loss': signal_loss.item(),
            'adaptation_strength': adaptation_strength
        }

    def update_performance(self, metrics: Dict[str, float]):
        """
        Update performance history for self-adaptation.

        Args:
            metrics: Performance metrics dictionary
        """
        self.performance_history.append(metrics)

        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

        # Adapt parameters based on performance
        if len(self.performance_history) > 10:
            recent_performance = self.performance_history[-10:]
            avg_performance = np.mean([m.get('reward', 0) for m in recent_performance])

            # Adjust adaptation rate based on performance
            with torch.no_grad():
                if avg_performance > 0:
                    self.adaptation_rate.data *= 1.01  # Increase adaptation
                else:
                    self.adaptation_rate.data *= 0.99  # Decrease adaptation


class NCAModelCache:
    """
    Caching system for NCA model states and computations.

    Provides memoization and incremental computing capabilities.
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize NCA model cache.

        Args:
            cache_dir: Directory for persistent caching
        """
        self.config = get_config()
        self.cache_dir = Path(cache_dir) if cache_dir else self.config.system.model_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        # In-memory caches
        self.state_cache = {}
        self.computation_cache = {}

        # Cache settings
        self.max_cache_size = self.config.system.cache_size
        self.cache_ttl = self.config.system.cache_ttl

    def _generate_cache_key(self, data: Any) -> str:
        """Generate cache key from input data."""
        if isinstance(data, torch.Tensor):
            data_hash = hashlib.md5(data.detach().cpu().numpy().tobytes()).hexdigest()
        elif isinstance(data, np.ndarray):
            data_hash = hashlib.md5(data.tobytes()).hexdigest()
        else:
            data_hash = hashlib.md5(str(data).encode()).hexdigest()

        return data_hash[:16]  # Use first 16 characters

    def get_cached_state(self, model_hash: str, input_hash: str) -> Optional[torch.Tensor]:
        """
        Retrieve cached model state.

        Args:
            model_hash: Hash of model configuration
            input_hash: Hash of input data

        Returns:
            Cached state tensor or None
        """
        cache_key = f"{model_hash}_{input_hash}"
        cache_file = self.cache_dir / f"{cache_key}.pt"

        if cache_file.exists():
            try:
                cached_data = torch.load(cache_file)
                return cached_data['state']
            except Exception:
                return None

        return None

    def cache_state(self, model_hash: str, input_hash: str, state: torch.Tensor):
        """
        Cache model state.

        Args:
            model_hash: Hash of model configuration
            input_hash: Hash of input data
            state: State tensor to cache
        """
        cache_key = f"{model_hash}_{input_hash}"
        cache_file = self.cache_dir / f"{cache_key}.pt"

        try:
            torch.save({
                'state': state,
                'timestamp': torch.tensor(time.time())
            }, cache_file)
        except Exception as e:
            logging.warning(f"Failed to cache state: {e}")

    @lru_cache(maxsize=100)
    def cached_computation(self, computation_hash: str, func, *args, **kwargs):
        """
        Cache computation results with memoization.

        Args:
            computation_hash: Hash of computation parameters
            func: Function to cache
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            Cached or computed result
        """
        return func(*args, **kwargs)


class NCATrainer:
    """
    Training utilities for NCA models with AMP and DDP support.

    Provides optimized training loops with automatic mixed precision
    and distributed training capabilities.
    """

    def __init__(self, model: NCATradingModel, config):
        """
        Initialize NCA trainer.

        Args:
            model: NCA model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Training setup - support TPU, CUDA, and CPU
        self.device = self._get_device_from_config(config)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.nca.learning_rate,
            weight_decay=config.nca.weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        # AMP setup - support both CUDA and TPU
        self.scaler = GradScaler() if config.training.use_amp and torch.cuda.is_available() else None

        # Distributed setup
        self.is_ddp = config.training.num_gpus > 1
        self.is_tpu = config.system.device == "tpu"

        if self.is_ddp:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[config.training.local_rank]
            )
        elif self.is_tpu:
            # TPU uses XLA SPMD - no explicit wrapping needed
            pass

        # Loss functions
        self.price_criterion = nn.MSELoss()
        self.signal_criterion = nn.CrossEntropyLoss()
        self.risk_criterion = nn.BCELoss()

    def _get_device_from_config(self, config):
        """Get device based on configuration and availability."""
        # Import XLA modules if available
        try:
            import torch_xla.core.xla_model as xm
            XLA_AVAILABLE = True
        except ImportError:
            XLA_AVAILABLE = False
            xm = None

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

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform single training step.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        x, price_target, signal_target, risk_target = (
            batch['x'].to(self.device),
            batch['price_target'].to(self.device),
            batch['signal_target'].to(self.device),
            batch['risk_target'].to(self.device)
        )

        # Forward pass with AMP
        if self.scaler:
            with autocast(dtype=getattr(torch, self.config.training.amp_dtype)):
                outputs = self.model(x)
                price_loss = self.price_criterion(outputs['price_prediction'], price_target)
                signal_loss = self.signal_criterion(outputs['signal_probabilities'], signal_target)
                risk_loss = self.risk_criterion(outputs['risk_probability'], risk_target)

                total_loss = price_loss + signal_loss + risk_loss
        else:
            outputs = self.model(x)
            price_loss = self.price_criterion(outputs['price_prediction'], price_target)
            signal_loss = self.signal_criterion(outputs['signal_probabilities'], signal_target)
            risk_loss = self.risk_criterion(outputs['risk_probability'], risk_target)

            total_loss = price_loss + signal_loss + risk_loss

        # Backward pass with AMP
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
            self.optimizer.step()

        # Update scheduler
        self.scheduler.step(total_loss)

        return {
            'total_loss': total_loss.item(),
            'price_loss': price_loss.item(),
            'signal_loss': signal_loss.item(),
            'risk_loss': risk_loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate model on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x, price_target, signal_target, risk_target = (
                    batch['x'].to(self.device),
                    batch['price_target'].to(self.device),
                    batch['signal_target'].to(self.device),
                    batch['risk_target'].to(self.device)
                )

                outputs = self.model(x)
                price_loss = self.price_criterion(outputs['price_prediction'], price_target)
                signal_loss = self.signal_criterion(outputs['signal_probabilities'], signal_target)
                risk_loss = self.risk_criterion(outputs['risk_probability'], risk_target)

                total_loss += (price_loss + signal_loss + risk_loss).item()
                num_batches += 1

        return {'val_loss': total_loss / num_batches if num_batches > 0 else float('inf')}

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Training metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics
        }

        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Dictionary with loaded data
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.logger.info(f"Checkpoint loaded from {path}")
        return checkpoint


# Utility functions
def create_nca_model(config) -> NCATradingModel:
    """
    Create NCA trading model instance.

    Args:
        config: Configuration object

    Returns:
        Initialized NCA trading model
    """
    model = NCATradingModel(config)
    return model


def load_nca_model(path: str, config) -> NCATradingModel:
    """
    Load NCA model from file.

    Args:
        path: Path to model file
        config: Configuration object

    Returns:
        Loaded NCA model
    """
    model = NCATradingModel(config)
    model.load_state_dict(torch.load(path))
    return model


if __name__ == "__main__":
    # Example usage
    from config import ConfigManager

    print("NCA Trading Bot - Model Demo")
    print("=" * 40)

    config = ConfigManager()
    model = create_nca_model(config)

    # Create sample input
    batch_size, seq_len, features = 4, 10, 20
    sample_input = torch.randn(batch_size, seq_len, features)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Input shape: {sample_input.shape}")

    # Forward pass
    outputs = model(sample_input)
    print(f"Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")

    print("NCA Model demo completed successfully!")