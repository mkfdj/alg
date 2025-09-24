# JAX TPU v5e-8 Optimizations for NCA Trading Bot

This document outlines the JAX/TPU v5e-8 optimizations implemented for the NCA Trading Bot, providing code snippets and explanations for key components.

## Overview

The project has been optimized for TPU v5e-8 (8 cores) using JAX/Flax, with support for:
- Large batches (1024+ samples)
- Mixed precision (bfloat16/float16)
- Distributed sharding across 8 TPU cores
- Memory management for 40GB storage (35GB utilization)
- XLA compilation optimizations

## Key JAX Components

### 1. JAX NCA Model with Sharding

```python
from nca_model import create_jax_nca_model, create_jax_train_state
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

# Create JAX NCA model for TPU v5e-8
config = get_config()
model = create_jax_nca_model(config)

# Setup sharding mesh for 8 TPU cores
devices = jax.devices()  # Should return 8 TPU devices
mesh = Mesh(mesh_utils.create_device_mesh((1, 8)), axis_names=('data', 'model'))

# Create train state with sharding
rng = jax.random.PRNGKey(42)
state, mesh = create_jax_train_state(model, config.__dict__, rng)
```

### 2. Sharding Setup for TPU v5e-8

```python
# JAX sharding configuration for TPU v5e-8
def setup_tpu_sharding():
    """Setup sharding for TPU v5e-8 (8 cores)."""
    devices = jax.devices()

    if len(devices) == 8:  # TPU v5e-8
        # Create 1x8 mesh (1 chip row, 8 cores)
        mesh = Mesh(
            mesh_utils.create_device_mesh((1, 8)),
            axis_names=('data', 'model')
        )

        # Sharding specifications
        data_sharding = P('data', None)      # Shard data across cores
        model_sharding = P(None, 'model')    # Replicate model across cores
        batch_sharding = P('data')           # Shard batch dimension

        return mesh, data_sharding, model_sharding, batch_sharding
    else:
        # Fallback for other configurations
        return None, None, None, None

# Usage
mesh, data_shard, model_shard, batch_shard = setup_tpu_sharding()
```

### 3. Large Batch Training with Gradient Accumulation

```python
# Large batch training setup (2048 batch size)
config.training.batch_size = 2048
config.training.micro_batch_size = 256
config.training.gradient_accumulation_steps = 8

@jax.jit
def train_step_with_accumulation(state, batch, config):
    """Training step with gradient accumulation for large batches."""

    def loss_fn(params, micro_batch):
        # Forward pass on micro-batch
        outputs = state.apply_fn({'params': params}, micro_batch['x'])
        loss = compute_loss(outputs, micro_batch)
        return loss

    # Split large batch into micro-batches
    micro_batches = split_batch_into_micro_batches(batch, config.micro_batch_size)

    # Accumulate gradients
    grads_accum = None
    for micro_batch in micro_batches:
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(state.params, micro_batch)

        if grads_accum is None:
            grads_accum = grads
        else:
            grads_accum = jax.tree_util.tree_map(
                lambda x, y: x + y, grads_accum, grads
            )

    # Average gradients
    grads_accum = jax.tree_util.tree_map(
        lambda x: x / len(micro_batches), grads_accum
    )

    # Apply gradients
    state = state.apply_gradients(grads=grads_accum)
    return state
```

### 4. Mixed Precision (bfloat16) Setup

```python
# Mixed precision configuration for TPU v5e-8
config.training.use_mixed_precision = True
config.training.precision_dtype = "bf16"  # bfloat16 preferred for TPU
config.training.compute_dtype = "f32"    # f32 for stable gradients

# JAX mixed precision setup
from jax import config as jax_config
jax_config.update("jax_default_matmul_precision", "bf16")

# Model with mixed precision
class JAXNCATradingModel(nn.Module):
    config: dict
    dtype: jnp.dtype = jnp.bfloat16  # Default to bfloat16

    def setup(self):
        # All layers use bfloat16
        self.input_conv = nn.Conv(..., dtype=self.dtype)
        self.nca_layers = [
            JAXNCACell(..., dtype=self.dtype)
            for _ in range(self.config['nca']['num_layers'])
        ]

    @nn.compact
    def __call__(self, x):
        # Mixed precision forward pass
        with jax.default_matmul_precision("bf16"):
            # Model computations in bfloat16
            x = self.input_conv(x)
            for layer in self.nca_layers:
                x = layer(x)
            return x
```

### 5. JAX PPO Training with Sharding

```python
from trainer import JAXPPOTrainer

# Initialize JAX PPO trainer for TPU
observation_dim = 60  # Sequence length * features
action_dim = 3        # Buy, Hold, Sell
jax_ppo = JAXPPOTrainer(observation_dim, action_dim, config)

# Training loop with large batches
for epoch in range(config.training.num_epochs):
    # Collect trajectories with large batch size
    trajectories = collect_trajectories(env, batch_size=config.training.batch_size)

    # Update policy with JAX
    metrics = jax_ppo.update_policy()

    print(f"Epoch {epoch}: Loss = {metrics['total_loss']:.4f}")
```

### 6. Memory Management for 40GB Storage

```python
# Memory management configuration
config.tpu.max_memory_gb = 35.0  # Fill to 35GB (87.5% of 40GB)
config.tpu.memory_fraction = 0.875
config.tpu.enable_memory_optimization = True

# JAX memory optimization techniques
def optimize_memory_usage():
    """Memory optimization for TPU v5e-8."""

    # 1. Use rematerialization (recomputation) to save memory
    from jax.experimental import maps
    with maps.Mesh(mesh.devices, axis_names=('x', 'y')):
        with nn.remat_policy("dots_with_no_batch_dims"):
            # Model forward pass with rematerialization
            outputs = model.apply(params, batch)

    # 2. Gradient checkpointing
    @jax.remat
    def checkpointed_layer(x):
        return nn.Dense(256)(nn.relu(nn.Dense(128)(x)))

    # 3. Use bfloat16 to reduce memory footprint
    # (Already configured above)

    # 4. Memory-efficient data loading
    def create_memory_efficient_dataloader(batch_size, max_memory_gb):
        """Create dataloader that respects memory limits."""
        available_memory = max_memory_gb * (1024**3)  # Convert to bytes
        sample_memory = estimate_sample_memory()  # Estimate per sample

        # Calculate safe batch size
        safe_batch_size = min(batch_size, available_memory // sample_memory // 4)
        return safe_batch_size
```

### 7. XLA Compilation Optimizations

```python
# XLA compilation setup for TPU v5e-8
config.tpu.xla_compile = True
config.tpu.xla_persistent_cache = True
config.tpu.xla_memory_fraction = 0.8

# JAX compilation with XLA optimizations
@jax.jit
def compiled_train_step(state, batch):
    """XLA-compiled training step."""
    def loss_fn(params):
        outputs = state.apply_fn({'params': params}, batch['x'])
        loss = compute_ppo_loss(outputs, batch)
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state

# Persistent compilation cache
jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')
jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
```

### 8. Distributed Training Setup

```python
# Distributed training across TPU v5e-8 cores
def setup_distributed_training():
    """Setup distributed training for TPU pod."""

    # Initialize JAX distributed
    jax.distributed.initialize(
        coordinator_address="localhost:12345",
        num_processes=1,  # Single host with 8 cores
        process_id=0
    )

    # Create mesh for 8 TPU cores
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('data', 'model'))

    # Data parallelism across cores
    data_parallel_sharding = P('data', None)

    return mesh, data_parallel_sharding

# Usage in training
mesh, sharding = setup_distributed_training()

with mesh:
    # All operations within this context use the mesh
    state = jax.device_put(state, sharding)
    batch = jax.device_put(batch, sharding)

    # Training step
    state = compiled_train_step(state, batch)
```

### 9. Performance Monitoring

```python
# TPU performance monitoring
def monitor_tpu_performance():
    """Monitor TPU v5e-8 performance metrics."""

    # Memory usage
    memory_stats = jax.device_get(jax.device_count())

    # Compilation time
    with jax.profiler.trace('/tmp/tensorboard'):
        # Training code here
        pass

    # XLA metrics
    if config.tpu.xla_metrics_enabled:
        # Enable XLA profiling
        jax.config.update('jax_log_compiles', True)

    return memory_stats

# Integration with training loop
for step in range(num_steps):
    # Training step
    state, metrics = train_step(state, batch)

    # Periodic monitoring
    if step % 100 == 0:
        perf_stats = monitor_tpu_performance()
        log_performance_metrics(perf_stats, step)
```

## Configuration Summary

```python
# Complete TPU v5e-8 configuration
config = {
    # TPU Hardware
    'tpu_cores': 8,
    'tpu_topology': '2x4',

    # JAX/TPU Settings
    'use_jax': True,
    'jax_backend': 'tpu',

    # Mixed Precision
    'mixed_precision': True,
    'precision_dtype': 'bf16',
    'compute_dtype': 'f32',

    # Large Batch Training
    'batch_size': 2048,
    'micro_batch_size': 256,
    'gradient_accumulation_steps': 8,

    # Memory Management
    'max_memory_gb': 35.0,
    'memory_fraction': 0.875,

    # Sharding
    'enable_sharding': True,
    'sharding_strategy': '2d',
    'mesh_shape': (1, 8),
    'axis_names': ('data', 'model'),

    # XLA Optimizations
    'xla_compile': True,
    'xla_persistent_cache': True,
    'enable_fusion': True,
    'enable_remat': True,

    # Performance
    'tpu_metrics_enabled': True,
    'xla_metrics_enabled': True,
    'profiling_enabled': False
}
```

## Usage Examples

### Basic Training Setup

```python
from config import get_config
from nca_model import create_jax_nca_model, create_jax_train_state
from trainer import JAXPPOTrainer

# Load configuration
config = get_config()

# Create JAX model and trainer
model = create_jax_nca_model(config)
jax_ppo = JAXPPOTrainer(observation_dim=60, action_dim=3, config=config)

# Setup sharding
rng = jax.random.PRNGKey(42)
state, mesh = create_jax_train_state(model, config.__dict__, rng)

print("JAX TPU v5e-8 setup complete!")
print(f"Using {len(jax.devices())} TPU cores")
```

### Large Batch Inference

```python
# Large batch inference optimized for TPU
@jax.jit
def large_batch_inference(params, batch):
    """Process large batches efficiently on TPU."""
    # Split large batch across cores
    batch_size = batch.shape[0]
    core_batch_size = batch_size // 8  # 8 TPU cores

    # Process in parallel across cores
    def single_core_inference(core_batch):
        return model.apply({'params': params}, core_batch)

    # Parallel inference
    core_batches = jnp.split(batch, 8, axis=0)
    results = jax.pmap(single_core_inference)(core_batches)

    # Concatenate results
    return jnp.concatenate(results, axis=0)

# Usage
large_batch = jnp.ones((2048, 60, 20))  # Large batch
predictions = large_batch_inference(state.params, large_batch)
```

This implementation provides significant performance improvements over the previous PyTorch/XLA setup, leveraging JAX's optimized TPU compilation and the v5e-8's specialized architecture for large-scale reinforcement learning training.