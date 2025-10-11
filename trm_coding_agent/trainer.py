"""
Trainer for TRM Coding Agent with PPO and distributed TPU support.

This module implements the training loop with PPO fine-tuning,
distributed computing across TPU chips, and comprehensive logging.
"""

import os
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, pmap, jit
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax.distributed
import optax
import flax
from flax.training.train_state import TrainState
import numpy as np

from .config import TRMConfig, Config
from .trm_model import TRMModel, create_trm_model
from .data_handler import DataHandler, DatasetSample
from .coder import CoderEnvironment, ActionType
from .utils import setup_tpu, save_checkpoint, load_checkpoint, count_parameters

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class TrainingMetrics:
    """Training metrics for logging."""
    step: int
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    grad_norm: float
    recursion_depth_mean: float
    binary_decision_ratio: float
    code_success_rate: float
    test_pass_rate: float
    episode_reward_mean: float
    wall_time: float


class PPOTrainer:
    """PPO Trainer for TRM Coding Agent."""

    def __init__(
        self,
        model: TRMModel,
        config: Config,
        mesh: Optional[Mesh] = None
    ):
        self.model = model
        self.config = config
        self.trm_config = config.trm
        self.env_config = config.environment

        # Setup distributed computing
        if mesh is None:
            self.mesh, self.device_info = setup_tpu()
        else:
            self.mesh = mesh
            self.device_info = {}

        # Initialize components
        self.data_handler = DataHandler(config)
        self.coder_env = CoderEnvironment(config)

        # Training state
        self.train_state = None
        self.optimizer_state = None

        # Metrics tracking
        self.metrics_history = []
        self.best_val_loss = float('inf')
        self.steps_since_improvement = 0

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging and monitoring."""
        import logging
        import datetime

        # Create logger
        self.logger = logging.getLogger(f"TRMTrainer_{jax.process_index()}")
        self.logger.setLevel(logging.INFO)

        # File handler
        log_file = os.path.join(
            self.config.logging.log_file.replace('.log', f'_{jax.process_index()}.log')
        )
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Setup wandb if enabled
        if self.config.logging.use_wandb and jax.process_index() == 0:
            try:
                import wandb
                wandb.init(
                    project=self.config.logging.wandb_project or "trm_coding_agent",
                    config=self.config.to_dict(),
                    name=f"trm_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self.wandb = wandb
                self.logger.info("Wandb logging initialized")
            except ImportError:
                self.logger.warning("Wandb not available, skipping wandb logging")
                self.wandb = None
        else:
            self.wandb = None

        # Setup tensorboard if enabled
        if self.config.logging.use_tensorboard and jax.process_index() == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(self.config.logging.tensorboard_log_dir)
                self.logger.info("Tensorboard logging initialized")
            except ImportError:
                self.logger.warning("Tensorboard not available, skipping tensorboard logging")
                self.tb_writer = None
        else:
            self.tb_writer = None

    def initialize_training(self, resume_from: Optional[str] = None):
        """
        Initialize training state and optimizer.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        self.logger.info("Initializing training...")

        # Create initial train state
        self.train_state = self.model.create_train_state(
            learning_rate=self.trm_config.learning_rate,
            weight_decay=self.trm_config.weight_decay
        )

        # Load checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            self.logger.info(f"Resuming from checkpoint: {resume_from}")
            checkpoint_data = load_checkpoint(resume_from)
            self.train_state = checkpoint_data['state']
            self.logger.info(f"Resumed from step {checkpoint_data['step']}")
        else:
            self.logger.info("Starting training from scratch")

        # Log model info
        param_count = count_parameters(self.train_state.params)
        self.logger.info(f"Model parameters: {param_count:,}")
        self.logger.info(f"Device mesh: {self.mesh.shape}")
        self.logger.info(f"Batch size: {self.trm_config.batch_size}")
        self.logger.info(f"Recursion depth: {self.trm_config.recursion_depth}")

    def prepare_data(self, dataset_names: Optional[List[str]] = None):
        """
        Prepare training and validation data.

        Args:
            dataset_names: List of dataset names to use
        """
        self.logger.info("Preparing training data...")

        # Load datasets
        train_samples, val_samples, test_samples = self.data_handler.load_datasets(
            dataset_names=dataset_names,
            max_samples=self.trm_config.max_samples_per_dataset,
            binary_augmentation=True
        )

        self.logger.info(f"Loaded {len(train_samples)} training samples")
        self.logger.info(f"Loaded {len(val_samples)} validation samples")
        self.logger.info(f"Loaded {len(test_samples)} test samples")

        # Tokenize data
        train_tokenized = self.data_handler.tokenize_samples(
            train_samples,
            max_length=self.trm_config.max_sequence_length
        )
        val_tokenized = self.data_handler.tokenize_samples(
            val_samples,
            max_length=self.trm_config.max_sequence_length
        )

        # Create batches
        self.train_batches = self.data_handler.create_batches(
            train_tokenized,
            batch_size=self.trm_config.batch_size,
            shuffle=True
        )
        self.val_batches = self.data_handler.create_batches(
            val_tokenized,
            batch_size=self.trm_config.batch_size,
            shuffle=False
        )

        self.logger.info(f"Created {len(self.train_batches)} training batches")
        self.logger.info(f"Created {len(self.val_batches)} validation batches")

        # Store original samples for environment interaction
        self.train_samples = train_samples
        self.val_samples = val_samples

    def train_step_ppo(
        self,
        train_state: TrainState,
        batch: Dict[str, jnp.ndarray],
        key: random.PRNGKey
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """
        Single PPO training step.

        Args:
            train_state: Current training state
            batch: Training batch
            key: Random key

        Returns:
            Updated training state and metrics
        """
        def loss_fn(params):
            # Forward pass
            logits, states = self.model.forward(
                params,
                batch['prompt_input_ids'],
                deterministic=False
            )

            # Compute policy loss
            shift_logits = logits[:, :-1, :]
            shift_labels = batch['solution_input_ids'][:, 1:]

            policy_loss = optax.softmax_cross_entropy_with_integer_labels(
                shift_logits, shift_labels
            )
            policy_loss = policy_loss.mean()

            # Compute value loss (simplified)
            value_pred = jnp.mean(states['final_z'], axis=-1)
            value_target = batch['binary_decisions'].astype(jnp.float32)
            value_loss = jnp.mean((value_pred - value_target) ** 2)

            # Compute entropy bonus
            probs = jax.nn.softmax(shift_logits, axis=-1)
            entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
            entropy_bonus = jnp.mean(entropy)

            # Combine losses
            total_loss = (
                policy_loss
                + self.trm_config.ppo_value_loss_coef * value_loss
                - self.trm_config.ppo_entropy_coef * entropy_bonus
            )

            return total_loss, {
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'entropy_bonus': entropy_bonus,
                'logits': logits,
                'states': states
            }

        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(train_state.params)

        # Apply PPO clipping
        if hasattr(train_state, 'old_params'):
            ratio = jax.tree_map(
                lambda p, old_p: jnp.exp(p - old_p),
                train_state.params,
                train_state.old_params
            )
            # Simplified PPO clipping - would need proper implementation
            pass

        # Update state
        new_train_state = train_state.apply_gradients(grads=grads)

        # Compute metrics
        metrics = aux
        metrics['loss'] = loss
        metrics['grad_norm'] = jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in jax.tree_leaves(grads))
        )
        metrics['binary_decision_ratio'] = jnp.mean(batch['binary_decisions'])
        metrics['recursion_depth_mean'] = self.trm_config.recursion_depth

        return new_train_state, metrics

    def train_epoch(self, epoch: int) -> TrainingMetrics:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Training metrics
        """
        self.logger.info(f"Starting epoch {epoch}")

        epoch_metrics = []
        key = random.PRNGKey(self.config.seed + epoch)

        for step, batch in enumerate(self.train_batches):
            # Shard batch across devices
            sharded_batch = self._shard_batch(batch)

            # Training step
            key, subkey = random.split(key)
            self.train_state, step_metrics = self.train_step_ppo(
                self.train_state,
                sharded_batch,
                subkey
            )

            # Collect metrics
            epoch_metrics.append(step_metrics)

            # Log progress
            if step % self.config.logging.log_frequency == 0:
                self.logger.info(
                    f"Epoch {epoch}, Step {step}: "
                    f"Loss={step_metrics['loss']:.4f}, "
                    f"Grad Norm={step_metrics['grad_norm']:.4f}"
                )

                # Log to wandb
                if self.wandb:
                    self.wandb.log({
                        'epoch': epoch,
                        'step': epoch * len(self.train_batches) + step,
                        'train_loss': step_metrics['loss'],
                        'grad_norm': step_metrics['grad_norm'],
                        'binary_decision_ratio': step_metrics['binary_decision_ratio'],
                        'learning_rate': self.trm_config.learning_rate
                    })

            # Save checkpoint
            current_step = epoch * len(self.train_batches) + step
            if (current_step % self.config.logging.checkpoint_frequency == 0 and
                jax.process_index() == 0):
                checkpoint_path = os.path.join(
                    self.trm_config.checkpoint_dir,
                    f"checkpoint_step_{current_step}.pkl"
                )
                save_checkpoint(
                    self.train_state,
                    checkpoint_path,
                    current_step,
                    step_metrics
                )

        # Compute epoch metrics
        epoch_metrics_dict = self._compute_epoch_metrics(epoch_metrics)
        epoch_metrics_dict['epoch'] = epoch
        epoch_metrics_dict['step'] = epoch * len(self.train_batches)

        return TrainingMetrics(**epoch_metrics_dict)

    def evaluate(self, epoch: int) -> TrainingMetrics:
        """
        Evaluate model on validation set.

        Args:
            epoch: Current epoch number

        Returns:
            Validation metrics
        """
        self.logger.info(f"Evaluating epoch {epoch}")

        val_metrics = []

        for step, batch in enumerate(self.val_batches):
            # Shard batch
            sharded_batch = self._shard_batch(batch)

            # Evaluation step
            step_metrics = self.model.eval_step(
                self.train_state.params,
                sharded_batch
            )

            val_metrics.append(step_metrics)

            if step % 50 == 0:
                self.logger.info(f"Validation step {step}: Loss={step_metrics['loss']:.4f}")

        # Compute validation metrics
        val_metrics_dict = self._compute_epoch_metrics(val_metrics)
        val_metrics_dict['epoch'] = epoch
        val_metrics_dict['step'] = epoch * len(self.train_batches)

        # Add code generation evaluation
        if len(self.val_samples) > 0:
            code_metrics = self._evaluate_code_generation(epoch)
            val_metrics_dict.update(code_metrics)

        return TrainingMetrics(**val_metrics_dict)

    def _evaluate_code_generation(self, epoch: int) -> Dict[str, float]:
        """Evaluate code generation quality."""
        self.logger.info("Evaluating code generation...")

        # Sample a few validation examples
        num_samples = min(10, len(self.val_samples))
        indices = np.random.choice(len(self.val_samples), num_samples, replace=False)

        success_count = 0
        test_pass_count = 0
        total_reward = 0.0

        for i, idx in enumerate(indices):
            sample = self.val_samples[idx]

            # Reset environment
            state = self.coder_env.reset(sample)

            # Generate code using model
            if self.data_handler.tokenizer:
                tokenized = self.data_handler.tokenize_samples([sample])
                input_ids = tokenized['prompt_input_ids'][0:1]  # Batch size 1

                # Generate code
                generated_ids = self.model.generate(
                    self.train_state.params,
                    input_ids,
                    max_length=256,
                    deterministic=True
                )

                # Decode to text (simplified)
                generated_code = self._decode_ids_to_code(generated_ids[0])

                # Update state
                state.current_code = generated_code

                # Validate generated code
                validation_result = self.data_handler.validate_generated_code(
                    generated_code,
                    sample.validation,
                    sample
                )

                if validation_result['syntax_valid']:
                    success_count += 1

                if validation_result['validation_passed']:
                    test_pass_count += 1

                total_reward += state.total_reward

        # Compute metrics
        code_success_rate = success_count / num_samples
        test_pass_rate = test_pass_count / num_samples
        avg_reward = total_reward / num_samples

        return {
            'code_success_rate': code_success_rate,
            'test_pass_rate': test_pass_rate,
            'episode_reward_mean': avg_reward,
            'wall_time': time.time()
        }

    def _decode_ids_to_code(self, ids: jnp.ndarray) -> str:
        """Decode token IDs to code string (simplified)."""
        # This is a placeholder - would use actual tokenizer decoding
        return f"def generated_function():\n    # Generated code (ids: {ids.shape})"

    def _shard_batch(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Shard batch across TPU devices."""
        sharded_batch = {}

        for key, value in batch.items():
            # Create sharding specification
            sharding = NamedSharding(self.mesh, P('tp', None))
            sharded_batch[key] = jax.device_put(value, sharding)

        return sharded_batch

    def _compute_epoch_metrics(self, metrics_list: List[Dict[str, jnp.ndarray]]) -> Dict[str, float]:
        """Compute average metrics across epoch."""
        if not metrics_list:
            return {}

        # Stack metrics
        stacked = {
            key: np.array([m[key] for m in metrics_list])
            for key in metrics_list[0].keys()
        }

        # Compute averages
        avg_metrics = {
            key: float(np.mean(values))
            for key, values in stacked.items()
        }

        return avg_metrics

    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")

        # Initialize training
        self.initialize_training(resume_from)

        # Prepare data
        self.prepare_data()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            start_time = time.time()

            # Training epoch
            train_metrics = self.train_epoch(epoch)

            # Validation
            val_metrics = self.evaluate(epoch)

            # Combine metrics
            combined_metrics = {
                'train_loss': train_metrics.train_loss,
                'val_loss': val_metrics.val_loss,
                'learning_rate': self.trm_config.learning_rate,
                'grad_norm': train_metrics.grad_norm,
                'recursion_depth_mean': train_metrics.recursion_depth_mean,
                'binary_decision_ratio': train_metrics.binary_decision_ratio,
                'code_success_rate': val_metrics.code_success_rate,
                'test_pass_rate': val_metrics.test_pass_rate,
                'episode_reward_mean': val_metrics.episode_reward_mean,
                'wall_time': time.time() - start_time
            }

            # Log metrics
            self.logger.info(
                f"Epoch {epoch} completed in {combined_metrics['wall_time']:.2f}s - "
                f"Train Loss: {combined_metrics['train_loss']:.4f}, "
                f"Val Loss: {combined_metrics['val_loss']:.4f}, "
                f"Code Success: {combined_metrics['code_success_rate']:.2%}"
            )

            # Log to wandb
            if self.wandb:
                self.wandb.log(combined_metrics)

            # Log to tensorboard
            if self.tb_writer:
                for key, value in combined_metrics.items():
                    self.tb_writer.add_scalar(key, value, epoch)

            # Save best model
            if val_metrics.val_loss < best_val_loss:
                best_val_loss = val_metrics.val_loss
                patience_counter = 0

                if jax.process_index() == 0:
                    best_model_path = os.path.join(
                        self.trm_config.checkpoint_dir,
                        "best_model.pkl"
                    )
                    save_checkpoint(
                        self.train_state,
                        best_model_path,
                        epoch * len(self.train_batches),
                        combined_metrics
                    )
                    self.logger.info(f"New best model saved (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= 10:  # Configurable patience
                self.logger.info("Early stopping triggered")
                break

        # Final save
        if jax.process_index() == 0:
            final_model_path = os.path.join(
                self.trm_config.checkpoint_dir,
                "final_model.pkl"
            )
            save_checkpoint(
                self.train_state,
                final_model_path,
                num_epochs * len(self.train_batches),
                combined_metrics
            )
            self.logger.info("Final model saved")

        self.logger.info("Training completed!")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training process."""
        return {
            'total_epochs': len(self.metrics_history),
            'best_val_loss': self.best_val_loss,
            'final_metrics': self.metrics_history[-1] if self.metrics_history else {},
            'device_info': self.device_info,
            'config': self.config.to_dict()
        }


class DistributedTrainer:
    """Distributed trainer for multi-TPU training."""

    def __init__(self, config: Config):
        self.config = config
        self.mesh, self.device_info = setup_tpu()

        # Create trainer on each device
        self.model = create_trm_model(config.trm)
        self.trainer = PPOTrainer(self.model, config, self.mesh)

    def train_distributed(self, num_epochs: int, resume_from: Optional[str] = None):
        """Run distributed training."""
        # Synchronize all processes
        jax.distributed.barrier()

        # Start training
        self.trainer.train(num_epochs, resume_from)

        # Synchronize at end
        jax.distributed.barrier()


if __name__ == "__main__":
    # Test trainer
    from .config import get_config

    config = get_config("debug")
    trainer = PPOTrainer(create_trm_model(config.trm), config)

    print("Testing TRM trainer...")

    # Test initialization
    trainer.initialize_training()

    print("Trainer tests completed!")