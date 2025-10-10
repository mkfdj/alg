"""
Training module for NCA Trading Bot
Implements PPO training and NCA evolution
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit, grad, value_and_grad
from flax.training import train_state
import optax
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import wandb
from tqdm import tqdm
import time

from .config import Config
from .nca_model import AdaptiveNCA, NCAEnsemble, create_nca_loss_function, create_sample_pool, apply_damage
from .trader import TradingEnvironment, PPOAgent
from .data_handler import DataHandler


class PPOTrainer:
    """Proximal Policy Optimization trainer for trading"""

    def __init__(self, config: Config, env: TradingEnvironment, agent: PPOAgent):
        self.config = config
        self.env = env
        self.agent = agent

        # PPO hyperparameters
        self.gamma = config.rl_gamma
        self.lambda_gae = config.rl_lambda
        self.clip_eps = config.rl_clip_eps
        self.value_coef = config.rl_value_coef
        self.entropy_coef = config.rl_entropy_coef
        self.max_grad_norm = config.rl_max_grad_norm
        self.update_epochs = config.rl_update_epochs
        self.batch_size = config.rl_batch_size

        # Initialize optimizer for both networks
        self.actor_optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(config.rl_learning_rate)
        )

        self.critic_optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(config.rl_learning_rate)
        )

        # Initialize training state
        self.initialize_training()

    def initialize_training(self):
        """Initialize training state and optimizers"""
        # Reinitialize actor and critic with new optimizers
        dummy_obs = jnp.zeros((1, self.env.observation_space.shape[0]))

        actor_params = self.agent.actor.init(jrand.PRNGKey(0), dummy_obs)
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.agent.actor.apply,
            params=actor_params,
            tx=self.actor_optimizer
        )

        critic_params = self.agent.critic.init(jrand.PRNGKey(0), dummy_obs)
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.agent.critic.apply,
            params=critic_params,
            tx=self.critic_optimizer
        )

    @jit
    def compute_gae(self, rewards, values, dones, next_values):
        """Compute Generalized Advantage Estimation"""
        advantages = jnp.zeros_like(rewards)
        last_advantage = 0

        # Reverse iteration for GAE
        def compute_step(carry, inputs):
            last_advantage, last_value = carry
            reward, value, done, next_value = inputs

            delta = reward + self.gamma * next_value * (1 - done) - value
            advantage = delta + self.gamma * self.lambda_gae * (1 - done) * last_advantage

            return (advantage, value), advantage

        _, advantages = lax.scan(
            compute_step,
            (jnp.array(0.0), values[-1]),
            (rewards, values, dones, next_values),
            reverse=True
        )

        return advantages

    @jit
    def compute_losses(self, actor_params, critic_params, trajectories):
        """Compute PPO losses"""
        observations = jnp.stack([t['observation'] for t in trajectories])
        actions = jnp.array([t['action'] for t in trajectories])
        old_log_probs = jnp.array([t['log_prob'] for t in trajectories])
        advantages = jnp.array([t['advantage'] for t in trajectories])
        returns = jnp.array([t['return'] for t in trajectories])

        # Normalize advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        # Compute current action probabilities and values
        logits = vmap(self.agent.actor.apply, in_axes=(None, 0))(actor_params, observations)
        current_log_probs = jnp.log(jnp.take_along_axis(
            nn.softmax(logits), actions[:, None], axis=-1
        )).squeeze(-1)

        values = vmap(self.agent.critic.apply, in_axes=(None, 0))(critic_params, observations)

        # PPO policy loss
        ratio = jnp.exp(current_log_probs - old_log_probs)
        clipped_ratio = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        policy_loss = -jnp.minimum(ratio * advantages, clipped_ratio * advantages)
        policy_loss = jnp.mean(policy_loss)

        # Value loss
        value_loss = jnp.mean((values - returns) ** 2)

        # Entropy loss
        entropy = -jnp.sum(nn.softmax(logits) * nn.log_softmax(logits + 1e-8), axis=-1)
        entropy_loss = -jnp.mean(entropy)

        # Total loss
        total_loss = (policy_loss +
                     self.value_coef * value_loss +
                     self.entropy_coef * entropy_loss)

        return total_loss, {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'entropy': jnp.mean(entropy)
        }

    def collect_trajectories(self, num_episodes: int = 10) -> List[Dict]:
        """Collect trajectories from environment"""
        trajectories = []

        for _ in tqdm(range(num_episodes), desc="Collecting trajectories"):
            obs, _ = self.env.reset()
            episode_data = []

            while True:
                # Get action probabilities
                logits = self.agent.actor.apply(self.actor_state.params, obs)
                action_probs = nn.softmax(logits)

                # Sample action
                action = jrand.categorical(jrand.PRNGKey(int(time.time())), action_probs)

                # Store log probability
                log_prob = jnp.log(action_probs[action] + 1e-8)

                # Take step
                next_obs, reward, done, truncated, info = self.env.step(int(action))

                episode_data.append({
                    'observation': obs,
                    'action': int(action),
                    'reward': reward,
                    'next_observation': next_obs,
                    'done': done,
                    'log_prob': log_prob,
                    'value': self.agent.critic.apply(self.critic_state.params, obs)
                })

                obs = next_obs

                if done or truncated:
                    break

            # Compute returns and advantages for the episode
            if len(episode_data) > 1:
                episode_returns = self._compute_episode_returns(episode_data)
                episode_advantages = self._compute_episode_advantages(episode_data, episode_returns)

                for i, data in enumerate(episode_data):
                    data['return'] = episode_returns[i]
                    data['advantage'] = episode_advantages[i]

                trajectories.extend(episode_data)

        return trajectories

    def _compute_episode_returns(self, episode_data: List[Dict]) -> jnp.ndarray:
        """Compute discounted returns for an episode"""
        rewards = jnp.array([d['reward'] for d in episode_data])
        returns = jnp.zeros_like(rewards)

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                returns = returns.at[i].set(rewards[i])
            else:
                returns = returns.at[i].set(rewards[i] + self.gamma * returns[i + 1])

        return returns

    def _compute_episode_advantages(self, episode_data: List[Dict], returns: jnp.ndarray) -> jnp.ndarray:
        """Compute advantages for an episode"""
        values = jnp.array([d['value'] for d in episode_data])
        return returns - values

    def update_networks(self, trajectories: List[Dict]) -> Dict[str, float]:
        """Update actor and critic networks using collected trajectories"""
        metrics = {}

        # Prepare batches
        num_samples = len(trajectories)
        indices = jnp.arange(num_samples)

        for epoch in range(self.update_epochs):
            # Shuffle indices
            shuffled_indices = jrand.permutation(jrand.PRNGKey(epoch), indices)

            # Process in batches
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_indices = shuffled_indices[start:end]

                batch = [trajectories[i] for i in batch_indices]

                # Compute losses and gradients
                (total_loss, loss_metrics), (actor_grads, critic_grads) = value_and_grad(
                    self.compute_losses, has_aux=True
                )(
                    self.actor_state.params,
                    self.critic_state.params,
                    batch
                )

                # Update networks
                self.actor_state = self.actor_state.apply_gradients(grads=actor_grads)
                self.critic_state = self.critic_state.apply_gradients(grads=critic_grads)

                # Store metrics
                for key, value in loss_metrics.items():
                    metrics[f'{key}_epoch_{epoch}'] = float(value)
                metrics[f'total_loss_epoch_{epoch}'] = float(total_loss)

        return metrics

    def train(self, num_iterations: int = 1000, episodes_per_iteration: int = 10):
        """Main training loop"""
        print(f"Starting PPO training for {num_iterations} iterations")

        if self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.__dict__
            )

        best_reward = float('-inf')

        for iteration in range(num_iterations):
            # Collect trajectories
            trajectories = self.collect_trajectories(episodes_per_iteration)

            if not trajectories:
                print("No trajectories collected, skipping iteration")
                continue

            # Update networks
            metrics = self.update_networks(trajectories)

            # Evaluate current policy
            eval_reward = self.evaluate_policy(num_episodes=5)

            # Log metrics
            metrics['iteration'] = iteration
            metrics['eval_reward'] = eval_reward
            metrics['num_trajectories'] = len(trajectories)

            # Log to wandb
            if self.config.wandb_project:
                wandb.log(metrics)

            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Eval Reward = {eval_reward:.2f}")

            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                self.save_checkpoint(f'best_model_iteration_{iteration}')

            if iteration % 100 == 0:
                self.save_checkpoint(f'checkpoint_iteration_{iteration}')

        if self.config.wandb_project:
            wandb.finish()

        print(f"Training completed. Best reward: {best_reward:.2f}")

    def evaluate_policy(self, num_episodes: int = 10) -> float:
        """Evaluate current policy"""
        total_reward = 0

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Get action (deterministic for evaluation)
                logits = self.agent.actor.apply(self.actor_state.params, obs)
                action = jnp.argmax(nn.softmax(logits))

                # Take step
                obs, reward, done, truncated, info = self.env.step(int(action))
                episode_reward += reward

            total_reward += episode_reward

        return total_reward / num_episodes

    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        import os
        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f'{checkpoint_name}.pkl')

        checkpoint_data = {
            'actor_state': self.actor_state,
            'critic_state': self.critic_state,
            'config': self.config,
            'iteration': checkpoint_name
        }

        import pickle
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        import pickle
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        self.actor_state = checkpoint_data['actor_state']
        self.critic_state = checkpoint_data['critic_state']

        print(f"Checkpoint loaded: {checkpoint_path}")


class NCATrainer:
    """Trainer for Neural Cellular Automata"""

    def __init__(self, config: Config, data_handler: DataHandler):
        self.config = config
        self.data_handler = data_handler

        # Initialize NCA ensemble
        self.nca_ensemble = NCAEnsemble(config)
        self.rng_key = jrand.PRNGKey(42)
        self.nca_ensemble.initialize(self.rng_key)

        # Training parameters
        self.nca_learning_rate = config.nca_learning_rate
        self.sample_pool_size = 1024
        self.damage_prob = 0.1

        # Initialize sample pool
        self.sample_pool = None
        self.initialize_sample_pool()

    def initialize_sample_pool(self):
        """Initialize sample pool with seed grids"""
        initial_grid = self.nca_ensemble.models[0].initialize_grid(1, seed=42)
        self.sample_pool = create_sample_pool(initial_grid, self.sample_pool_size)

    def train_nca_ensemble(self, num_iterations: int = 1000):
        """Train NCA ensemble on target patterns"""
        print(f"Training NCA ensemble for {num_iterations} iterations")

        # Load training data
        sequences, targets = self.data_handler.load_synthetic_data(
            n_samples=self.sample_pool_size,
            complexity="medium"
        )

        # Create target patterns from targets
        target_patterns = []
        for target in targets:
            target_pattern = self.data_handler.create_target_pattern(
                np.array([target]), self.config.nca_grid_size
            )
            target_patterns.append(target_pattern)

        target_patterns = jnp.stack(target_patterns)

        if self.config.wandb_project:
            wandb.init(
                project=f"{self.config.wandb_project}_nca",
                entity=self.config.wandb_entity,
                config=self.config.__dict__
            )

        for iteration in tqdm(range(num_iterations), desc="Training NCA Ensemble"):
            # Sample batch from pool
            batch_indices = jrand.choice(
                self.rng_key,
                len(self.sample_pool),
                shape=(self.config.rl_batch_size,),
                replace=False
            )

            batch_grids = self.sample_pool[batch_indices]
            batch_targets = target_patterns[batch_indices]

            # Train each NCA in ensemble
            ensemble_metrics = {}

            for model_idx, (model, state) in enumerate(zip(self.nca_ensemble.models, self.nca_ensemble.states)):
                # Create loss function for this model
                loss_fn = create_nca_loss_function(batch_targets[model_idx])

                # Train step
                self.rng_key, subkey = jrand.split(self.rng_key)
                new_state, metrics = self._train_nca_step(
                    state, model, batch_grids[model_idx], subkey, loss_fn
                )

                self.nca_ensemble.states[model_idx] = new_state

                # Store metrics
                for key, value in metrics.items():
                    ensemble_metrics[f'model_{model_idx}_{key}'] = float(value)

            # Update sample pool
            self._update_sample_pool(batch_indices, batch_targets)

            # Log metrics
            ensemble_metrics['iteration'] = iteration

            if self.config.wandb_project:
                wandb.log(ensemble_metrics)

            if iteration % 100 == 0:
                avg_loss = ensemble_metrics.get('model_0_total_loss', 0)
                print(f"Iteration {iteration}: Loss = {avg_loss:.4f}")

        if self.config.wandb_project:
            wandb.finish()

        print("NCA ensemble training completed")

    def _train_nca_step(self, state, model, batch, rng_key, loss_fn):
        """Single training step for NCA"""
        grad_fn = value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params, model, batch, rng_key)

        # Apply gradients
        new_state = state.apply_gradients(grads=grads)

        metrics = {
            'loss': loss,
            **aux
        }

        return new_state, metrics

    def _update_sample_pool(self, batch_indices, target_patterns):
        """Update sample pool with evolved grids"""
        for i, idx in enumerate(batch_indices):
            # Evolve grid
            grid = self.sample_pool[idx]
            evolved_grid = self.nca_ensemble.models[0].evolve(
                grid, self.rng_key, steps=self.config.nca_evolution_steps
            )

            # Apply damage for robustness
            damaged_grid = apply_damage(evolved_grid, self.damage_prob)

            # Replace in sample pool
            self.sample_pool = self.sample_pool.at[idx].set(damaged_grid)


class CombinedTrainer:
    """Combined trainer for NCA and RL components"""

    def __init__(self, config: Config):
        self.config = config
        self.data_handler = DataHandler(config)
        self.env = TradingEnvironment(config, self.data_handler)
        self.agent = PPOAgent(
            config,
            observation_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n
        )

        self.nca_trainer = NCATrainer(config, self.data_handler)
        self.ppo_trainer = PPOTrainer(config, self.env, self.agent)

    def train(self, nca_iterations: int = 1000, ppo_iterations: int = 1000):
        """Train both NCA and PPO components"""
        print("Starting combined training")

        # Phase 1: Train NCA ensemble
        print("\n=== Phase 1: Training NCA Ensemble ===")
        self.nca_trainer.train_nca_ensemble(nca_iterations)

        # Phase 2: Train PPO agent
        print("\n=== Phase 2: Training PPO Agent ===")
        self.ppo_trainer.train(ppo_iterations)

        print("\nCombined training completed")

    def evaluate(self, num_episodes: int = 100):
        """Evaluate the complete system"""
        print(f"Evaluating complete system for {num_episodes} episodes")

        total_rewards = []
        total_returns = []
        shapre_ratios = []
        max_drawdowns = []

        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Get action from trained agent
                logits = self.agent.actor.apply(self.ppo_trainer.actor_state.params, obs)
                action = jnp.argmax(nn.softmax(logits))

                # Take step
                obs, reward, done, truncated, info = self.env.step(int(action))
                episode_reward += reward

            # Collect metrics
            total_rewards.append(episode_reward)
            total_returns.append(info['return_pct'])
            shapre_ratios.append(info['sharpe_ratio'])
            max_drawdowns.append(info['max_drawdown'])

        # Print evaluation results
        print(f"\n=== Evaluation Results ===")
        print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Average Return: {np.mean(total_returns):.2f}% ± {np.std(total_returns):.2f}%")
        print(f"Average Sharpe Ratio: {np.mean(shapre_ratios):.2f} ± {np.std(shapre_ratios):.2f}")
        print(f"Average Max Drawdown: {np.mean(max_drawdowns):.2f} ± {np.std(max_drawdowns):.2f}")

        return {
            'avg_reward': np.mean(total_rewards),
            'avg_return': np.mean(total_returns),
            'avg_sharpe': np.mean(shapre_ratios),
            'avg_drawdown': np.mean(max_drawdowns),
            'std_reward': np.std(total_rewards),
            'std_return': np.std(total_returns),
            'std_sharpe': np.std(shapre_ratios),
            'std_drawdown': np.std(max_drawdowns)
        }