"""
Trading environment and RL agent implementation
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
from jax import lax, vmap, jit, grad
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces
import alpaca_trade_api as tradeapi
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime, timedelta

from .config import Config
from .nca_model import AdaptiveNCA, NCAEnsemble
from .data_handler import DataHandler


class TradingEnvironment(gym.Env):
    """Trading environment for reinforcement learning"""

    def __init__(self, config: Config, data_handler: DataHandler):
        super().__init__()
        self.config = config
        self.data_handler = data_handler

        # Trading parameters
        self.initial_balance = config.trading_initial_balance
        self.balance = self.initial_balance
        self.position = 0.0  # Current position (0 = no position, positive = long, negative = short)
        self.position_size = 0.0  # Size of current position
        self.portfolio_value = self.initial_balance
        self.trade_history = []

        # Data parameters
        self.sequence_length = config.data_sequence_length
        self.current_step = 0
        self.max_steps = 1000

        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Observation space: NCA grid + portfolio state
        grid_height, grid_width = config.nca_grid_size
        grid_channels = config.nca_channels
        nca_features = grid_height * grid_width * grid_channels

        portfolio_features = 8  # balance, position, portfolio_value, etc.
        market_features = 20   # additional market indicators
        total_features = nca_features + portfolio_features + market_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )

        # Initialize NCA ensemble
        self.nca_ensemble = None
        self.initialize_nca()

        # Market data
        self.current_data = None
        self.market_data = None

    def initialize_nca(self):
        """Initialize NCA ensemble"""
        self.nca_ensemble = NCAEnsemble(self.config)
        rng_key = jrand.PRNGKey(42)
        self.nca_ensemble.initialize(rng_key)

    def reset(self, seed: Optional[int] = None):
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)

        # Reset portfolio state
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_size = 0.0
        self.portfolio_value = self.initial_balance
        self.trade_history = []

        # Reset data
        self.current_step = 0

        # Load initial market data
        self.load_market_data()

        # Get initial observation
        observation = self._get_observation()

        return observation, {}

    def load_market_data(self):
        """Load current market data"""
        tickers = self.config.top_tickers[:5]  # Use top 5 tickers for environment
        self.market_data = {}

        for ticker in tickers:
            try:
                df = self.data_handler.load_realtime_data(ticker, period="1mo")
                if not df.empty:
                    self.market_data[ticker] = df
            except Exception as e:
                print(f"Error loading data for {ticker}: {e}")

    def step(self, action):
        """Execute one trading step"""
        # Execute action
        reward = self._execute_action(action)

        # Update market data
        self._update_market_data()

        # Get new observation
        observation = self._get_observation()

        # Check if episode is done
        done = self._is_done()

        # Get info
        info = self._get_info()

        self.current_step += 1

        return observation, reward, done, False, info

    def _execute_action(self, action) -> float:
        """Execute trading action and return reward"""
        if not self.market_data or len(self.market_data) == 0:
            return 0.0

        # Get current price (use first ticker's price as reference)
        ticker = list(self.market_data.keys())[0]
        current_price = self.market_data[ticker]['close'].iloc[-1]

        if action == 1:  # Buy
            position_size = self._calculate_position_size(current_price)
            if position_size > 0 and self.balance >= position_size * current_price:
                self.position_size = position_size
                self.position = 1.0  # Long position
                cost = position_size * current_price
                self.balance -= cost + cost * self.config.trading_commission

                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'price': current_price,
                    'size': position_size,
                    'cost': cost
                })

        elif action == 2:  # Sell
            if self.position_size > 0:
                # Close long position
                proceeds = self.position_size * current_price
                commission = proceeds * self.config.trading_commission
                self.balance += proceeds - commission

                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'price': current_price,
                    'size': self.position_size,
                    'proceeds': proceeds - commission
                })

                self.position = 0.0
                self.position_size = 0.0

        # Update portfolio value
        self.portfolio_value = self.balance + (self.position_size * current_price if self.position_size > 0 else 0)

        # Calculate reward (profit - risk penalty)
        profit = self.portfolio_value - self.initial_balance
        risk_penalty = self._calculate_risk_penalty()
        reward = profit - risk_penalty

        return reward

    def _calculate_position_size(self, current_price: float) -> float:
        """Calculate position size using Kelly criterion"""
        # Simplified Kelly calculation
        win_rate = 0.55  # Assumed win rate
        avg_win = 0.05   # 5% average win
        avg_loss = 0.03  # 3% average loss

        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate

        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

        # Calculate position size
        available_balance = self.balance * self.config.trading_max_position_size
        position_value = available_balance * kelly_fraction
        position_size = position_value / current_price

        return position_size

    def _calculate_risk_penalty(self) -> float:
        """Calculate risk penalty based on current position"""
        if self.position_size == 0:
            return 0.0

        # Risk penalty based on position size and volatility
        volatility_penalty = self.position_size * 0.01  # 1% of position size
        concentration_penalty = (self.position_size / self.portfolio_value) * 0.005

        return volatility_penalty + concentration_penalty

    def _update_market_data(self):
        """Update market data with latest prices"""
        for ticker in self.market_data:
            try:
                latest_data = self.data_handler.load_realtime_data(ticker, period="1d")
                if not latest_data.empty:
                    # Append latest data
                    self.market_data[ticker] = pd.concat([
                        self.market_data[ticker],
                        latest_data.tail(1)
                    ]).drop_duplicates()
            except Exception as e:
                print(f"Error updating data for {ticker}: {e}")

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if not self.market_data:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # Get NCA grid from first ticker
        ticker = list(self.market_data.keys())[0]
        df = self.market_data[ticker]

        if len(df) < self.sequence_length:
            # Pad with zeros if not enough data
            padding = self.sequence_length - len(df)
            sequence = np.zeros((self.sequence_length, 10))
            sequence[-len(df):] = df[['open', 'high', 'low', 'close', 'volume',
                                      'rsi_14', 'macd', 'bollinger_width', 'atr_14', 'returns']].values
        else:
            sequence = df[['open', 'high', 'low', 'close', 'volume',
                           'rsi_14', 'macd', 'bollinger_width', 'atr_14', 'returns']].tail(self.sequence_length).values

        # Create NCA grid
        nca_grid = self.data_handler.create_nca_grid(sequence)
        nca_features = nca_grid.flatten()

        # Portfolio state features
        portfolio_features = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position,                         # Position type
            self.position_size / self.initial_balance,  # Normalized position size
            self.portfolio_value / self.initial_balance,  # Normalized portfolio value
            len(self.trade_history),               # Number of trades
            self.current_step / self.max_steps,    # Progress through episode
            self._calculate_sharpe_ratio(),        # Sharpe ratio
            self._calculate_max_drawdown()         # Max drawdown
        ])

        # Market features
        market_features = self._get_market_features()

        # Combine all features
        observation = np.concatenate([nca_features, portfolio_features, market_features])

        return observation.astype(np.float32)

    def _get_market_features(self) -> np.ndarray:
        """Get market-wide features"""
        features = []

        for ticker in list(self.market_data.keys())[:4]:  # Use up to 4 tickers
            df = self.market_data[ticker]
            if len(df) > 0:
                latest = df.iloc[-1]
                features.extend([
                    latest['returns'] if not pd.isna(latest['returns']) else 0,
                    latest['rsi_14'] if not pd.isna(latest['rsi_14']) else 50,
                    latest['macd'] if not pd.isna(latest['macd']) else 0,
                    latest['atr_14'] if not pd.isna(latest['atr_14']) else 0,
                    latest['volume'] / df['volume'].mean() if not pd.isna(latest['volume']) and df['volume'].mean() > 0 else 1
                ])
            else:
                features.extend([0, 50, 0, 0, 1])

        # Pad with zeros if less than 4 tickers
        while len(features) < 20:
            features.extend([0, 50, 0, 0, 1])

        return np.array(features[:20])

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of trading performance"""
        if len(self.trade_history) < 2:
            return 0.0

        returns = []
        prev_value = self.initial_balance

        for trade in self.trade_history:
            if trade['action'] == 'sell':
                current_value = self.balance + (self.position_size * trade['price'] if self.position_size > 0 else 0)
                ret = (current_value - prev_value) / prev_value
                returns.append(ret)
                prev_value = current_value

        if not returns:
            return 0.0

        returns = np.array(returns)
        return np.mean(returns) / (np.std(returns) + 1e-8)

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.trade_history) < 2:
            return 0.0

        peak_value = self.initial_balance
        max_drawdown = 0.0

        for trade in self.trade_history:
            if trade['action'] == 'sell':
                current_value = self.balance + (self.position_size * trade['price'] if self.position_size > 0 else 0)
                peak_value = max(peak_value, current_value)
                drawdown = (peak_value - current_value) / peak_value
                max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _is_done(self) -> bool:
        """Check if episode is done"""
        # Done if:
        # 1. Portfolio value drops below 50% of initial
        # 2. Maximum steps reached
        # 3. Maximum drawdown exceeded

        portfolio_loss = (self.initial_balance - self.portfolio_value) / self.initial_balance
        max_drawdown = self._calculate_max_drawdown()

        return (portfolio_loss > 0.5 or
                self.current_step >= self.max_steps or
                max_drawdown > 0.3)

    def _get_info(self) -> Dict:
        """Get additional information"""
        return {
            'balance': self.balance,
            'position': self.position,
            'position_size': self.position_size,
            'portfolio_value': self.portfolio_value,
            'num_trades': len(self.trade_history),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown(),
            'return_pct': ((self.portfolio_value - self.initial_balance) / self.initial_balance) * 100
        }


class PPOAgent:
    """Proximal Policy Optimization agent for trading"""

    def __init__(self, config: Config, observation_dim: int, action_dim: int):
        self.config = config
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Create actor and critic networks
        self.actor = self._create_actor_network()
        self.critic = self._create_critic_network()

        # Initialize networks
        self.rng_key = jrand.PRNGKey(42)
        self.actor_state = None
        self.critic_state = None
        self.initialize_networks()

    def _create_actor_network(self):
        """Create actor network"""
        class Actor(nn.Module):
            hidden_dims: List[int]
            action_dim: int

            @nn.compact
            def __call__(self, x):
                for hidden_dim in self.hidden_dims:
                    x = nn.Dense(hidden_dim)(x)
                    x = nn.tanh(x)
                # Output action logits
                logits = nn.Dense(self.action_dim)(x)
                return logits

        return Actor(
            hidden_dims=[128, 64, 32],
            action_dim=self.action_dim
        )

    def _create_critic_network(self):
        """Create critic network"""
        class Critic(nn.Module):
            hidden_dims: List[int]

            @nn.compact
            def __call__(self, x):
                for hidden_dim in self.hidden_dims:
                    x = nn.Dense(hidden_dim)(x)
                    x = nn.tanh(x)
                # Output value
                value = nn.Dense(1)(x)
                return value.squeeze(-1)

        return Critic(hidden_dims=[128, 64, 32])

    def initialize_networks(self):
        """Initialize actor and critic networks"""
        # Initialize actor
        dummy_obs = jnp.zeros((1, self.observation_dim))
        actor_params = self.actor.init(self.rng_key, dummy_obs)
        actor_tx = optax.adam(self.config.rl_learning_rate)
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=actor_tx
        )

        # Initialize critic
        critic_params = self.critic.init(self.rng_key, dummy_obs)
        critic_tx = optax.adam(self.config.rl_learning_rate)
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=critic_tx
        )

    @jax.jit
    def select_action(self, actor_params, observation, rng_key):
        """Select action using actor network"""
        logits = self.actor.apply(actor_params, observation)
        action_probs = nn.softmax(logits)
        action = jrand.categorical(rng_key, action_probs)
        return action, action_probs

    @jax.jit
    def get_value(self, critic_params, observation):
        """Get state value from critic network"""
        return self.critic.apply(critic_params, observation)

    def update(self, trajectories):
        """Update actor and critic networks using PPO"""
        # Implement PPO update logic here
        # This would include:
        # 1. Compute advantages using GAE
        # 2. Compute PPO loss
        # 3. Update networks
        # For brevity, this is a placeholder implementation

        metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy_loss': 0.0
        }

        return metrics


class AlpacaTrader:
    """Interface to Alpaca trading API"""

    def __init__(self, config: Config, paper_mode: bool = True):
        self.config = config
        self.paper_mode = paper_mode

        # Initialize Alpaca API
        api_config = config.get_alpaca_config(paper_mode)
        self.api = tradeapi.REST(
            key_id=api_config['key_id'],
            secret_key=api_config['secret_key'],
            base_url=api_config['base_url']
        )

        # Check account status
        try:
            self.account = self.api.get_account()
            print(f"Connected to Alpaca {'Paper' if paper_mode else 'Live'} Trading")
            print(f"Account ID: {self.account.id}")
            print(f"Buying Power: ${self.account.buying_power}")
        except Exception as e:
            print(f"Error connecting to Alpaca: {e}")

    def submit_order(self, symbol: str, qty: float, side: str,
                    order_type: str = 'market', time_in_force: str = 'day'):
        """Submit order to Alpaca"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            print(f"Order submitted: {side} {qty} shares of {symbol}")
            return order
        except Exception as e:
            print(f"Error submitting order: {e}")
            return None

    def get_positions(self):
        """Get current positions"""
        try:
            return self.api.list_positions()
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []

    def get_account(self):
        """Get account information"""
        try:
            return self.api.get_account()
        except Exception as e:
            print(f"Error getting account info: {e}")
            return None

    def cancel_all_orders(self):
        """Cancel all open orders"""
        try:
            self.api.cancel_all_orders()
            print("All orders cancelled")
        except Exception as e:
            print(f"Error cancelling orders: {e}")


class TradingBot:
    """Main trading bot that combines NCA and RL"""

    def __init__(self, config: Config):
        self.config = config
        self.data_handler = DataHandler(config)
        self.env = TradingEnvironment(config, self.data_handler)
        self.agent = PPOAgent(
            config,
            observation_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n
        )
        self.alpaca = AlpacaTrader(config, paper_mode=True)

    def train(self, num_episodes: int = 1000):
        """Train the trading bot"""
        print(f"Starting training for {num_episodes} episodes")

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            trajectory = []

            while not done:
                # Select action
                action, action_probs = self.agent.select_action(
                    self.agent.actor_state.params, obs, self.agent.rng_key
                )

                # Take step
                next_obs, reward, done, _, info = self.env.step(action.item())

                # Store trajectory
                trajectory.append({
                    'observation': obs,
                    'action': action.item(),
                    'reward': reward,
                    'next_observation': next_obs,
                    'done': done
                })

                obs = next_obs
                episode_reward += reward

                self.agent.rng_key, _ = jrand.split(self.agent.rng_key)

            # Update agent
            metrics = self.agent.update(trajectory)

            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                      f"Portfolio Value: ${info['portfolio_value']:.2f}, "
                      f"Sharpe: {info['sharpe_ratio']:.2f}")

    def evaluate(self, num_episodes: int = 100):
        """Evaluate the trading bot"""
        print(f"Evaluating for {num_episodes} episodes")

        total_rewards = []
        total_returns = []
        total_sharpe = []
        total_drawdown = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select action (deterministic for evaluation)
                action_probs = nn.softmax(
                    self.agent.actor.apply(self.agent.actor_state.params, obs)
                )
                action = jnp.argmax(action_probs)

                # Take step
                next_obs, reward, done, _, info = self.env.step(action.item())

                obs = next_obs
                episode_reward += reward

            # Collect metrics
            total_rewards.append(episode_reward)
            total_returns.append(info['return_pct'])
            total_sharpe.append(info['sharpe_ratio'])
            total_drawdown.append(info['max_drawdown'])

        # Print evaluation results
        print(f"Evaluation Results:")
        print(f"Average Reward: {np.mean(total_rewards):.2f}")
        print(f"Average Return: {np.mean(total_returns):.2f}%")
        print(f"Average Sharpe Ratio: {np.mean(total_sharpe):.2f}")
        print(f"Average Max Drawdown: {np.mean(total_drawdown):.2f}")

    def start_live_trading(self, paper_mode: bool = True):
        """Start live trading"""
        if not paper_mode:
            confirm = input("WARNING: This will start live trading with real money. Continue? (y/N): ")
            if confirm.lower() != 'y':
                print("Live trading cancelled")
                return

        print(f"Starting {'paper' if paper_mode else 'live'} trading...")

        while True:
            try:
                # Get current observation
                obs, _ = self.env.reset()

                # Select action
                action_probs = nn.softmax(
                    self.agent.actor.apply(self.agent.actor_state.params, obs)
                )
                action = jnp.argmax(action_probs)

                # Execute trade if not paper mode
                if not paper_mode and action != 0:  # Not hold
                    # Execute actual trade via Alpaca
                    # This would need proper implementation
                    pass

                # Wait for next trading opportunity
                # In practice, this would be market hours
                import time
                time.sleep(60)  # Wait 1 minute

            except KeyboardInterrupt:
                print("\nTrading stopped by user")
                break
            except Exception as e:
                print(f"Error in live trading: {e}")
                time.sleep(60)  # Wait before retrying