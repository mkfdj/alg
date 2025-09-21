"""
Trading environment and execution module for NCA Trading Bot.

This module provides Gym-compatible trading environments, Alpaca API integration
for paper trading, and comprehensive risk management capabilities.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.stream import Stream
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from .config import get_config
from .nca_model import NCATradingModel


class TradingEnvironment(gym.Env):
    """
    Gym-compatible trading environment for reinforcement learning.

    Provides a complete trading simulation environment with realistic
    market conditions, transaction costs, and risk management.
    """

    def __init__(self, data: pd.DataFrame, config, window_size: int = 60):
        """
        Initialize trading environment.

        Args:
            data: Historical market data
            config: Trading configuration
            window_size: Size of observation window
        """
        super().__init__()

        self.config = config
        self.data = data
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)

        # Environment state
        self.current_step = 0
        self.initial_balance = 100000  # $100k starting balance
        self.balance = self.initial_balance
        self.position = 0  # Number of shares
        self.position_value = 0
        self.portfolio_value = self.initial_balance
        self.trades = []
        self.total_reward = 0

        # Market data
        self.prices = data['Close'].values
        self.highs = data['High'].values
        self.lows = data['Low'].values
        self.volumes = data['Volume'].values

        # Technical indicators
        self.features = self._prepare_features()

        # Action space: [hold, buy, sell] with position sizing
        self.action_space = spaces.MultiDiscrete([3, 10])  # 3 actions, 10 position sizes

        # Observation space: window of features
        obs_shape = (window_size, self.features.shape[1])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        # Episode tracking
        self.episode_length = len(data) - window_size
        self.max_position = config.trading.max_position_size
        self.transaction_cost = config.trading.commission_per_share

    def _prepare_features(self) -> np.ndarray:
        """Prepare feature matrix from market data."""
        # Use technical indicators from data if available
        feature_columns = [col for col in self.data.columns
                          if col not in ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]

        if feature_columns:
            return self.data[feature_columns].values
        else:
            # Fallback to basic features
            return self.data[['Close']].values

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Reset options

        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_value = 0
        self.portfolio_value = self.initial_balance
        self.trades = []
        self.total_reward = 0

        # Get initial observation
        observation = self._get_observation()
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'step': self.current_step
        }

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action array [action_type, position_size]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        action_type, position_size = action
        position_size = position_size / 10.0  # Normalize to [0, 1]

        # Execute trade
        reward = self._execute_trade(action_type, position_size)

        # Update portfolio value
        current_price = self.prices[self.current_step + self.window_size]
        self.position_value = self.position * current_price
        self.portfolio_value = self.balance + self.position_value

        # Move to next step
        self.current_step += 1

        # Check termination conditions
        terminated = self.current_step >= self.episode_length
        truncated = self.portfolio_value < self.initial_balance * (1 - self.config.trading.max_drawdown)

        # Get next observation
        observation = self._get_observation()

        # Calculate reward
        portfolio_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        risk_adjusted_reward = reward + 0.1 * portfolio_return

        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'current_price': current_price,
            'reward': reward,
            'total_reward': self.total_reward,
            'step': self.current_step
        }

        return observation, risk_adjusted_reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation window."""
        start_idx = self.current_step
        end_idx = start_idx + self.window_size

        if end_idx > len(self.features):
            # Pad with zeros if at the end
            obs = np.zeros((self.window_size, self.features.shape[1]))
            valid_data = self.features[start_idx:]
            obs[:len(valid_data)] = valid_data
        else:
            obs = self.features[start_idx:end_idx]

        return obs.astype(np.float32)

    def _execute_trade(self, action_type: int, position_size: float) -> float:
        """
        Execute trade based on action.

        Args:
            action_type: 0=hold, 1=buy, 2=sell
            position_size: Position size fraction

        Returns:
            Reward for this action
        """
        current_price = self.prices[self.current_step + self.window_size]
        reward = 0

        if action_type == 0:  # Hold
            reward = 0
        elif action_type == 1:  # Buy
            max_shares = int((self.balance * self.max_position) / current_price)
            shares_to_buy = int(max_shares * position_size)

            if shares_to_buy > 0 and self.balance >= shares_to_buy * current_price:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                if self.balance >= cost:
                    self.balance -= cost
                    self.position += shares_to_buy

                    trade = {
                        'type': 'buy',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'timestamp': self.current_step
                    }
                    self.trades.append(trade)

                    reward = 0.1  # Small positive reward for entering position
        elif action_type == 2:  # Sell
            shares_to_sell = int(abs(self.position) * position_size)

            if shares_to_sell > 0 and self.position > 0:
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.position -= shares_to_sell

                trade = {
                    'type': 'sell',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'timestamp': self.current_step
                }
                self.trades.append(trade)

                # Reward based on profit/loss
                profit = revenue - (shares_to_sell * self.trades[-2]['price'] if len(self.trades) > 1 else 0)
                reward = profit / self.initial_balance

        return reward

    def render(self, mode: str = 'human'):
        """Render environment state."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Position: {self.position} shares")
            print(f"Total Reward: {self.total_reward:.4f}")


class AlpacaTrader:
    """
    Alpaca API integration for paper trading.

    Provides real-time trading capabilities with risk management
    and position tracking.
    """

    def __init__(self, config):
        """
        Initialize Alpaca trader.

        Args:
            config: Trading configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize Alpaca API
        self.api = REST(
            key_id=config.api.alpaca_api_key,
            secret_key=config.api.alpaca_secret_key,
            base_url=config.api.alpaca_base_url
        )

        # Trading state
        self.account = None
        self.positions = {}
        self.orders = []
        self.is_trading = False

        # Risk management
        self.daily_pnl = 0
        self.max_daily_loss = -config.trading.max_daily_loss * 100000  # Convert to dollars
        self.max_position_size = config.trading.max_position_size

        # Initialize account info
        self._update_account_info()

    def _update_account_info(self):
        """Update account information from Alpaca."""
        try:
            self.account = self.api.get_account()
            self.logger.info(f"Account updated: ${self.account.equity}")
        except Exception as e:
            self.logger.error(f"Failed to update account info: {e}")

    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions."""
        try:
            positions = self.api.list_positions()
            self.positions = {
                pos.symbol: {
                    'qty': int(pos.qty),
                    'market_value': float(pos.market_value),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'unrealized_pl': float(pos.unrealized_pl)
                }
                for pos in positions
            }
            return self.positions
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return {}

    def place_order(self, symbol: str, qty: int, side: str,
                   order_type: str = 'market', time_in_force: str = 'day') -> Optional[str]:
        """
        Place trading order.

        Args:
            symbol: Stock symbol
            qty: Quantity to trade
            side: 'buy' or 'sell'
            order_type: Order type ('market', 'limit', etc.)
            time_in_force: Time in force

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Check risk limits
            if not self._check_risk_limits(symbol, qty, side):
                return None

            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )

            self.orders.append({
                'id': order.id,
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'status': order.status
            })

            self.logger.info(f"Order placed: {side} {qty} {symbol}")
            return order.id

        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return None

    def _check_risk_limits(self, symbol: str, qty: int, side: str) -> bool:
        """
        Check if trade complies with risk limits.

        Args:
            symbol: Stock symbol
            qty: Quantity
            side: Trade side

        Returns:
            True if trade is allowed
        """
        # Check daily loss limit
        if self.daily_pnl < self.max_daily_loss:
            self.logger.warning("Daily loss limit reached")
            return False

        # Check position size limits
        current_positions = self.get_positions()
        total_value = sum(pos['market_value'] for pos in current_positions.values())

        # Estimate new position value
        try:
            quote = self.api.get_latest_quote(symbol)
            estimated_price = quote.askprice if side == 'buy' else quote.bidprice
            new_position_value = qty * estimated_price

            if (total_value + new_position_value) > float(self.account.buying_power) * self.max_position_size:
                self.logger.warning("Position size limit reached")
                return False
        except Exception as e:
            self.logger.error(f"Failed to check position limits: {e}")
            return False

        return True

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel pending order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        try:
            self.api.cancel_order(order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """
        Get order status.

        Args:
            order_id: Order ID

        Returns:
            Order status dictionary
        """
        try:
            order = self.api.get_order(order_id)
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'filled_qty': order.filled_qty,
                'status': order.status,
                'filled_avg_price': order.filled_avg_price
            }
        except Exception as e:
            self.logger.error(f"Failed to get order status: {e}")
            return None


class TradingAgent:
    """
    High-level trading agent that combines NCA model with trading execution.

    Provides decision-making, risk management, and trade execution capabilities.
    """

    def __init__(self, model: NCATradingModel, config):
        """
        Initialize trading agent.

        Args:
            model: NCA trading model
            config: Trading configuration
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Trading components
        self.alpaca_trader = AlpacaTrader(config)
        self.risk_manager = RiskManager(config)

        # Agent state
        self.is_live_trading = False
        self.predictions = []
        self.trades = []

        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }

    def make_decision(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make trading decision based on market data.

        Args:
            market_data: Current market data

        Returns:
            Trading decision dictionary
        """
        try:
            # Prepare input for model
            features = self._prepare_features(market_data)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)

            # Get model prediction
            prediction = self.model.predict(features_tensor)

            # Apply risk management
            decision = self.risk_manager.evaluate_trade(
                prediction, market_data, self.alpaca_trader.get_positions()
            )

            self.predictions.append({
                'timestamp': datetime.now(),
                'prediction': prediction,
                'decision': decision
            })

            return decision

        except Exception as e:
            self.logger.error(f"Error making trading decision: {e}")
            return {'action': 'hold', 'confidence': 0, 'reason': 'error'}

    def execute_trade(self, decision: Dict[str, Any]) -> bool:
        """
        Execute trading decision.

        Args:
            decision: Trading decision dictionary

        Returns:
            True if trade executed successfully
        """
        try:
            if decision['action'] == 'hold':
                return True

            # Get current positions and account info
            positions = self.alpaca_trader.get_positions()
            account = self.alpaca_trader.account

            if not account:
                self.logger.error("No account information available")
                return False

            # Calculate position size
            position_size = self._calculate_position_size(
                decision, float(account.buying_power)
            )

            if position_size <= 0:
                return False

            # Place order
            symbol = decision.get('symbol', 'SPY')  # Default to SPY
            side = 'buy' if decision['action'] == 'buy' else 'sell'

            order_id = self.alpaca_trader.place_order(
                symbol=symbol,
                qty=position_size,
                side=side
            )

            if order_id:
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': side,
                    'quantity': position_size,
                    'order_id': order_id,
                    'decision': decision
                }
                self.trades.append(trade_record)

                # Update performance metrics
                self._update_performance_metrics(trade_record)

                self.logger.info(f"Trade executed: {side} {position_size} {symbol}")
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False

    def _prepare_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input."""
        # Extract relevant features from market data
        feature_columns = [col for col in market_data.columns
                          if col not in ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]

        if feature_columns:
            return market_data[feature_columns].values[-self.config.data.sequence_length:]
        else:
            return market_data[['Close']].values[-self.config.data.sequence_length:]

    def _calculate_position_size(self, decision: Dict[str, Any], available_capital: float) -> int:
        """
        Calculate position size based on risk management rules.

        Args:
            decision: Trading decision
            available_capital: Available capital for trading

        Returns:
            Position size in shares
        """
        confidence = decision.get('confidence', 0)
        risk_per_trade = self.config.trading.risk_per_trade

        # Base position size on confidence and risk limits
        max_position_value = available_capital * self.config.trading.max_position_size
        risk_amount = available_capital * risk_per_trade

        # Adjust position size based on confidence
        position_value = max_position_value * confidence

        # Ensure position doesn't exceed risk limits
        position_value = min(position_value, risk_amount / 0.02)  # Assume 2% stop loss

        # Get estimated price
        symbol = decision.get('symbol', 'SPY')
        try:
            quote = self.alpaca_trader.api.get_latest_quote(symbol)
            estimated_price = quote.askprice if decision['action'] == 'buy' else quote.bidprice
        except:
            estimated_price = 100  # Fallback price

        shares = int(position_value / estimated_price)
        return max(shares, 0)

    def _update_performance_metrics(self, trade: Dict):
        """Update performance tracking metrics."""
        self.performance_metrics['total_trades'] += 1

        # Update win/loss ratio (simplified)
        if trade['decision'].get('expected_profit', 0) > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if self.performance_metrics['total_trades'] == 0:
            return self.performance_metrics

        win_rate = (self.performance_metrics['winning_trades'] /
                   self.performance_metrics['total_trades'])

        return {
            **self.performance_metrics,
            'win_rate': win_rate,
            'profit_factor': (self.performance_metrics['winning_trades'] /
                            max(self.performance_metrics['losing_trades'], 1))
        }


class RiskManager:
    """
    Risk management system for trading operations.

    Provides comprehensive risk assessment and position sizing.
    """

    def __init__(self, config):
        """
        Initialize risk manager.

        Args:
            config: Trading configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Risk parameters
        self.max_position_size = config.trading.max_position_size
        self.max_daily_loss = config.trading.max_daily_loss
        self.stop_loss_pct = config.trading.stop_loss_pct
        self.take_profit_pct = config.trading.take_profit_pct

    def evaluate_trade(self, prediction: Dict, market_data: pd.DataFrame,
                      current_positions: Dict) -> Dict[str, Any]:
        """
        Evaluate trade based on risk management rules.

        Args:
            prediction: Model prediction
            market_data: Current market data
            current_positions: Current positions

        Returns:
            Trade evaluation dictionary
        """
        signal = prediction['trading_signal']
        confidence = prediction['confidence']
        risk_prob = prediction['risk_probability']

        # Check confidence threshold
        if confidence < self.config.trading.confidence_threshold:
            return {
                'action': 'hold',
                'confidence': confidence,
                'reason': 'confidence_below_threshold'
            }

        # Check risk probability
        if risk_prob > 0.7:  # High risk
            return {
                'action': 'hold',
                'confidence': confidence,
                'reason': 'high_risk_probability'
            }

        # Check position limits
        total_exposure = sum(pos['market_value'] for pos in current_positions.values())
        max_exposure = 100000 * self.max_position_size  # Assuming $100k account

        if total_exposure >= max_exposure:
            return {
                'action': 'hold',
                'confidence': confidence,
                'reason': 'position_limit_reached'
            }

        # Calculate position size
        position_size = self._calculate_safe_position_size(
            prediction, market_data, total_exposure
        )

        return {
            'action': signal,
            'confidence': confidence,
            'position_size': position_size,
            'stop_loss': self._calculate_stop_loss(market_data),
            'take_profit': self._calculate_take_profit(market_data),
            'risk_amount': position_size * market_data['Close'].iloc[-1] * self.stop_loss_pct
        }

    def _calculate_safe_position_size(self, prediction: Dict,
                                    market_data: pd.DataFrame,
                                    current_exposure: float) -> float:
        """
        Calculate safe position size based on risk parameters.

        Args:
            prediction: Model prediction
            market_data: Market data
            current_exposure: Current portfolio exposure

        Returns:
            Safe position size
        """
        current_price = market_data['Close'].iloc[-1]
        risk_per_trade = self.config.trading.risk_per_trade

        # Base position size on Kelly criterion
        win_probability = prediction['confidence']
        win_loss_ratio = self.take_profit_pct / self.stop_loss_pct

        kelly_fraction = win_probability - ((1 - win_probability) / win_loss_ratio)
        kelly_position = kelly_fraction * self.config.trading.kelly_fraction

        # Adjust for current exposure
        available_capital = 100000 - current_exposure  # Assuming $100k account
        max_position_value = available_capital * self.max_position_size

        position_value = max_position_value * kelly_position
        position_size = position_value / current_price

        return position_size

    def _calculate_stop_loss(self, market_data: pd.DataFrame) -> float:
        """Calculate stop loss price."""
        current_price = market_data['Close'].iloc[-1]
        return current_price * (1 - self.stop_loss_pct)

    def _calculate_take_profit(self, market_data: pd.DataFrame) -> float:
        """Calculate take profit price."""
        current_price = market_data['Close'].iloc[-1]
        return current_price * (1 + self.take_profit_pct)


class LiveTradingEngine:
    """
    Live trading engine with real-time data streaming and execution.

    Provides continuous trading capabilities with market data streaming.
    """

    def __init__(self, trading_agent: TradingAgent, config):
        """
        Initialize live trading engine.

        Args:
            trading_agent: Trading agent instance
            config: Trading configuration
        """
        self.trading_agent = trading_agent
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Trading state
        self.is_running = False
        self.stream = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Market data buffer
        self.market_data_buffer = {}

    async def start_trading(self, symbols: List[str]):
        """
        Start live trading session.

        Args:
            symbols: List of symbols to trade
        """
        self.is_running = True
        self.logger.info(f"Starting live trading for symbols: {symbols}")

        # Initialize market data streaming
        await self._initialize_streaming(symbols)

        # Start trading loop
        await self._trading_loop(symbols)

    async def stop_trading(self):
        """Stop live trading session."""
        self.is_running = False
        self.logger.info("Stopping live trading")

        if self.stream:
            await self.stream.stop_ws()

    async def _initialize_streaming(self, symbols: List[str]):
        """Initialize real-time data streaming."""
        try:
            self.stream = Stream(
                key_id=self.config.api.alpaca_api_key,
                secret_key=self.config.api.alpaca_secret_key,
                base_url=self.config.api.alpaca_base_url
            )

            # Subscribe to market data
            for symbol in symbols:
                self.stream.subscribe_bars(self._on_bar_update, symbol)
                self.stream.subscribe_trades(self._on_trade_update, symbol)

            self.logger.info("Market data streaming initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize streaming: {e}")

    async def _trading_loop(self, symbols: List[str]):
        """Main trading loop."""
        while self.is_running:
            try:
                # Get latest market data for each symbol
                for symbol in symbols:
                    if symbol in self.market_data_buffer:
                        market_data = self.market_data_buffer[symbol]

                        # Make trading decision
                        decision = self.trading_agent.make_decision(market_data)

                        # Execute trade if needed
                        if decision['action'] != 'hold':
                            await asyncio.get_event_loop().run_in_executor(
                                self.executor,
                                self.trading_agent.execute_trade,
                                decision
                            )

                # Wait before next iteration
                await asyncio.sleep(60)  # Trade every minute

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)

    def _on_bar_update(self, bar):
        """Handle bar (OHLCV) updates."""
        symbol = bar.symbol

        if symbol not in self.market_data_buffer:
            self.market_data_buffer[symbol] = pd.DataFrame()

        # Update market data buffer
        new_data = pd.DataFrame([{
            'timestamp': bar.timestamp,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume
        }])

        self.market_data_buffer[symbol] = pd.concat(
            [self.market_data_buffer[symbol], new_data],
            ignore_index=True
        ).tail(1000)  # Keep last 1000 bars

    def _on_trade_update(self, trade):
        """Handle trade updates."""
        # Process trade updates for high-frequency signals
        symbol = trade.symbol
        self.logger.debug(f"Trade update: {symbol} @ {trade.price}")


# Utility functions
def create_trading_environment(data: pd.DataFrame, config) -> TradingEnvironment:
    """
    Create trading environment for RL training.

    Args:
        data: Historical market data
        config: Trading configuration

    Returns:
        Configured trading environment
    """
    return TradingEnvironment(data, config)


def create_trading_agent(model: NCATradingModel, config) -> TradingAgent:
    """
    Create trading agent with NCA model.

    Args:
        model: NCA trading model
        config: Trading configuration

    Returns:
        Configured trading agent
    """
    return TradingAgent(model, config)


if __name__ == "__main__":
    # Example usage
    from .config import ConfigManager

    print("NCA Trading Bot - Trading Module Demo")
    print("=" * 40)

    config = ConfigManager()

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'Open': np.random.randn(1000).cumsum() + 100,
        'High': np.random.randn(1000).cumsum() + 101,
        'Low': np.random.randn(1000).cumsum() + 99,
        'Close': np.random.randn(1000).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 1000)
    })

    # Create trading environment
    env = create_trading_environment(sample_data, config)
    print(f"Trading environment created with {env.observation_space.shape} observation space")

    # Test environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial portfolio value: ${info['portfolio_value']}")

    # Take sample action
    action = np.array([1, 5])  # Buy with 50% position size
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"After action - Portfolio value: ${info['portfolio_value']:.2f}, Reward: {reward:.4f}")

    print("Trading module demo completed successfully!")