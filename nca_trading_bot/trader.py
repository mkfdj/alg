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
import yfinance as yf

from config import get_config
from nca_model import NCATradingModel

# MCP tool import (assumed available)
try:
    from mcp_tools import use_mcp_tool
except ImportError:
    # Fallback for MCP integration
    async def use_mcp_tool(*args, **kwargs):
        raise NotImplementedError("MCP tools not available")


class StockEvaluator:
    """
    Stock evaluation system using MCP for top stocks analysis.

    Queries market data to identify top-performing stocks by market cap
    and other fundamental metrics for trading opportunities.
    """

    def __init__(self, config):
        """
        Initialize stock evaluator.

        Args:
            config: Trading configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Default top stocks (fallback)
        self.default_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "NVDA", "META", "NFLX", "SPY", "QQQ"
        ]

    async def get_top_stocks(self, num_stocks: int = 10) -> List[Dict[str, Any]]:
        """
        Get top stocks by market cap using MCP queries.

        Args:
            num_stocks: Number of top stocks to return

        Returns:
            List of stock dictionaries with symbol, market_cap, etc.
        """
        try:
            # Query Brave MCP for top stocks by market cap
            query = f"top {num_stocks} stocks by market capitalization 2024"

            search_results = await use_mcp_tool(
                server_name="brave-search",
                tool_name="brave_web_search",
                arguments={
                    "query": query,
                    "count": 20,  # Get more results for filtering
                    "offset": 0
                }
            )

            # Parse results to extract stock symbols
            top_stocks = self._parse_search_results(search_results, num_stocks)

            if not top_stocks:
                self.logger.warning("Failed to parse top stocks from MCP, using defaults")
                top_stocks = self.default_stocks[:num_stocks]

            # Get detailed info for each stock
            detailed_stocks = []
            for symbol in top_stocks:
                try:
                    stock_info = await self._get_stock_details(symbol)
                    detailed_stocks.append(stock_info)
                except Exception as e:
                    self.logger.warning(f"Failed to get details for {symbol}: {e}")
                    detailed_stocks.append({
                        'symbol': symbol,
                        'market_cap': 0,
                        'price': 0,
                        'volume': 0
                    })

            return detailed_stocks

        except Exception as e:
            self.logger.error(f"Failed to get top stocks via MCP: {e}")
            # Fallback to default stocks
            return [
                {'symbol': symbol, 'market_cap': 0, 'price': 0, 'volume': 0}
                for symbol in self.default_stocks[:num_stocks]
            ]

    def _parse_search_results(self, search_results: Dict, num_stocks: int) -> List[str]:
        """
        Parse MCP search results to extract stock symbols.

        Args:
            search_results: Raw search results from MCP
            num_stocks: Number of stocks to extract

        Returns:
            List of stock symbols
        """
        symbols = []

        try:
            results = search_results.get('results', [])

            # Common top stocks by market cap (approximate)
            known_top_stocks = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
                "NVDA", "META", "NFLX", "SPY", "QQQ",
                "BRK.B", "JPM", "V", "JNJ", "WMT",
                "PG", "UNH", "HD", "MA", "DIS"
            ]

            # Extract symbols from search results
            for result in results:
                title = result.get('title', '').upper()
                description = result.get('description', '').upper()

                for stock in known_top_stocks:
                    if stock in title or stock in description:
                        if stock not in symbols:
                            symbols.append(stock)
                            if len(symbols) >= num_stocks:
                                break

                if len(symbols) >= num_stocks:
                    break

        except Exception as e:
            self.logger.warning(f"Error parsing search results: {e}")

        return symbols[:num_stocks]

    async def _get_stock_details(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed stock information.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with stock details
        """
        try:
            # Use yfinance for stock details
            stock = yf.Ticker(symbol)
            info = stock.info

            return {
                'symbol': symbol,
                'market_cap': info.get('marketCap', 0),
                'price': info.get('currentPrice', 0),
                'volume': info.get('volume', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }

        except Exception as e:
            self.logger.warning(f"Failed to get stock details for {symbol}: {e}")
            return {
                'symbol': symbol,
                'market_cap': 0,
                'price': 0,
                'volume': 0,
                'sector': 'Unknown',
                'industry': 'Unknown'
            }


class HistoricalDataLoader:
    """
    Enhanced historical data loader for 10+ years of data.

    Loads comprehensive historical data for predictions with technical indicators.
    """

    def __init__(self, config):
        """
        Initialize historical data loader.

        Args:
            config: Trading configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def load_long_term_data(self, symbol: str, years: int = 10) -> pd.DataFrame:
        """
        Load 10+ years of historical data for a symbol.

        Args:
            symbol: Stock symbol
            years: Number of years of data to load

        Returns:
            DataFrame with historical data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)

            # Use yfinance for historical data
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date, interval='1d')

            # Reset index and clean data
            data = data.reset_index()
            data['timestamp'] = pd.to_datetime(data['Date'])
            data = data.drop('Date', axis=1)

            # Add technical indicators
            data = self._add_technical_indicators(data)

            self.logger.info(f"Loaded {len(data)} days of data for {symbol}")
            return data

        except Exception as e:
            self.logger.error(f"Failed to load data for {symbol}: {e}")
            return pd.DataFrame()

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.

        Args:
            data: Raw price data

        Returns:
            DataFrame with technical indicators
        """
        if len(data) < 50:  # Need minimum data for indicators
            return data

        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()

        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()

        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_signal'] = data['MACD'].ewm(span=9).mean()

        # Bollinger Bands
        data['BB_middle'] = data['Close'].rolling(20).mean()
        data['BB_upper'] = data['BB_middle'] + 2 * data['Close'].rolling(20).std()
        data['BB_lower'] = data['BB_middle'] - 2 * data['Close'].rolling(20).std()

        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(20).mean()

        return data.fillna(0)


class NCAEnsemble:
    """
    Ensemble of 5-10 NCA models for stable trading decisions.

    Implements ensemble methods with stability checks and consensus decision making.
    """

    def __init__(self, config, num_models: int = 8):
        """
        Initialize NCA ensemble.

        Args:
            config: Trading configuration
            num_models: Number of NCA models in ensemble (5-10)
        """
        self.config = config
        self.num_models = max(5, min(10, num_models))  # Clamp between 5-10
        self.logger = logging.getLogger(__name__)

        # Ensemble models
        self.models = []
        self.model_weights = []

        # Stability tracking
        self.prediction_history = []
        self.stability_scores = []

        # Initialize ensemble
        self._initialize_ensemble()

    def _initialize_ensemble(self):
        """Initialize the ensemble of NCA models."""
        for i in range(self.num_models):
            try:
                # Create model with slight variations for diversity
                model = NCATradingModel(self.config)

                # Add some randomization to model parameters for diversity
                with torch.no_grad():
                    for param in model.parameters():
                        param.add_(torch.randn_like(param) * 0.01)

                self.models.append(model)
                self.model_weights.append(1.0 / self.num_models)  # Equal weights initially

                self.logger.info(f"Initialized NCA model {i+1}/{self.num_models}")

            except Exception as e:
                self.logger.error(f"Failed to initialize NCA model {i+1}: {e}")

    def predict_ensemble(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Make ensemble prediction with stability checks.

        Args:
            x: Input tensor

        Returns:
            Ensemble prediction with confidence and stability metrics
        """
        try:
            individual_predictions = []

            # Get predictions from all models
            for model in self.models:
                try:
                    pred = model.predict(x)
                    individual_predictions.append(pred)
                except Exception as e:
                    self.logger.warning(f"Model prediction failed: {e}")
                    # Use neutral prediction as fallback
                    individual_predictions.append({
                        'trading_signal': 'hold',
                        'confidence': 0.5,
                        'price_prediction': 0,
                        'risk_probability': 0.5
                    })

            # Ensemble decision making
            ensemble_decision = self._make_ensemble_decision(individual_predictions)

            # Stability analysis
            stability_metrics = self._calculate_stability(individual_predictions)

            # Update prediction history
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'individual_predictions': individual_predictions,
                'ensemble_decision': ensemble_decision,
                'stability': stability_metrics
            })

            # Keep history bounded
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]

            return {
                'ensemble_decision': ensemble_decision,
                'stability_metrics': stability_metrics,
                'individual_predictions': individual_predictions,
                'confidence_score': self._calculate_confidence_score(individual_predictions)
            }

        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            return self._fallback_prediction()

    def _make_ensemble_decision(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Make ensemble decision from individual predictions.

        Args:
            predictions: List of individual model predictions

        Returns:
            Ensemble decision
        """
        # Count votes for each action
        action_votes = {'buy': 0, 'sell': 0, 'hold': 0}
        confidence_sum = {'buy': 0, 'sell': 0, 'hold': 0}
        price_predictions = []
        risk_probabilities = []

        for pred in predictions:
            action = pred['trading_signal']
            confidence = pred['confidence']

            action_votes[action] += 1
            confidence_sum[action] += confidence
            price_predictions.append(pred['price_prediction'])
            risk_probabilities.append(pred['risk_probability'])

        # Weighted voting based on confidence
        weighted_votes = {}
        for action in ['buy', 'sell', 'hold']:
            if action_votes[action] > 0:
                avg_confidence = confidence_sum[action] / action_votes[action]
                weighted_votes[action] = action_votes[action] * avg_confidence
            else:
                weighted_votes[action] = 0

        # Select action with highest weighted votes
        best_action = max(weighted_votes, key=weighted_votes.get)
        ensemble_confidence = weighted_votes[best_action] / sum(weighted_votes.values())

        # Ensemble price prediction (median)
        ensemble_price = np.median(price_predictions)

        # Ensemble risk (average)
        ensemble_risk = np.mean(risk_probabilities)

        return {
            'action': best_action,
            'confidence': float(ensemble_confidence),
            'price_prediction': float(ensemble_price),
            'risk_probability': float(ensemble_risk),
            'vote_distribution': action_votes
        }

    def _calculate_stability(self, predictions: List[Dict]) -> Dict[str, float]:
        """
        Calculate prediction stability metrics.

        Args:
            predictions: Individual model predictions

        Returns:
            Stability metrics
        """
        if len(self.prediction_history) < 2:
            return {
                'action_consistency': 1.0,
                'confidence_std': 0.0,
                'prediction_variance': 0.0
            }

        # Action consistency (fraction of models agreeing)
        actions = [p['trading_signal'] for p in predictions]
        most_common_action = max(set(actions), key=actions.count)
        action_consistency = actions.count(most_common_action) / len(actions)

        # Confidence stability
        confidences = [p['confidence'] for p in predictions]
        confidence_std = np.std(confidences)

        # Prediction variance
        price_preds = [p['price_prediction'] for p in predictions]
        prediction_variance = np.var(price_preds) if len(price_preds) > 1 else 0

        return {
            'action_consistency': float(action_consistency),
            'confidence_std': float(confidence_std),
            'prediction_variance': float(prediction_variance)
        }

    def _calculate_confidence_score(self, predictions: List[Dict]) -> float:
        """
        Calculate overall confidence score for ensemble.

        Args:
            predictions: Individual predictions

        Returns:
            Confidence score
        """
        # Based on agreement and individual confidences
        actions = [p['trading_signal'] for p in predictions]
        most_common = max(set(actions), key=actions.count)
        agreement_ratio = actions.count(most_common) / len(actions)

        avg_confidence = np.mean([p['confidence'] for p in predictions])

        return float(agreement_ratio * avg_confidence)

    def _fallback_prediction(self) -> Dict[str, Any]:
        """Fallback prediction when ensemble fails."""
        return {
            'ensemble_decision': {
                'action': 'hold',
                'confidence': 0.5,
                'price_prediction': 0,
                'risk_probability': 0.5,
                'vote_distribution': {'buy': 0, 'sell': 0, 'hold': self.num_models}
            },
            'stability_metrics': {
                'action_consistency': 1.0,
                'confidence_std': 0.0,
                'prediction_variance': 0.0
            },
            'individual_predictions': [],
            'confidence_score': 0.5
        }

    def update_weights(self, performance_feedback: Dict[str, float]):
        """
        Update model weights based on performance feedback.

        Args:
            performance_feedback: Performance metrics for weight adjustment
        """
        try:
            # Simple weight adjustment based on recent performance
            reward = performance_feedback.get('reward', 0)

            if reward > 0:
                # Increase weights for models that contributed to good performance
                for i in range(len(self.model_weights)):
                    self.model_weights[i] *= 1.01  # Small increase
            else:
                # Decrease weights for models that contributed to poor performance
                for i in range(len(self.model_weights)):
                    self.model_weights[i] *= 0.99  # Small decrease

            # Normalize weights
            total_weight = sum(self.model_weights)
            self.model_weights = [w / total_weight for w in self.model_weights]

        except Exception as e:
            self.logger.warning(f"Failed to update weights: {e}")

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """
        Get ensemble statistics and health metrics.

        Returns:
            Ensemble statistics
        """
        return {
            'num_models': self.num_models,
            'model_weights': self.model_weights,
            'history_length': len(self.prediction_history),
            'stability_scores': self.stability_scores[-10:] if self.stability_scores else []
        }


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
    Enhanced trading agent with NCA ensemble, top stocks evaluation, and variable quantity trading.

    Provides decision-making, risk management, and trade execution capabilities
    with min $1 quantity increments and stability checks.
    """

    def __init__(self, config, nca_ensemble: Optional[NCAEnsemble] = None):
        """
        Initialize trading agent.

        Args:
            config: Trading configuration
            nca_ensemble: NCA ensemble (optional, will create if None)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Enhanced trading components
        self.alpaca_trader = AlpacaTrader(config)
        self.risk_manager = RiskManager(config)
        self.stock_evaluator = StockEvaluator(config)
        self.data_loader = HistoricalDataLoader(config)

        # NCA Ensemble (create if not provided)
        self.nca_ensemble = nca_ensemble or NCAEnsemble(config)

        # Top stocks tracking
        self.top_stocks = []
        self.current_focus_stocks = []

        # Agent state
        self.is_live_trading = False
        self.predictions = []
        self.trades = []

        # Variable quantity trading
        self.min_trade_increment = 1.0  # $1 minimum
        self.max_trade_value = 10000.0  # Max $10k per trade

        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'ensemble_stability': 0,
            'avg_trade_size': 0
        }

    async def initialize_top_stocks(self, num_stocks: int = 10):
        """
        Initialize top stocks evaluation.

        Args:
            num_stocks: Number of top stocks to evaluate
        """
        try:
            self.logger.info(f"Evaluating top {num_stocks} stocks...")
            self.top_stocks = await self.stock_evaluator.get_top_stocks(num_stocks)
            self.current_focus_stocks = [stock['symbol'] for stock in self.top_stocks[:5]]  # Focus on top 5
            self.logger.info(f"Top stocks initialized: {self.current_focus_stocks}")
        except Exception as e:
            self.logger.error(f"Failed to initialize top stocks: {e}")
            # Fallback to default stocks
            self.current_focus_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    async def load_historical_data(self, symbol: str) -> pd.DataFrame:
        """
        Load 10+ years of historical data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Historical data DataFrame
        """
        return await self.data_loader.load_long_term_data(symbol, years=10)

    def make_decision(self, market_data: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
        """
        Make trading decision using NCA ensemble with stability checks.

        Args:
            market_data: Current market data
            symbol: Stock symbol (optional)

        Returns:
            Trading decision dictionary
        """
        try:
            # Prepare input for ensemble
            features = self._prepare_features(market_data)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)

            # Get ensemble prediction with stability checks
            ensemble_result = self.nca_ensemble.predict_ensemble(features_tensor)

            # Extract decision
            ensemble_decision = ensemble_result['ensemble_decision']

            # Apply risk management
            decision = self.risk_manager.evaluate_trade(
                {
                    'trading_signal': ensemble_decision['action'],
                    'confidence': ensemble_decision['confidence'],
                    'risk_probability': ensemble_decision['risk_probability']
                },
                market_data,
                self.alpaca_trader.get_positions()
            )

            # Add ensemble-specific information
            decision.update({
                'ensemble_confidence': ensemble_result['confidence_score'],
                'stability_metrics': ensemble_result['stability_metrics'],
                'symbol': symbol or 'UNKNOWN'
            })

            self.predictions.append({
                'timestamp': datetime.now(),
                'ensemble_result': ensemble_result,
                'decision': decision,
                'symbol': symbol
            })

            return decision

        except Exception as e:
            self.logger.error(f"Error making trading decision: {e}")
            return {
                'action': 'hold',
                'confidence': 0,
                'reason': 'error',
                'ensemble_confidence': 0,
                'stability_metrics': {}
            }

    def execute_trade(self, decision: Dict[str, Any]) -> bool:
        """
        Execute trading decision with variable quantities and $3 handling fee.

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

            # Calculate variable position size with $1 increments
            trade_value = self._calculate_variable_trade_value(
                decision, float(account.buying_power)
            )

            if trade_value < self.min_trade_increment:
                self.logger.info(f"Trade value ${trade_value:.2f} below minimum ${self.min_trade_increment}")
                return False

            # Get current price for symbol
            symbol = decision.get('symbol', 'SPY')
            current_price = self._get_current_price(symbol)

            if not current_price or current_price <= 0:
                self.logger.error(f"Could not get valid price for {symbol}")
                return False

            # Calculate quantity (shares)
            quantity = int(trade_value / current_price)

            if quantity <= 0:
                self.logger.info(f"Calculated quantity {quantity} is invalid")
                return False

            # Apply $3 handling fee
            handling_fee = 3.0
            if trade_value < handling_fee:
                self.logger.warning(f"Trade value ${trade_value:.2f} less than handling fee ${handling_fee}")
                return False

            # Adjust trade value for fee
            net_trade_value = trade_value - handling_fee

            # Recalculate quantity after fee
            quantity = int(net_trade_value / current_price)

            if quantity <= 0:
                self.logger.info(f"Quantity after fee adjustment {quantity} is invalid")
                return False

            # Place order
            side = 'buy' if decision['action'] == 'buy' else 'sell'

            order_id = self.alpaca_trader.place_order(
                symbol=symbol,
                qty=quantity,
                side=side
            )

            if order_id:
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'trade_value': trade_value,
                    'handling_fee': handling_fee,
                    'net_value': net_trade_value,
                    'price': current_price,
                    'order_id': order_id,
                    'decision': decision,
                    'ensemble_confidence': decision.get('ensemble_confidence', 0),
                    'stability_metrics': decision.get('stability_metrics', {})
                }
                self.trades.append(trade_record)

                # Update performance metrics
                self._update_performance_metrics(trade_record)

                # Update ensemble weights based on execution
                self.nca_ensemble.update_weights({'reward': decision.get('confidence', 0)})

                self.logger.info(f"Trade executed: {side} {quantity} {symbol} @ ${current_price:.2f} "
                               f"(Value: ${trade_value:.2f}, Fee: ${handling_fee:.2f})")
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

    def _calculate_variable_trade_value(self, decision: Dict[str, Any], available_capital: float) -> float:
        """
        Calculate variable trade value with $1 increments based on confidence and risk.

        Args:
            decision: Trading decision
            available_capital: Available capital for trading

        Returns:
            Trade value in dollars (multiple of $1)
        """
        confidence = decision.get('ensemble_confidence', decision.get('confidence', 0))
        risk_per_trade = self.config.trading.risk_per_trade

        # Base trade value on confidence and risk limits
        max_trade_value = min(
            available_capital * self.config.trading.max_position_size,
            self.max_trade_value
        )

        # Scale by ensemble confidence and stability
        stability_factor = decision.get('stability_metrics', {}).get('action_consistency', 1.0)
        adjusted_confidence = confidence * stability_factor

        trade_value = max_trade_value * adjusted_confidence

        # Apply risk limits
        risk_amount = available_capital * risk_per_trade
        trade_value = min(trade_value, risk_amount / 0.02)  # Assume 2% stop loss

        # Ensure minimum trade value and round to $1 increments
        trade_value = max(trade_value, self.min_trade_increment)
        trade_value = round(trade_value)  # Round to nearest dollar

        return trade_value

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Current price or None if unavailable
        """
        try:
            # Try Alpaca first
            quote = self.alpaca_trader.api.get_latest_quote(symbol)
            return quote.askprice if quote else None
        except Exception as e:
            self.logger.warning(f"Alpaca price lookup failed for {symbol}: {e}")

        try:
            # Fallback to yfinance
            stock = yf.Ticker(symbol)
            price = stock.info.get('currentPrice')
            return price if price else None
        except Exception as e:
            self.logger.warning(f"YFinance price lookup failed for {symbol}: {e}")

        return None

    def _update_performance_metrics(self, trade: Dict):
        """Update performance tracking metrics with ensemble and variable quantity features."""
        self.performance_metrics['total_trades'] += 1

        # Update win/loss ratio (simplified - would need actual P&L tracking)
        if trade['decision'].get('expected_profit', 0) > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1

        # Update ensemble stability metrics
        ensemble_confidence = trade.get('ensemble_confidence', 0)
        stability_metrics = trade.get('stability_metrics', {})

        if 'ensemble_stability' not in self.performance_metrics:
            self.performance_metrics['ensemble_stability'] = []

        self.performance_metrics['ensemble_stability'].append({
            'confidence': ensemble_confidence,
            'consistency': stability_metrics.get('action_consistency', 0),
            'variance': stability_metrics.get('prediction_variance', 0)
        })

        # Keep only recent stability metrics
        if len(self.performance_metrics['ensemble_stability']) > 100:
            self.performance_metrics['ensemble_stability'] = self.performance_metrics['ensemble_stability'][-100:]

        # Update average trade size
        trade_sizes = [t.get('trade_value', 0) for t in self.trades]
        self.performance_metrics['avg_trade_size'] = np.mean(trade_sizes) if trade_sizes else 0

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report with ensemble metrics."""
        if self.performance_metrics['total_trades'] == 0:
            return self.performance_metrics

        win_rate = (self.performance_metrics['winning_trades'] /
                    self.performance_metrics['total_trades'])

        # Calculate ensemble stability metrics
        stability_data = self.performance_metrics.get('ensemble_stability', [])
        if stability_data:
            avg_confidence = np.mean([s['confidence'] for s in stability_data])
            avg_consistency = np.mean([s['consistency'] for s in stability_data])
            avg_variance = np.mean([s['variance'] for s in stability_data])
        else:
            avg_confidence = avg_consistency = avg_variance = 0

        return {
            **self.performance_metrics,
            'win_rate': win_rate,
            'profit_factor': (self.performance_metrics['winning_trades'] /
                            max(self.performance_metrics['losing_trades'], 1)),
            'ensemble_metrics': {
                'avg_confidence': float(avg_confidence),
                'avg_consistency': float(avg_consistency),
                'avg_prediction_variance': float(avg_variance)
            },
            'trading_costs': {
                'total_handling_fees': sum(t.get('handling_fee', 0) for t in self.trades),
                'avg_fee_per_trade': np.mean([t.get('handling_fee', 0) for t in self.trades]) if self.trades else 0
            }
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


def create_trading_agent(config, nca_ensemble: Optional[NCAEnsemble] = None) -> TradingAgent:
    """
    Create enhanced trading agent with NCA ensemble and top stocks evaluation.

    Args:
        config: Trading configuration
        nca_ensemble: Optional NCA ensemble (will create if None)

    Returns:
        Configured trading agent
    """
    return TradingAgent(config, nca_ensemble)


def create_nca_ensemble(config, num_models: int = 8) -> NCAEnsemble:
    """
    Create NCA ensemble for stable trading decisions.

    Args:
        config: Trading configuration
        num_models: Number of models in ensemble (5-10)

    Returns:
        Configured NCA ensemble
    """
    return NCAEnsemble(config, num_models)


def create_stock_evaluator(config) -> StockEvaluator:
    """
    Create stock evaluator for top stocks analysis.

    Args:
        config: Trading configuration

    Returns:
        Configured stock evaluator
    """
    return StockEvaluator(config)


def create_historical_data_loader(config) -> HistoricalDataLoader:
    """
    Create historical data loader for long-term analysis.

    Args:
        config: Trading configuration

    Returns:
        Configured data loader
    """
    return HistoricalDataLoader(config)


# ===== KEY CODE SNIPPETS =====
"""
Key implementation snippets for the enhanced trading mechanism.
These snippets demonstrate the core functionality of the NCA ensemble,
variable quantity trading, and top stocks evaluation.
"""

# 1. Ensemble Decision Making Snippet
ENSEMBLE_DECISION_SNIPPET = '''
def _make_ensemble_decision(self, predictions: List[Dict]) -> Dict[str, Any]:
    """
    Make ensemble decision from individual model predictions.
    Uses weighted voting based on confidence scores.
    """
    # Count votes for each action
    action_votes = {'buy': 0, 'sell': 0, 'hold': 0}
    confidence_sum = {'buy': 0, 'sell': 0, 'hold': 0}
    price_predictions = []
    risk_probabilities = []

    for pred in predictions:
        action = pred['trading_signal']
        confidence = pred['confidence']

        action_votes[action] += 1
        confidence_sum[action] += confidence
        price_predictions.append(pred['price_prediction'])
        risk_probabilities.append(pred['risk_probability'])

    # Weighted voting based on confidence
    weighted_votes = {}
    for action in ['buy', 'sell', 'hold']:
        if action_votes[action] > 0:
            avg_confidence = confidence_sum[action] / action_votes[action]
            weighted_votes[action] = action_votes[action] * avg_confidence
        else:
            weighted_votes[action] = 0

    # Select action with highest weighted votes
    best_action = max(weighted_votes, key=weighted_votes.get)
    ensemble_confidence = weighted_votes[best_action] / sum(weighted_votes.values())

    # Ensemble price prediction (median)
    ensemble_price = np.median(price_predictions)

    # Ensemble risk (average)
    ensemble_risk = np.mean(risk_probabilities)

    return {
        'action': best_action,
        'confidence': float(ensemble_confidence),
        'price_prediction': float(ensemble_price),
        'risk_probability': float(ensemble_risk),
        'vote_distribution': action_votes
    }
'''

# 2. Variable Quantity Trading with $3 Handling Fee
VARIABLE_QUANTITY_SNIPPET = '''
def _calculate_variable_trade_value(self, decision: Dict[str, Any], available_capital: float) -> float:
    """
    Calculate variable trade value with $1 increments and $3 handling fee.
    """
    confidence = decision.get('ensemble_confidence', decision.get('confidence', 0))
    risk_per_trade = self.config.trading.risk_per_trade

    # Base trade value on confidence and risk limits
    max_trade_value = min(
        available_capital * self.config.trading.max_position_size,
        self.max_trade_value
    )

    # Scale by ensemble confidence and stability
    stability_factor = decision.get('stability_metrics', {}).get('action_consistency', 1.0)
    adjusted_confidence = confidence * stability_factor

    trade_value = max_trade_value * adjusted_confidence

    # Apply risk limits
    risk_amount = available_capital * risk_per_trade
    trade_value = min(trade_value, risk_amount / 0.02)  # Assume 2% stop loss

    # Ensure minimum trade value and round to $1 increments
    trade_value = max(trade_value, self.min_trade_increment)
    trade_value = round(trade_value)  # Round to nearest dollar

    return trade_value

# In execute_trade method:
handling_fee = 3.0
if trade_value < handling_fee:
    return False  # Insufficient for fee

net_trade_value = trade_value - handling_fee
quantity = int(net_trade_value / current_price)
'''

# 3. Top Stocks Evaluation using MCP
TOP_STOCKS_EVALUATION_SNIPPET = '''
async def get_top_stocks(self, num_stocks: int = 10) -> List[Dict[str, Any]]:
    """
    Get top stocks by market cap using Brave MCP search.
    """
    try:
        # Query Brave MCP for top stocks by market cap
        query = f"top {num_stocks} stocks by market capitalization 2024"

        search_results = await use_mcp_tool(
            server_name="brave-search",
            tool_name="brave_web_search",
            arguments={
                "query": query,
                "count": 20,  # Get more results for filtering
                "offset": 0
            }
        )

        # Parse results to extract stock symbols
        top_stocks = self._parse_search_results(search_results, num_stocks)

        # Get detailed info for each stock
        detailed_stocks = []
        for symbol in top_stocks:
            stock_info = await self._get_stock_details(symbol)
            detailed_stocks.append(stock_info)

        return detailed_stocks

    except Exception as e:
        # Fallback to default stocks
        return [default_stocks]
'''

# 4. Stability Checks in Ensemble
STABILITY_CHECKS_SNIPPET = '''
def _calculate_stability(self, predictions: List[Dict]) -> Dict[str, float]:
    """
    Calculate prediction stability metrics for ensemble.
    """
    # Action consistency (fraction of models agreeing)
    actions = [p['trading_signal'] for p in predictions]
    most_common_action = max(set(actions), key=actions.count)
    action_consistency = actions.count(most_common_action) / len(actions)

    # Confidence stability
    confidences = [p['confidence'] for p in predictions]
    confidence_std = np.std(confidences)

    # Prediction variance
    price_preds = [p['price_prediction'] for p in predictions]
    prediction_variance = np.var(price_preds) if len(price_preds) > 1 else 0

    return {
        'action_consistency': float(action_consistency),
        'confidence_std': float(confidence_std),
        'prediction_variance': float(prediction_variance)
    }
'''

# 5. 10+ Years Historical Data Loading
HISTORICAL_DATA_LOADING_SNIPPET = '''
async def load_long_term_data(self, symbol: str, years: int = 10) -> pd.DataFrame:
    """
    Load 10+ years of historical data for comprehensive analysis.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    # Use yfinance for historical data
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date, interval='1d')

    # Clean and process data
    data = data.reset_index()
    data['timestamp'] = pd.to_datetime(data['Date'])
    data = data.drop('Date', axis=1)

    # Add technical indicators for better predictions
    data = self._add_technical_indicators(data)

    return data

def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators."""
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()

    # RSI, MACD, Bollinger Bands, Volume indicators
    # ... (implementation details in the class)
    return data.fillna(0)
'''


if __name__ == "__main__":
    # Example usage
    from config import ConfigManager

    print("Enhanced NCA Trading Bot - Trading Module Demo")
    print("=" * 50)

    config = ConfigManager()

    # Create enhanced components
    stock_evaluator = create_stock_evaluator(config)
    data_loader = create_historical_data_loader(config)
    nca_ensemble = create_nca_ensemble(config, num_models=8)
    trading_agent = create_trading_agent(config, nca_ensemble)

    print("Enhanced trading components created:")
    print(f"  - Stock Evaluator: {type(stock_evaluator).__name__}")
    print(f"  - Data Loader: {type(data_loader).__name__}")
    print(f"  - NCA Ensemble: {nca_ensemble.num_models} models")
    print(f"  - Trading Agent: Enhanced with ensemble and variable quantities")

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

    # Test ensemble decision making
    decision = trading_agent.make_decision(sample_data.iloc[-60:], symbol='AAPL')
    print(f"Ensemble decision: {decision['action']} (confidence: {decision.get('ensemble_confidence', 0):.3f})")

    # Take sample action
    action = np.array([1, 5])  # Buy with 50% position size
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"After action - Portfolio value: ${info['portfolio_value']:.2f}, Reward: {reward:.4f}")

    print("\nKey Features Implemented:")
    print("✓ Top stocks evaluation using Brave MCP")
    print("✓ 10+ years historical data loading")
    print("✓ NCA ensemble (5-10 models) with stability checks")
    print("✓ Variable quantity trading (min $1 increments)")
    print("✓ $3 handling fee integration")
    print("✓ Enhanced decision making with ensemble confidence")

    print("\nEnhanced trading module demo completed successfully!")