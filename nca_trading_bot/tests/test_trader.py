"""
Unit tests for trader module.

Tests trading environment, Alpaca integration, and risk management functionality.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from ..trader import (
    TradingEnvironment, AlpacaTrader, TradingAgent, RiskManager,
    LiveTradingEngine, create_trading_environment, create_trading_agent
)
from ..config import ConfigManager
from ..nca_model import NCATradingModel


class TestTradingEnvironment(unittest.TestCase):
    """Test cases for TradingEnvironment class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 101,
            'Low': np.random.randn(100).cumsum() + 99,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        })

        self.config = ConfigManager()
        self.env = TradingEnvironment(self.sample_data, self.config)

    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.initial_balance, 100000)
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.balance, 100000)

        # Check action and observation spaces
        self.assertEqual(self.env.action_space.shape, (2,))
        self.assertEqual(len(self.env.observation_space.shape), 2)

    def test_reset(self):
        """Test environment reset."""
        obs, info = self.env.reset()

        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.balance, 100000)
        self.assertEqual(self.env.position, 0)
        self.assertIn('portfolio_value', info)
        self.assertIn('balance', info)
        self.assertIn('position', info)

    def test_step(self):
        """Test environment step."""
        self.env.reset()

        # Test hold action
        action = [0, 0]  # Hold
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertEqual(self.env.current_step, 1)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn('portfolio_value', info)

    def test_buy_action(self):
        """Test buy action execution."""
        self.env.reset()

        # Buy action
        action = [1, 5]  # Buy with 50% position
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check that position is created
        self.assertGreater(self.env.position, 0)
        self.assertLess(self.env.balance, 100000)

        # Check that trade is recorded
        self.assertGreater(len(self.env.trades), 0)
        self.assertEqual(self.env.trades[-1]['type'], 'buy')

    def test_sell_action(self):
        """Test sell action execution."""
        self.env.reset()

        # First buy some shares
        buy_action = [1, 5]
        self.env.step(buy_action)

        # Then sell
        sell_action = [2, 5]  # Sell with 50% position
        obs, reward, terminated, truncated, info = self.env.step(sell_action)

        # Check that position is reduced
        self.assertLess(self.env.position, 10)  # Should be close to 0

        # Check that trade is recorded
        self.assertGreater(len(self.env.trades), 1)
        self.assertEqual(self.env.trades[-1]['type'], 'sell')

    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        self.env.reset()

        initial_value = self.env.portfolio_value

        # Buy some shares
        action = [1, 5]
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Portfolio value should remain relatively stable (small change due to transaction costs)
        self.assertAlmostEqual(self.env.portfolio_value, initial_value, delta=1000)

    def test_termination_conditions(self):
        """Test environment termination conditions."""
        self.env.reset()

        # Move to end of episode
        self.env.current_step = self.env.episode_length - 1
        action = [0, 0]
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertTrue(terminated)

    def test_render(self):
        """Test environment rendering."""
        self.env.reset()

        # Should not raise exception
        self.env.render()

    def test_observation_shape(self):
        """Test observation shape."""
        obs, info = self.env.reset()

        expected_shape = (self.env.window_size, self.sample_data.shape[1] - 1)  # -1 for timestamp
        self.assertEqual(obs.shape, expected_shape)


class TestAlpacaTrader(unittest.TestCase):
    """Test cases for AlpacaTrader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.trader = AlpacaTrader(self.config)

    @patch('alpaca_trade_api.REST')
    def test_initialization(self, mock_rest):
        """Test Alpaca trader initialization."""
        mock_api = Mock()
        mock_rest.return_value = mock_api

        trader = AlpacaTrader(self.config)

        self.assertIsNotNone(trader.api)
        self.assertEqual(trader.max_position_size, self.config.trading.max_position_size)
        self.assertEqual(trader.max_daily_loss, -self.config.trading.max_daily_loss * 100000)

    @patch('alpaca_trade_api.REST')
    def test_get_positions(self, mock_rest):
        """Test getting positions."""
        mock_api = Mock()
        mock_position = Mock()
        mock_position.symbol = 'AAPL'
        mock_position.qty = 100
        mock_position.market_value = 15000.0
        mock_position.avg_entry_price = 150.0
        mock_position.unrealized_pl = 500.0

        mock_api.list_positions.return_value = [mock_position]
        mock_rest.return_value = mock_api

        trader = AlpacaTrader(self.config)
        positions = trader.get_positions()

        self.assertIn('AAPL', positions)
        self.assertEqual(positions['AAPL']['qty'], 100)
        self.assertEqual(positions['AAPL']['market_value'], 15000.0)

    @patch('alpaca_trade_api.REST')
    def test_place_order(self, mock_rest):
        """Test order placement."""
        mock_api = Mock()
        mock_order = Mock()
        mock_order.id = 'test_order_id'
        mock_order.status = 'accepted'

        mock_api.submit_order.return_value = mock_order
        mock_api.get_account.return_value.buying_power = 100000
        mock_rest.return_value = mock_api

        trader = AlpacaTrader(self.config)
        order_id = trader.place_order('AAPL', 10, 'buy')

        self.assertEqual(order_id, 'test_order_id')
        mock_api.submit_order.assert_called_once()

    @patch('alpaca_trade_api.REST')
    def test_risk_limits(self, mock_rest):
        """Test risk limit checking."""
        mock_api = Mock()
        mock_api.get_account.return_value.buying_power = 100000
        mock_rest.return_value = mock_api

        trader = AlpacaTrader(self.config)

        # Test valid trade
        self.assertTrue(trader._check_risk_limits('AAPL', 10, 'buy'))

        # Test invalid trade (exceeds position limit)
        trader.max_position_size = 0  # Set to 0 to trigger limit
        self.assertFalse(trader._check_risk_limits('AAPL', 10, 'buy'))

    @patch('alpaca_trade_api.REST')
    def test_cancel_order(self, mock_rest):
        """Test order cancellation."""
        mock_api = Mock()
        mock_rest.return_value = mock_api

        trader = AlpacaTrader(self.config)
        result = trader.cancel_order('test_order_id')

        mock_api.cancel_order.assert_called_once_with('test_order_id')
        self.assertTrue(result)

    @patch('alpaca_trade_api.REST')
    def test_get_order_status(self, mock_rest):
        """Test order status retrieval."""
        mock_api = Mock()
        mock_order = Mock()
        mock_order.id = 'test_order_id'
        mock_order.symbol = 'AAPL'
        mock_order.qty = 10
        mock_order.filled_qty = 10
        mock_order.status = 'filled'
        mock_order.filled_avg_price = 150.0

        mock_api.get_order.return_value = mock_order
        mock_rest.return_value = mock_api

        trader = AlpacaTrader(self.config)
        status = trader.get_order_status('test_order_id')

        self.assertIsNotNone(status)
        self.assertEqual(status['id'], 'test_order_id')
        self.assertEqual(status['symbol'], 'AAPL')
        self.assertEqual(status['status'], 'filled')


class TestTradingAgent(unittest.TestCase):
    """Test cases for TradingAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.model = NCATradingModel(self.config)
        self.agent = TradingAgent(self.model, self.config)

    def test_initialization(self):
        """Test trading agent initialization."""
        self.assertEqual(self.agent.model, self.model)
        self.assertEqual(self.agent.config, self.config)
        self.assertIsNotNone(self.agent.alpaca_trader)
        self.assertIsNotNone(self.agent.risk_manager)

        # Check performance metrics structure
        expected_keys = ['total_trades', 'winning_trades', 'losing_trades',
                        'total_pnl', 'sharpe_ratio', 'max_drawdown']
        for key in expected_keys:
            self.assertIn(key, self.agent.performance_metrics)

    def test_make_decision(self):
        """Test trading decision making."""
        # Create sample market data
        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'Open': np.random.randn(10) + 100,
            'High': np.random.randn(10) + 101,
            'Low': np.random.randn(10) + 99,
            'Close': np.random.randn(10) + 100,
            'Volume': np.random.randint(1000, 10000, 10)
        })

        decision = self.agent.make_decision(market_data)

        # Check decision structure
        expected_keys = ['action', 'confidence', 'position_size', 'stop_loss', 'take_profit', 'risk_amount']
        for key in expected_keys:
            self.assertIn(key, decision)

        # Check action values
        self.assertIn(decision['action'], ['buy', 'hold', 'sell'])
        self.assertIsInstance(decision['confidence'], float)

    def test_calculate_position_size(self):
        """Test position size calculation."""
        decision = {
            'action': 'buy',
            'confidence': 0.8,
            'symbol': 'AAPL'
        }

        position_size = self.agent._calculate_position_size(decision, 100000)

        self.assertIsInstance(position_size, int)
        self.assertGreater(position_size, 0)

    def test_update_performance_metrics(self):
        """Test performance metrics update."""
        trade = {
            'decision': {'expected_profit': 100}
        }

        initial_trades = self.agent.performance_metrics['total_trades']

        self.agent._update_performance_metrics(trade)

        self.assertEqual(self.agent.performance_metrics['total_trades'], initial_trades + 1)
        self.assertEqual(self.agent.performance_metrics['winning_trades'], 1)

    def test_get_performance_report(self):
        """Test performance report generation."""
        report = self.agent.get_performance_report()

        expected_keys = ['total_trades', 'winning_trades', 'losing_trades',
                        'total_pnl', 'sharpe_ratio', 'max_drawdown',
                        'win_rate', 'profit_factor']

        for key in expected_keys:
            self.assertIn(key, report)
            self.assertIsInstance(report[key], (int, float))


class TestRiskManager(unittest.TestCase):
    """Test cases for RiskManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.risk_manager = RiskManager(self.config)

    def test_initialization(self):
        """Test risk manager initialization."""
        self.assertEqual(self.risk_manager.max_position_size, self.config.trading.max_position_size)
        self.assertEqual(self.risk_manager.stop_loss_pct, self.config.trading.stop_loss_pct)
        self.assertEqual(self.risk_manager.take_profit_pct, self.config.trading.take_profit_pct)

    def test_evaluate_trade(self):
        """Test trade evaluation."""
        prediction = {
            'trading_signal': 'buy',
            'confidence': 0.8,
            'risk_probability': 0.3
        }

        market_data = pd.DataFrame({
            'Close': [100, 101, 102]
        })

        current_positions = {
            'AAPL': {'market_value': 5000}
        }

        evaluation = self.risk_manager.evaluate_trade(prediction, market_data, current_positions)

        # Check evaluation structure
        expected_keys = ['action', 'confidence', 'position_size', 'stop_loss', 'take_profit', 'risk_amount']
        for key in expected_keys:
            self.assertIn(key, evaluation)

        # Check values
        self.assertEqual(evaluation['action'], 'buy')
        self.assertEqual(evaluation['confidence'], 0.8)
        self.assertIsInstance(evaluation['position_size'], (int, float))

    def test_calculate_safe_position_size(self):
        """Test safe position size calculation."""
        prediction = {'confidence': 0.8}
        market_data = pd.DataFrame({'Close': [100]})
        current_exposure = 10000

        position_size = self.risk_manager._calculate_safe_position_size(
            prediction, market_data, current_exposure
        )

        self.assertIsInstance(position_size, (int, float))
        self.assertGreaterEqual(position_size, 0)

    def test_stop_loss_calculation(self):
        """Test stop loss price calculation."""
        market_data = pd.DataFrame({'Close': [100]})

        stop_loss = self.risk_manager._calculate_stop_loss(market_data)

        expected_stop = 100 * (1 - self.config.trading.stop_loss_pct)
        self.assertEqual(stop_loss, expected_stop)

    def test_take_profit_calculation(self):
        """Test take profit price calculation."""
        market_data = pd.DataFrame({'Close': [100]})

        take_profit = self.risk_manager._calculate_take_profit(market_data)

        expected_profit = 100 * (1 + self.config.trading.take_profit_pct)
        self.assertEqual(take_profit, expected_profit)


class TestLiveTradingEngine(unittest.TestCase):
    """Test cases for LiveTradingEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.model = NCATradingModel(self.config)
        self.agent = TradingAgent(self.model, self.config)
        self.engine = LiveTradingEngine(self.agent, self.config)

    def test_initialization(self):
        """Test live trading engine initialization."""
        self.assertEqual(self.engine.trading_agent, self.agent)
        self.assertEqual(self.engine.config, self.config)
        self.assertFalse(self.engine.is_running)
        self.assertIsNone(self.engine.stream)

    @patch('stream.Stream')
    def test_initialize_streaming(self, mock_stream):
        """Test streaming initialization."""
        mock_stream_instance = Mock()
        mock_stream.return_value = mock_stream_instance

        async def run_test():
            await self.engine._initialize_streaming(['AAPL', 'MSFT'])

            mock_stream.assert_called_once()
            self.assertIsNotNone(self.engine.stream)

        asyncio.run(run_test())

    def test_bar_update_handling(self):
        """Test bar update handling."""
        # Create mock bar
        mock_bar = Mock()
        mock_bar.symbol = 'AAPL'
        mock_bar.timestamp = datetime.now()
        mock_bar.open = 100.0
        mock_bar.high = 101.0
        mock_bar.low = 99.0
        mock_bar.close = 100.5
        mock_bar.volume = 1000

        # Handle bar update
        self.engine._on_bar_update(mock_bar)

        # Check that data is stored
        self.assertIn('AAPL', self.engine.market_data_buffer)
        self.assertEqual(len(self.engine.market_data_buffer['AAPL']), 1)

    def test_trade_update_handling(self):
        """Test trade update handling."""
        # Create mock trade
        mock_trade = Mock()
        mock_trade.symbol = 'AAPL'
        mock_trade.price = 100.5

        # Handle trade update (should not raise exception)
        self.engine._on_trade_update(mock_trade)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=50),
            'Open': np.random.randn(50) + 100,
            'High': np.random.randn(50) + 101,
            'Low': np.random.randn(50) + 99,
            'Close': np.random.randn(50) + 100,
            'Volume': np.random.randint(1000, 10000, 50)
        })

    def test_create_trading_environment(self):
        """Test create_trading_environment function."""
        env = create_trading_environment(self.sample_data, self.config)

        self.assertIsInstance(env, TradingEnvironment)
        self.assertEqual(env.initial_balance, 100000)

    def test_create_trading_agent(self):
        """Test create_trading_agent function."""
        model = NCATradingModel(self.config)
        agent = create_trading_agent(model, self.config)

        self.assertIsInstance(agent, TradingAgent)
        self.assertEqual(agent.model, model)
        self.assertEqual(agent.config, self.config)


if __name__ == '__main__':
    unittest.main()