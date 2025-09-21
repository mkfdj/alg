"""
Unit tests for utils module.

Tests caching, risk calculation, performance monitoring, and utility functions.
"""

import unittest
import time
import tempfile
import os
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from ..utils import (
    Cache, PersistentCache, cache_result, time_function, log_function,
    DDPUtils, RiskCalculator, PerformanceMonitor, LoggerUtils, DataUtils
)
from ..config import ConfigManager


class TestCache(unittest.TestCase):
    """Test cases for Cache class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = Cache(max_size=10, ttl=1)  # 1 second TTL for testing

    def test_cache_operations(self):
        """Test basic cache operations."""
        # Test set and get
        self.cache.set('key1', 'value1')
        self.assertEqual(self.cache.get('key1'), 'value1')

        # Test cache miss
        self.assertIsNone(self.cache.get('nonexistent'))

    def test_cache_ttl(self):
        """Test cache TTL functionality."""
        # Set item
        self.cache.set('key1', 'value1')
        self.assertEqual(self.cache.get('key1'), 'value1')

        # Wait for TTL to expire
        time.sleep(1.1)

        # Should return None due to TTL
        self.assertIsNone(self.cache.get('key1'))

    def test_cache_size_limit(self):
        """Test cache size limit."""
        # Fill cache to max size
        for i in range(10):
            self.cache.set(f'key{i}', f'value{i}')

        # Add one more item (should evict oldest)
        self.cache.set('key10', 'value10')

        # Check that cache size is maintained
        self.assertEqual(self.cache.stats()['size'], 10)

    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.cache.stats()
        expected_keys = ['size', 'max_size', 'hit_rate']

        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], (int, float))


class TestPersistentCache(unittest.TestCase):
    """Test cases for PersistentCache class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = PersistentCache(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_persistent_cache_operations(self):
        """Test persistent cache operations."""
        # Test set and get
        test_data = {'test': 'data'}
        self.cache.set('key1', test_data)

        retrieved = self.cache.get('key1')
        self.assertEqual(retrieved, test_data)

    def test_cache_file_creation(self):
        """Test that cache files are created."""
        test_data = 'test_value'
        self.cache.set('key1', test_data)

        cache_file = os.path.join(self.temp_dir, 'key1.pkl')
        self.assertTrue(os.path.exists(cache_file))

    def test_cache_clear(self):
        """Test cache clearing."""
        self.cache.set('key1', 'value1')
        self.cache.set('key2', 'value2')

        self.cache.clear()

        # Check that files are removed
        files = os.listdir(self.temp_dir)
        self.assertEqual(len(files), 0)


class TestDecorators(unittest.TestCase):
    """Test cases for decorator functions."""

    def test_cache_result_decorator(self):
        """Test cache_result decorator."""
        @cache_result(ttl=10)
        def expensive_function(x, y):
            return x + y

        # First call
        result1 = expensive_function(5, 3)

        # Second call (should be cached)
        result2 = expensive_function(5, 3)

        self.assertEqual(result1, 8)
        self.assertEqual(result2, 8)

    def test_time_function_decorator(self):
        """Test time_function decorator."""
        @time_function
        def slow_function():
            time.sleep(0.1)
            return 'done'

        # Should not raise exception and should print timing
        result = slow_function()
        self.assertEqual(result, 'done')

    def test_log_function_decorator(self):
        """Test log_function decorator."""
        @log_function
        def test_function():
            return 'test_result'

        # Should not raise exception
        result = test_function()
        self.assertEqual(result, 'test_result')


class TestDDPUtils(unittest.TestCase):
    """Test cases for DDPUtils class."""

    @patch('torch.distributed.init_process_group')
    @patch('torch.distributed.is_initialized', return_value=False)
    def test_setup_ddp(self, mock_is_initialized, mock_init_process_group):
        """Test DDP setup."""
        DDPUtils.setup_ddp(rank=0, world_size=2)
        mock_init_process_group.assert_called_once()

    @patch('torch.distributed.destroy_process_group')
    def test_cleanup_ddp(self, mock_destroy):
        """Test DDP cleanup."""
        DDPUtils.cleanup_ddp()
        mock_destroy.assert_called_once()

    @patch('torch.distributed.is_initialized', return_value=False)
    def test_is_main_process(self, mock_is_initialized):
        """Test main process detection."""
        self.assertTrue(DDPUtils.is_main_process())

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_rank', return_value=0)
    def test_is_main_process_ddp(self, mock_get_rank, mock_is_initialized):
        """Test main process detection in DDP."""
        self.assertTrue(DDPUtils.is_main_process())

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_rank', return_value=1)
    def test_is_not_main_process_ddp(self, mock_get_rank, mock_is_initialized):
        """Test non-main process detection in DDP."""
        self.assertFalse(DDPUtils.is_main_process())

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.all_reduce')
    @patch('torch.distributed.get_world_size', return_value=4)
    def test_all_reduce_tensor(self, mock_get_world_size, mock_all_reduce):
        """Test tensor all-reduce."""
        tensor = torch.tensor([1.0, 2.0, 3.0])

        result = DDPUtils.all_reduce_tensor(tensor)

        mock_all_reduce.assert_called_once()
        self.assertEqual(result, tensor)

    @patch('torch.distributed.is_initialized', return_value=False)
    def test_all_reduce_tensor_no_ddp(self, mock_is_initialized):
        """Test tensor all-reduce without DDP."""
        tensor = torch.tensor([1.0, 2.0, 3.0])

        result = DDPUtils.all_reduce_tensor(tensor)

        self.assertEqual(result, tensor)


class TestRiskCalculator(unittest.TestCase):
    """Test cases for RiskCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.calculator = RiskCalculator(self.config)

        # Create sample returns data
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 1000)

    def test_calculate_var(self):
        """Test Value at Risk calculation."""
        var = self.calculator.calculate_var(self.returns, confidence_level=0.95)

        self.assertIsInstance(var, float)
        self.assertGreater(var, 0)

        # Test with different confidence levels
        var_99 = self.calculator.calculate_var(self.returns, confidence_level=0.99)
        self.assertGreater(var_99, var)  # Higher confidence should give higher VaR

    def test_calculate_cvar(self):
        """Test Conditional Value at Risk calculation."""
        cvar = self.calculator.calculate_cvar(self.returns, confidence_level=0.95)

        self.assertIsInstance(cvar, float)
        self.assertGreater(cvar, 0)

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sharpe = self.calculator.calculate_sharpe_ratio(self.returns)

        self.assertIsInstance(sharpe, float)

        # Test with zero volatility
        zero_returns = np.zeros(100)
        sharpe_zero = self.calculator.calculate_sharpe_ratio(zero_returns)
        self.assertEqual(sharpe_zero, 0)

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Create portfolio values with drawdown
        values = np.array([100, 110, 105, 115, 108, 120, 110, 125])

        drawdown = self.calculator.calculate_max_drawdown(values)

        self.assertIsInstance(drawdown, float)
        self.assertGreaterEqual(drawdown, 0)
        self.assertLessEqual(drawdown, 1)

    def test_calculate_position_size(self):
        """Test position size calculation."""
        position_size = self.calculator.calculate_position_size(
            capital=10000,
            risk_per_trade=0.02,
            stop_loss_pct=0.05,
            entry_price=100
        )

        self.assertIsInstance(position_size, int)
        self.assertGreater(position_size, 0)

    def test_calculate_kelly_position_size(self):
        """Test Kelly criterion position sizing."""
        kelly_size = self.calculator.calculate_kelly_position_size(
            win_probability=0.6,
            win_loss_ratio=2.0
        )

        self.assertIsInstance(kelly_size, float)
        self.assertGreaterEqual(kelly_size, 0)
        self.assertLessEqual(kelly_size, 1)


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for PerformanceMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()

    def test_get_system_metrics(self):
        """Test system metrics collection."""
        metrics = self.monitor.get_system_metrics()

        expected_keys = ['cpu_percent', 'memory_percent', 'gpu_utilization', 'gpu_memory_percent', 'disk_percent']
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))

    def test_get_trading_metrics(self):
        """Test trading metrics calculation."""
        trades = [
            {'pnl': 100, 'type': 'buy'},
            {'pnl': -50, 'type': 'sell'},
            {'pnl': 75, 'type': 'buy'},
            {'pnl': -25, 'type': 'sell'},
            {'pnl': 200, 'type': 'buy'}
        ]

        metrics = self.monitor.get_trading_metrics(trades)

        expected_keys = ['total_trades', 'winning_trades', 'losing_trades', 'win_rate',
                        'profit_factor', 'sharpe_ratio', 'max_drawdown', 'total_profit', 'total_loss']

        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))

        # Check calculations
        self.assertEqual(metrics['total_trades'], 5)
        self.assertEqual(metrics['winning_trades'], 3)
        self.assertEqual(metrics['losing_trades'], 2)
        self.assertEqual(metrics['win_rate'], 0.6)

    def test_log_metrics(self):
        """Test metrics logging."""
        metrics = {'loss': 0.5, 'accuracy': 0.85}

        # Should not raise exception
        self.monitor.log_metrics(metrics, step=10)

        # Check that metrics are stored
        self.assertIn('loss', self.monitor.metrics)
        self.assertIn('accuracy', self.monitor.metrics)
        self.assertEqual(self.monitor.metrics['loss'], 0.5)
        self.assertEqual(self.monitor.metrics['accuracy'], 0.85)


class TestLoggerUtils(unittest.TestCase):
    """Test cases for LoggerUtils class."""

    def test_setup_logger(self):
        """Test logger setup."""
        logger = LoggerUtils.setup_logger('test_logger', 'INFO')

        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'test_logger')
        self.assertEqual(logger.level, 20)  # INFO level

    def test_log_trading_action(self):
        """Test trading action logging."""
        # Should not raise exception
        LoggerUtils.log_trading_action('buy', 'AAPL', 100, 150.25)

    def test_log_performance_metrics(self):
        """Test performance metrics logging."""
        metrics = {'win_rate': 0.6, 'total_pnl': 1000}

        # Should not raise exception
        LoggerUtils.log_performance_metrics(metrics)

    def test_log_error(self):
        """Test error logging."""
        error = ValueError("Test error")

        # Should not raise exception
        LoggerUtils.log_error(error, "test context")


class TestDataUtils(unittest.TestCase):
    """Test cases for DataUtils class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data with outliers and missing values
        np.random.seed(42)
        data = np.random.randn(100, 3)
        data[10, 0] = 100  # Add outlier
        data[20, 1] = np.nan  # Add missing value
        data[30, 2] = np.nan  # Add missing value

        self.sample_df = pd.DataFrame(data, columns=['A', 'B', 'C'])

    def test_clean_data(self):
        """Test data cleaning."""
        cleaned = DataUtils.clean_data(self.sample_df)

        # Check that missing values are handled
        self.assertFalse(cleaned.isnull().any().any())

        # Check that outliers are still present (cleaning doesn't remove outliers by default)
        self.assertTrue((cleaned['A'] == 100).any())

    def test_normalize_data(self):
        """Test data normalization."""
        normalized = DataUtils.normalize_data(self.sample_df, ['A', 'B'])

        # Check that specified columns are normalized
        self.assertAlmostEqual(normalized['A'].mean(), 0, places=1)
        self.assertAlmostEqual(normalized['A'].std(), 1, places=1)

        # Check that other columns are unchanged
        self.assertNotAlmostEqual(normalized['C'].mean(), 0, places=1)

    def test_calculate_returns(self):
        """Test returns calculation."""
        prices = np.array([100, 105, 102, 108, 107])

        returns = DataUtils.calculate_returns(prices)

        # Check shape
        self.assertEqual(len(returns), len(prices) - 1)

        # Check calculation
        expected_returns = np.array([0.05, -0.02857, 0.05882, -0.00926])
        np.testing.assert_array_almost_equal(returns, expected_returns, decimal=4)

    def test_calculate_rolling_metrics(self):
        """Test rolling metrics calculation."""
        data = np.random.randn(50)
        window = 10

        metrics = DataUtils.calculate_rolling_metrics(data, window)

        expected_keys = ['mean', 'std', 'min', 'max']
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertEqual(len(metrics[key]), len(data))

        # Check that early values are NaN
        self.assertTrue(np.isnan(metrics['mean'][0]))
        self.assertTrue(np.isfinite(metrics['mean'][window]))


if __name__ == '__main__':
    unittest.main()