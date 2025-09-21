"""
Unit tests for data handler module.

Tests data fetching, preprocessing, and technical indicator calculation.
"""

import unittest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np

from ..data_handler import DataFetcher, TechnicalIndicators, DataPreprocessor, DataHandler


class TestDataFetcher(unittest.TestCase):
    """Test cases for DataFetcher class."""

    def setUp(self):
        """Set up test fixtures."""
        self.fetcher = DataFetcher()

    @patch('yfinance.Ticker')
    def test_fetch_yfinance_data(self, mock_ticker):
        """Test fetching data from Yahoo Finance."""
        # Mock ticker and history data
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame({
            ('Open', 'AAPL'): [100, 101, 102],
            ('High', 'AAPL'): [105, 106, 107],
            ('Low', 'AAPL'): [99, 100, 101],
            ('Close', 'AAPL'): [104, 105, 106],
            ('Volume', 'AAPL'): [1000, 1100, 1200]
        })
        mock_ticker.return_value = mock_ticker_instance

        # Test data fetching
        async def run_test():
            data = await self.fetcher.fetch_yfinance_data(
                'AAPL', '2023-01-01', '2023-01-03'
            )

            self.assertEqual(len(data), 3)
            self.assertIn('Open', data.columns)
            self.assertIn('Close', data.columns)
            self.assertIn('Volume', data.columns)

        asyncio.run(run_test())

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_save_load_data(self, mock_file):
        """Test saving and loading data."""
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=3),
            'Open': [100, 101, 102],
            'Close': [104, 105, 106],
            'Volume': [1000, 1100, 1200]
        })

        # Test save
        with tempfile.TemporaryDirectory() as temp_dir:
            self.fetcher.data_handler.save_data(test_data, 'test.parquet', temp_dir)

            # Test load
            loaded_data = self.fetcher.data_handler.load_data('test.parquet', temp_dir)

            pd.testing.assert_frame_equal(test_data, loaded_data)


class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for TechnicalIndicators class."""

    def setUp(self):
        """Set up test fixtures."""
        self.indicators = TechnicalIndicators()

        # Create sample price data
        self.prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                               110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
        self.highs = self.prices + 2
        self.lows = self.prices - 2
        self.volumes = np.random.randint(1000, 2000, len(self.prices))

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        rsi = self.indicators.calculate_rsi(self.prices)

        self.assertEqual(len(rsi), len(self.prices))
        self.assertTrue(all(np.isnan(rsi[:13])))  # RSI needs 14 periods
        self.assertTrue(all(np.isfinite(rsi[13:])))  # Valid RSI values

    def test_calculate_macd(self):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = self.indicators.calculate_macd(self.prices)

        self.assertEqual(len(macd_line), len(self.prices))
        self.assertEqual(len(signal_line), len(self.prices))
        self.assertEqual(len(histogram), len(self.prices))

        # Check for NaN values in early periods
        self.assertTrue(all(np.isnan(macd_line[:25])))
        self.assertTrue(all(np.isnan(signal_line[:33])))

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = self.indicators.calculate_bollinger_bands(self.prices)

        self.assertEqual(len(upper), len(self.prices))
        self.assertEqual(len(middle), len(self.prices))
        self.assertEqual(len(lower), len(self.prices))

        # Check that upper > middle > lower
        valid_indices = np.isfinite(upper) & np.isfinite(middle) & np.isfinite(lower)
        self.assertTrue(all(upper[valid_indices] >= middle[valid_indices]))
        self.assertTrue(all(middle[valid_indices] >= lower[valid_indices]))

    def test_calculate_sma(self):
        """Test Simple Moving Average calculation."""
        sma = self.indicators.calculate_sma(self.prices, 5)

        self.assertEqual(len(sma), len(self.prices))
        self.assertTrue(all(np.isnan(sma[:4])))  # SMA needs 5 periods

    def test_calculate_ema(self):
        """Test Exponential Moving Average calculation."""
        ema = self.indicators.calculate_ema(self.prices, 5)

        self.assertEqual(len(ema), len(self.prices))
        self.assertTrue(all(np.isnan(ema[:4])))  # EMA needs 5 periods

    def test_calculate_stochastic(self):
        """Test Stochastic Oscillator calculation."""
        k_percent, d_percent = self.indicators.calculate_stochastic(
            self.highs, self.lows, self.prices
        )

        self.assertEqual(len(k_percent), len(self.prices))
        self.assertEqual(len(d_percent), len(self.prices))

        # Check value ranges
        valid_indices = np.isfinite(k_percent)
        self.assertTrue(all(k_percent[valid_indices] >= 0))
        self.assertTrue(all(k_percent[valid_indices] <= 100))

    def test_calculate_all_indicators(self):
        """Test calculation of all technical indicators."""
        # Create sample data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=50),
            'Open': self.prices[:50],
            'High': self.highs[:50],
            'Low': self.lows[:50],
            'Close': self.prices[:50],
            'Volume': self.volumes[:50]
        })

        # Calculate all indicators
        result = self.indicators.calculate_all_indicators(data)

        # Check that new columns are added
        expected_columns = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                           'BB_Upper', 'BB_Middle', 'BB_Lower',
                           'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                           'Stoch_K', 'Stoch_D', 'Price_Change',
                           'Volume_Change', 'ATR', 'OBV', 'Volatility']

        for column in expected_columns:
            self.assertIn(column, result.columns)

        # Check that no NaN values remain
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        self.assertFalse(result[numeric_columns].isnull().any().any())


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()

        # Create sample data
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100),
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100)
        })

    def test_create_sequences(self):
        """Test sequence creation."""
        data = np.random.randn(100, 5)
        sequence_length = 10
        prediction_horizon = 2

        X, y = self.preprocessor.create_sequences(data, sequence_length, prediction_horizon)

        expected_length = len(data) - sequence_length - prediction_horizon + 1
        self.assertEqual(len(X), expected_length)
        self.assertEqual(len(y), expected_length)
        self.assertEqual(X.shape[1:], (sequence_length, data.shape[1]))
        self.assertEqual(y.shape[1:], data.shape[1:])

    def test_normalize_features(self):
        """Test feature normalization."""
        data = self.sample_data.copy()

        # Test normalization
        normalized = self.preprocessor.normalize_features(data, ['feature1', 'feature2'])

        # Check that scalers are created
        self.assertIn('feature1', self.preprocessor.scalers)
        self.assertIn('feature2', self.preprocessor.scalers)

        # Check that values are normalized
        self.assertAlmostEqual(normalized['feature1'].mean(), 0, places=1)
        self.assertAlmostEqual(normalized['feature1'].std(), 1, places=1)

    def test_split_data(self):
        """Test data splitting."""
        train_data, val_data, test_data = self.preprocessor.split_data(
            self.sample_data, train_ratio=0.7, val_ratio=0.15
        )

        total_length = len(train_data) + len(val_data) + len(test_data)
        self.assertEqual(total_length, len(self.sample_data))

        # Check approximate ratios
        self.assertAlmostEqual(len(train_data) / len(self.sample_data), 0.7, places=1)
        self.assertAlmostEqual(len(val_data) / len(self.sample_data), 0.15, places=1)

    def test_prepare_for_training(self):
        """Test data preparation for training."""
        # Add more features to sample data
        data = self.sample_data.copy()
        data['RSI'] = np.random.randn(100)
        data['MACD'] = np.random.randn(100)

        result = self.preprocessor.prepare_for_training(data)

        # Check result structure
        self.assertIn('X_train', result)
        self.assertIn('y_train', result)
        self.assertIn('X_val', result)
        self.assertIn('y_val', result)
        self.assertIn('X_test', result)
        self.assertIn('y_test', result)
        self.assertIn('feature_columns', result)

        # Check data shapes
        self.assertEqual(result['X_train'].ndim, 3)  # Should be sequences
        self.assertEqual(result['y_train'].ndim, 2)


class TestDataHandler(unittest.TestCase):
    """Test cases for DataHandler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = DataHandler()

    def test_initialization(self):
        """Test DataHandler initialization."""
        self.assertIsInstance(self.handler.fetcher, DataFetcher)
        self.assertIsInstance(self.handler.indicators, TechnicalIndicators)
        self.assertIsInstance(self.handler.preprocessor, DataPreprocessor)

    @patch.object(DataHandler, 'get_historical_data')
    def test_get_multiple_tickers_data(self, mock_get_data):
        """Test fetching data for multiple tickers."""
        # Mock data for two tickers
        mock_data1 = pd.DataFrame({'Close': [100, 101, 102]})
        mock_data2 = pd.DataFrame({'Close': [200, 201, 202]})
        mock_get_data.side_effect = [mock_data1, mock_data2]

        async def run_test():
            data = await self.handler.get_multiple_tickers_data(
                ['AAPL', 'MSFT'], '2023-01-01', '2023-01-03'
            )

            self.assertEqual(len(data), 2)
            self.assertIn('AAPL', data)
            self.assertIn('MSFT', data)

        asyncio.run(run_test())

    def test_caching(self):
        """Test data caching functionality."""
        # Test cache key generation
        key = self.handler._generate_cache_key('test', '2023-01-01', '2023-01-03')
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 16)  # MD5 hash length

        # Test cache storage
        test_data = pd.DataFrame({'Close': [100, 101]})
        self.handler.cache_data(test_data, 'TEST', '2023-01-01', '2023-01-03')

        # Test cache retrieval
        cached_data = self.handler.get_cached_data('TEST', '2023-01-01', '2023-01-03')
        pd.testing.assert_frame_equal(test_data, cached_data)


if __name__ == '__main__':
    unittest.main()