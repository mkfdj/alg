"""
Data handling module for NCA Trading Bot.

This module provides comprehensive data fetching, preprocessing, and technical
indicator calculation capabilities for financial time series data.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf
import talib
from alpaca_trade_api.rest import REST
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .config import get_config


class DataFetcher:
    """
    Handles fetching financial data from multiple sources.

    Supports yfinance for free data and Alpaca API for premium data.
    Includes rate limiting and error handling.
    """

    def __init__(self):
        """Initialize data fetcher with API connections."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # Initialize API connections
        self.alpaca = None
        if self.config.api.alpaca_api_key and self.config.api.alpaca_secret_key:
            try:
                self.alpaca = REST(
                    key_id=self.config.api.alpaca_api_key,
                    secret_key=self.config.api.alpaca_secret_key,
                    base_url=self.config.api.alpaca_base_url
                )
                self.logger.info("Alpaca API initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Alpaca API: {e}")

        # Rate limiting
        self._last_requests = []
        self.requests_per_minute = self.config.api.requests_per_minute

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        now = datetime.now()
        # Remove requests older than 1 minute
        self._last_requests = [req_time for req_time in self._last_requests
                             if now - req_time < timedelta(minutes=1)]

        if len(self._last_requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self._last_requests[0]).seconds
            if sleep_time > 0:
                import time
                time.sleep(sleep_time)

        self._last_requests.append(now)

    async def fetch_yfinance_data(self, ticker: str, start_date: str,
                                end_date: str, interval: str = "1m") -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1m, 5m, 15m, 1h, 1d)

        Returns:
            DataFrame with OHLCV data
        """
        self._check_rate_limit()

        try:
            # Run yfinance in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                ticker_obj = yf.Ticker(ticker)
                df = await loop.run_in_executor(
                    executor, ticker_obj.history,
                    start_date, end_date, interval
                )

            if df.empty:
                raise ValueError(f"No data found for {ticker}")

            # Flatten column names and ensure proper format
            df.columns = df.columns.get_level_values(0)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            self.logger.info(f"Fetched {len(df)} records for {ticker} from Yahoo Finance")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {e}")
            raise

    async def fetch_alpaca_data(self, ticker: str, start_date: str,
                               end_date: str, timeframe: str = "1Min") -> pd.DataFrame:
        """
        Fetch data from Alpaca API.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Data timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.alpaca:
            raise ValueError("Alpaca API not initialized")

        self._check_rate_limit()

        try:
            # Convert timeframe format
            timeframe_map = {
                "1m": "1Min", "5m": "5Min", "15m": "15Min",
                "1h": "1Hour", "1d": "1Day"
            }
            alpaca_timeframe = timeframe_map.get(timeframe, timeframe)

            bars = self.alpaca.get_bars(
                symbol=ticker,
                timeframe=alpaca_timeframe,
                start=start_date,
                end=end_date,
                adjustment='raw'
            ).df

            if bars.empty:
                raise ValueError(f"No data found for {ticker}")

            # Reset index to make timestamp a column
            bars = bars.reset_index()
            bars = bars[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            bars.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']

            self.logger.info(f"Fetched {len(bars)} records for {ticker} from Alpaca")
            return bars

        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker} from Alpaca: {e}")
            raise

    async def fetch_data(self, ticker: str, start_date: str, end_date: str,
                        interval: str = "1m", use_alpaca: bool = None) -> pd.DataFrame:
        """
        Fetch data from available sources with fallback.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            use_alpaca: Force use of specific API (None for auto)

        Returns:
            DataFrame with OHLCV data
        """
        # Auto-select API based on availability and configuration
        if use_alpaca is None:
            use_alpaca = self.alpaca is not None and interval in ["1m", "5m", "15m"]

        try:
            if use_alpaca:
                return await self.fetch_alpaca_data(ticker, start_date, end_date, interval)
            else:
                return await self.fetch_yfinance_data(ticker, start_date, end_date, interval)

        except Exception as e:
            self.logger.warning(f"Primary fetch failed for {ticker}: {e}")

            # Try fallback
            try:
                if use_alpaca:
                    self.logger.info(f"Falling back to Yahoo Finance for {ticker}")
                    return await self.fetch_yfinance_data(ticker, start_date, end_date, interval)
                else:
                    if self.alpaca:
                        self.logger.info(f"Falling back to Alpaca for {ticker}")
                        return await self.fetch_alpaca_data(ticker, start_date, end_date, interval)
                    else:
                        raise
            except Exception as fallback_error:
                self.logger.error(f"All data sources failed for {ticker}: {fallback_error}")
                raise


class TechnicalIndicators:
    """
    Calculates technical indicators for financial data.

    Provides comprehensive technical analysis indicators including
    momentum, volatility, volume, and trend indicators.
    """

    def __init__(self):
        """Initialize technical indicators calculator."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

    def calculate_rsi(self, prices: np.ndarray, period: int = None) -> np.ndarray:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            prices: Array of closing prices
            period: RSI calculation period

        Returns:
            Array of RSI values
        """
        if period is None:
            period = self.config.data.rsi_period

        try:
            rsi = talib.RSI(prices, timeperiod=period)
            return rsi
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return np.full_like(prices, np.nan)

    def calculate_macd(self, prices: np.ndarray, fast: int = None,
                      slow: int = None, signal: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Moving Average Convergence Divergence (MACD).

        Args:
            prices: Array of closing prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        if fast is None:
            fast = self.config.data.macd_fast
        if slow is None:
            slow = self.config.data.macd_slow
        if signal is None:
            signal = self.config.data.macd_signal

        try:
            macd_line, signal_line, histogram = talib.MACD(
                prices, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            return macd_line, signal_line, histogram
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return np.full_like(prices, np.nan), np.full_like(prices, np.nan), np.full_like(prices, np.nan)

    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = None,
                                std_dev: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Array of closing prices
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        if period is None:
            period = self.config.data.bb_period
        if std_dev is None:
            std_dev = self.config.data.bb_std

        try:
            upper, middle, lower = talib.BBANDS(
                prices, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0
            )
            return upper, middle, lower
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return np.full_like(prices, np.nan), np.full_like(prices, np.nan), np.full_like(prices, np.nan)

    def calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Simple Moving Average.

        Args:
            prices: Array of prices
            period: Moving average period

        Returns:
            Array of SMA values
        """
        try:
            sma = talib.SMA(prices, timeperiod=period)
            return sma
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {e}")
            return np.full_like(prices, np.nan)

    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average.

        Args:
            prices: Array of prices
            period: EMA period

        Returns:
            Array of EMA values
        """
        try:
            ema = talib.EMA(prices, timeperiod=period)
            return ema
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return np.full_like(prices, np.nan)

    def calculate_stochastic(self, high: np.ndarray, low: np.ndarray,
                           close: np.ndarray, k_period: int = 14,
                           d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Stochastic Oscillator.

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            k_period: %K period
            d_period: %D period

        Returns:
            Tuple of (%K, %D)
        """
        try:
            k_percent, d_percent = talib.STOCH(
                high, low, close, fastk_period=k_period, slowk_period=d_period
            )
            return k_percent, d_percent
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {e}")
            return np.full_like(close, np.nan), np.full_like(close, np.nan)

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()

        # Extract price arrays
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values

        # Calculate indicators
        df['RSI'] = self.calculate_rsi(close)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(close)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(close)
        df['SMA_20'] = self.calculate_sma(close, 20)
        df['SMA_50'] = self.calculate_sma(close, 50)
        df['EMA_12'] = self.calculate_ema(close, 12)
        df['EMA_26'] = self.calculate_ema(close, 26)
        df['Stoch_K'], df['Stoch_D'] = self.calculate_stochastic(high, low, close)

        # Calculate additional features
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['ATR'] = self.calculate_atr(high, low, close)
        df['OBV'] = self.calculate_obv(close, volume)

        # Calculate volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(0)

        self.logger.info(f"Calculated technical indicators for {len(df)} records")
        return df

    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     period: int = 14) -> np.ndarray:
        """Calculate Average True Range."""
        try:
            return talib.ATR(high, low, close, timeperiod=period)
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return np.full_like(close, np.nan)

    def calculate_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate On-Balance Volume."""
        try:
            return talib.OBV(close, volume)
        except Exception as e:
            self.logger.error(f"Error calculating OBV: {e}")
            return np.full_like(close, np.nan)


class DataPreprocessor:
    """
    Handles data preprocessing and normalization for machine learning.

    Provides sequence creation, feature scaling, and data splitting functionality.
    """

    def __init__(self):
        """Initialize data preprocessor."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.scalers = {}

    def create_sequences(self, data: np.ndarray, sequence_length: int,
                        prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and target values for time series prediction.

        Args:
            data: Input data array
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict ahead

        Returns:
            Tuple of (X, y) where X is input sequences and y is target values
        """
        X, y = [], []

        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length + prediction_horizon - 1])

        return np.array(X), np.array(y)

    def normalize_features(self, data: pd.DataFrame, feature_columns: List[str] = None) -> pd.DataFrame:
        """
        Normalize features using StandardScaler or MinMaxScaler.

        Args:
            data: Input DataFrame
            feature_columns: Columns to normalize (None for all numeric columns)

        Returns:
            Normalized DataFrame
        """
        data = data.copy()

        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target-like columns from normalization
        exclude_columns = ['timestamp', 'target', 'prediction']
        feature_columns = [col for col in feature_columns if col not in exclude_columns]

        for column in feature_columns:
            if column not in self.scalers:
                if self.config.data.normalize_features:
                    if self.config.data.feature_range == (-1, 1):
                        self.scalers[column] = StandardScaler()
                    else:
                        self.scalers[column] = MinMaxScaler(feature_range=self.config.data.feature_range)

            if column in self.scalers:
                # Reshape for single feature scaling
                values = data[column].values.reshape(-1, 1)
                scaled_values = self.scalers[column].fit_transform(values)
                data[column] = scaled_values.flatten()

        self.logger.info(f"Normalized {len(feature_columns)} feature columns")
        return data

    def split_data(self, data: pd.DataFrame, train_ratio: float = None,
                  val_ratio: float = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            data: Input DataFrame
            train_ratio: Training set ratio
            val_ratio: Validation set ratio

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if train_ratio is None:
            train_ratio = self.config.data.train_test_split
        if val_ratio is None:
            val_ratio = self.config.data.validation_split

        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        test_size = n - train_size - val_size

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        self.logger.info(f"Split data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        return train_data, val_data, test_data

    def prepare_for_training(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prepare DataFrame for machine learning training.

        Args:
            df: Input DataFrame with features

        Returns:
            Dictionary containing processed data arrays
        """
        # Separate features and target
        feature_columns = [col for col in df.columns
                          if col not in ['timestamp', 'target', 'prediction']]

        X = df[feature_columns].values
        y = df.get('target', df['Close'].shift(-1).values).values

        # Remove NaN values
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]

        # Create sequences
        X_seq, y_seq = self.create_sequences(
            X, self.config.data.sequence_length, self.config.data.prediction_horizon
        )

        # Split data
        train_data, val_data, test_data = self.split_data(df.iloc[valid_indices])

        # Prepare training arrays
        X_train = train_data[feature_columns].values
        y_train = train_data.get('target', train_data['Close'].shift(-1).values).values

        X_val = val_data[feature_columns].values
        y_val = val_data.get('target', val_data['Close'].shift(-1).values).values

        X_test = test_data[feature_columns].values
        y_test = test_data.get('target', test_data['Close'].shift(-1).values).values

        # Remove NaN values from splits
        X_train = X_train[~np.isnan(y_train)]
        y_train = y_train[~np.isnan(y_train)]

        X_val = X_val[~np.isnan(y_val)]
        y_val = y_val[~np.isnan(y_val)]

        X_test = X_test[~np.isnan(y_test)]
        y_test = y_test[~np.isnan(y_test)]

        # Create sequences for training
        X_train_seq, y_train_seq = self.create_sequences(
            X_train, self.config.data.sequence_length, self.config.data.prediction_horizon
        )
        X_val_seq, y_val_seq = self.create_sequences(
            X_val, self.config.data.sequence_length, self.config.data.prediction_horizon
        )
        X_test_seq, y_test_seq = self.create_sequences(
            X_test, self.config.data.sequence_length, self.config.data.prediction_horizon
        )

        result = {
            'X_train': X_train_seq, 'y_train': y_train_seq,
            'X_val': X_val_seq, 'y_val': y_val_seq,
            'X_test': X_test_seq, 'y_test': y_test_seq,
            'feature_columns': feature_columns,
            'raw_data': {'train': train_data, 'val': val_data, 'test': test_data}
        }

        self.logger.info(f"Prepared data for training: X_train={X_train_seq.shape}, X_val={X_val_seq.shape}, X_test={X_test_seq.shape}")
        return result


class DataHandler:
    """
    Main data handling class that orchestrates data fetching, processing, and preparation.

    Provides a unified interface for all data operations in the trading bot.
    """

    def __init__(self):
        """Initialize data handler."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.fetcher = DataFetcher()
        self.indicators = TechnicalIndicators()
        self.preprocessor = DataPreprocessor()

    async def get_historical_data(self, ticker: str, start_date: str,
                                 end_date: str, interval: str = "1m") -> pd.DataFrame:
        """
        Get historical data with technical indicators.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            DataFrame with OHLCV and technical indicators
        """
        # Fetch raw data
        df = await self.fetcher.fetch_data(ticker, start_date, end_date, interval)

        # Add timestamp column if not present
        if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={'index': 'timestamp'})

        # Calculate technical indicators
        df = self.indicators.calculate_all_indicators(df)

        # Normalize features
        df = self.preprocessor.normalize_features(df)

        return df

    async def get_multiple_tickers_data(self, tickers: List[str], start_date: str,
                                       end_date: str, interval: str = "1m") -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            Dictionary mapping tickers to DataFrames
        """
        tasks = []
        for ticker in tickers:
            task = self.get_historical_data(ticker, start_date, end_date, interval)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data_dict = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to fetch data for {ticker}: {result}")
            else:
                data_dict[ticker] = result

        self.logger.info(f"Successfully fetched data for {len(data_dict)}/{len(tickers)} tickers")
        return data_dict

    def save_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                  filename: str, data_dir: str = None) -> None:
        """
        Save data to disk.

        Args:
            data: Data to save (DataFrame or dict of DataFrames)
            filename: Output filename
            data_dir: Directory to save data (None for default)
        """
        if data_dir is None:
            data_dir = self.config.system.data_dir

        data_path = Path(data_dir) / filename

        try:
            if isinstance(data, dict):
                # Save multiple tickers data
                with pd.HDFStore(data_path, mode='w') as store:
                    for ticker, df in data.items():
                        store.put(ticker, df, format='table')
            else:
                # Save single DataFrame
                data.to_parquet(data_path)

            self.logger.info(f"Data saved to {data_path}")

        except Exception as e:
            self.logger.error(f"Error saving data: {e}")

    def load_data(self, filename: str, data_dir: str = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load data from disk.

        Args:
            filename: Input filename
            data_dir: Directory to load from (None for default)

        Returns:
            Loaded data
        """
        if data_dir is None:
            data_dir = self.config.system.data_dir

        data_path = Path(data_dir) / filename

        try:
            if data_path.suffix == '.h5':
                # Load HDF5 file with multiple tickers
                data_dict = {}
                with pd.HDFStore(data_path, mode='r') as store:
                    for key in store.keys():
                        ticker = key.strip('/')
                        data_dict[ticker] = store.get(ticker)
                return data_dict
            else:
                # Load single DataFrame
                return pd.read_parquet(data_path)

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    @lru_cache(maxsize=100)
    def get_cached_data(self, ticker: str, start_date: str, end_date: str,
                       interval: str = "1m") -> Optional[pd.DataFrame]:
        """
        Get cached data if available.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            Cached DataFrame or None if not available
        """
        cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
        cache_file = self.config.system.data_dir / "cache" / f"{cache_key}.parquet"

        if cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                return None

        return None

    def cache_data(self, data: pd.DataFrame, ticker: str, start_date: str,
                   end_date: str, interval: str = "1m") -> None:
        """
        Cache data to disk.

        Args:
            data: Data to cache
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
        """
        cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
        cache_dir = self.config.system.data_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{cache_key}.parquet"

        try:
            data.to_parquet(cache_file)
        except Exception as e:
            self.logger.error(f"Error caching data: {e}")


# Global data handler instance
data_handler = DataHandler()


async def fetch_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Fetch sample data for testing and development.

    Returns:
        Dictionary of sample ticker data
    """
    config = get_config()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    return await data_handler.get_multiple_tickers_data(
        config.data.tickers[:3],  # First 3 tickers
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        '1d'  # Daily data for sample
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        print("NCA Trading Bot - Data Handler Demo")
        print("=" * 40)

        # Fetch sample data
        print("Fetching sample data...")
        data = await fetch_sample_data()

        print(f"Successfully fetched data for {len(data)} tickers:")
        for ticker, df in data.items():
            print(f"  {ticker}: {len(df)} records, {len(df.columns)} features")

        # Save sample data
        data_handler.save_data(data, "sample_data.h5")

        print("Sample data saved to data/sample_data.h5")

    # Run demo
    asyncio.run(main())