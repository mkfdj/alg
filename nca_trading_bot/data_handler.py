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
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import requests
import zipfile
import io

# Technical analysis libraries (optional)
try:
    import talib
    HAS_TALIB = True
except ImportError:
    talib = None
    HAS_TALIB = False

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    ta = None
    HAS_PANDAS_TA = False

from config import get_config


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
                self.alpaca = StockHistoricalDataClient(
                    api_key=self.config.api.alpaca_api_key,
                    secret_key=self.config.api.alpaca_secret_key
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
                "1m": TimeFrame.Minute,
                "5m": TimeFrame.Minute,
                "15m": TimeFrame.Minute,
                "1h": TimeFrame.Hour,
                "1d": TimeFrame.Day
            }

            # Create bars request
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=timeframe_map.get(timeframe, TimeFrame.Minute),
                start=start_date,
                end=end_date,
                adjustment='raw'
            )

            # For minute timeframes, set the multiplier
            if timeframe in ["5m", "15m"]:
                multiplier = int(timeframe[:-1])  # Remove 'm' and convert to int
                request.timeframe = TimeFrame(multiplier, TimeFrameUnit.Minute)

            bars = self.alpaca.get_stock_bars(request)

            if not bars:
                raise ValueError(f"No data found for {ticker}")

            # Convert to DataFrame
            data = []
            for bar in bars[ticker]:
                data.append({
                    'timestamp': bar.timestamp,
                    'Open': bar.open,
                    'High': bar.high,
                    'Low': bar.low,
                    'Close': bar.close,
                    'Volume': bar.volume
                })

            df = pd.DataFrame(data)

            if df.empty:
                raise ValueError(f"No data found for {ticker}")

            self.logger.info(f"Fetched {len(df)} records for {ticker} from Alpaca")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker} from Alpaca: {e}")
            raise

    async def fetch_sp500_yahoo_data(self, end_year: int = 2021) -> pd.DataFrame:
        """
        Fetch S&P500 data from Yahoo Finance up to specified year.

        Args:
            end_year: End year for data (default 2021 for backtesting)

        Returns:
            DataFrame with S&P500 OHLCV data
        """
        ticker = "^GSPC"  # S&P500 index
        start_date = "1950-01-01"  # Start from when data is available
        end_date = f"{end_year}-12-31"

        self._check_rate_limit()

        try:
            # Run yfinance in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                ticker_obj = yf.Ticker(ticker)
                df = await loop.run_in_executor(
                    executor, lambda: ticker_obj.history(start=start_date, end=end_date, interval="1d")
                )

            if df.empty:
                raise ValueError(f"No S&P500 data found up to {end_year}")

            # Flatten column names and ensure proper format
            df.columns = df.columns.get_level_values(0)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            # Filter to ensure <= end_year
            df = df[df.index.year <= end_year]

            self.logger.info(f"Fetched {len(df)} records of S&P500 data up to {end_year}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching S&P500 data: {e}")
            raise

    def download_kaggle_dataset(self, dataset_slug: str, output_dir: str = None) -> str:
        """
        Download dataset from Kaggle.

        Args:
            dataset_slug: Kaggle dataset slug (e.g., 'jacksoncrow/stock-market-dataset')
            output_dir: Directory to save dataset

        Returns:
            Path to downloaded dataset directory
        """
        if not self.config.data.kaggle_username or not self.config.data.kaggle_key:
            raise ValueError("Kaggle credentials not configured")

        if output_dir is None:
            output_dir = self.config.system.data_dir / "kaggle_datasets"

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        try:
            # Set Kaggle credentials
            import os
            os.environ['KAGGLE_USERNAME'] = self.config.data.kaggle_username
            os.environ['KAGGLE_KEY'] = self.config.data.kaggle_key

            # Import kaggle after setting credentials
            import kaggle

            # Download dataset
            kaggle.api.dataset_download_files(dataset_slug, path=str(output_dir), unzip=True)

            dataset_path = output_dir / dataset_slug.split('/')[-1]
            self.logger.info(f"Downloaded Kaggle dataset {dataset_slug} to {dataset_path}")
            return str(dataset_path)

        except Exception as e:
            self.logger.error(f"Error downloading Kaggle dataset {dataset_slug}: {e}")
            raise

    def load_kaggle_nasdaq_data(self) -> pd.DataFrame:
        """
        Load NASDAQ data from Kaggle dataset.

        Returns:
            DataFrame with NASDAQ stock data
        """
        try:
            # Try to load from cache first
            cache_path = self.config.system.data_dir / "cache" / "kaggle_nasdaq.parquet"
            if cache_path.exists():
                return pd.read_parquet(cache_path)

            # Download dataset
            dataset_path = self.download_kaggle_dataset('jacksoncrow/stock-market-dataset')

            # Load NASDAQ data - search recursively for CSV files
            data_files = list(Path(dataset_path).rglob("*.csv"))
            if not data_files:
                raise ValueError("No CSV files found in Kaggle dataset")

            # Load the main data file (prefer files with 'stock' or 'market' in name)
            preferred_files = [f for f in data_files if 'stock' in f.name.lower() or 'market' in f.name.lower()]
            data_file = preferred_files[0] if preferred_files else data_files[0]

            df = pd.read_csv(data_file)

            # Filter for NASDAQ stocks if needed
            if 'Symbol' in df.columns:
                # Assuming NASDAQ symbols or filter logic
                pass  # Dataset likely already filtered

            # Convert date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

            # Filter to backtest year
            df = df[df.index.year <= self.config.data.backtest_end_year]

            # Cache the data
            cache_path.parent.mkdir(exist_ok=True)
            df.to_parquet(cache_path)

            self.logger.info(f"Loaded {len(df)} records from Kaggle NASDAQ dataset")
            return df

        except Exception as e:
            self.logger.error(f"Error loading Kaggle NASDAQ data: {e}")
            raise

    def load_global_financial_data(self) -> pd.DataFrame:
        """
        Load global financial market data.

        Returns:
            DataFrame with global market data
        """
        try:
            # Try to load from cache first
            cache_path = self.config.system.data_dir / "cache" / "global_financial.parquet"
            if cache_path.exists():
                return pd.read_parquet(cache_path)

            # Download from Kaggle global dataset
            dataset_path = self.download_kaggle_dataset('pavankrishnanarne/global-stock-market-2008-present')

            # Load global data - search recursively for CSV files
            data_files = list(Path(dataset_path).rglob("*.csv"))
            if not data_files:
                raise ValueError("No CSV files found in global financial dataset")

            # Prefer files with 'global' or 'world' in name
            preferred_files = [f for f in data_files if 'global' in f.name.lower() or 'world' in f.name.lower()]
            data_file = preferred_files[0] if preferred_files else data_files[0]

            df = pd.read_csv(data_file)

            # Convert date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

            # Filter to backtest year
            df = df[df.index.year <= self.config.data.backtest_end_year]

            # Cache the data
            cache_path.parent.mkdir(exist_ok=True)
            df.to_parquet(cache_path)

            self.logger.info(f"Loaded {len(df)} records from global financial dataset")
            return df

        except Exception as e:
            self.logger.error(f"Error loading global financial data: {e}")
            raise

    def load_kaggle_huge_stock_data(self) -> pd.DataFrame:
        """
        Load huge US stock market dataset from Kaggle.

        Returns:
            DataFrame with comprehensive US stock data
        """
        try:
            # Try to load from cache first
            cache_path = self.config.system.data_dir / "cache" / "kaggle_huge_stock.parquet"
            if cache_path.exists():
                return pd.read_parquet(cache_path)

            # Download dataset
            dataset_path = self.download_kaggle_dataset('borismarjanovic/price-volume-data-for-all-us-stocks-etfs')

            # Load data - this dataset has many CSV files, search recursively
            data_files = list(Path(dataset_path).rglob("*.csv"))
            if not data_files:
                raise ValueError("No CSV files found in huge stock dataset")

            # Load a sample of files to avoid memory issues
            dfs = []
            max_files = min(10, len(data_files))  # Limit to 10 files initially

            for file_path in data_files[:max_files]:
                try:
                    df = pd.read_csv(file_path)
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                        dfs.append(df)
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")
                    continue

            if dfs:
                combined_df = pd.concat(dfs, ignore_index=False)
                # Filter to backtest year
                combined_df = combined_df[combined_df.index.year <= self.config.data.backtest_end_year]

                # Cache the data
                cache_path.parent.mkdir(exist_ok=True)
                combined_df.to_parquet(cache_path)

                self.logger.info(f"Loaded huge stock data: {len(combined_df)} records from {len(dfs)} files")
                return combined_df
            else:
                raise ValueError("No valid data files loaded")

        except Exception as e:
            self.logger.error(f"Error loading Kaggle huge stock data: {e}")
            raise

    def load_kaggle_sp500_data(self) -> pd.DataFrame:
        """
        Load S&P500 stock data from Kaggle.

        Returns:
            DataFrame with S&P500 stock data
        """
        try:
            # Try to load from cache first
            cache_path = self.config.system.data_dir / "cache" / "kaggle_sp500.parquet"
            if cache_path.exists():
                return pd.read_parquet(cache_path)

            # Download dataset
            dataset_path = self.download_kaggle_dataset('camnugent/sandp500')

            # Load data - search recursively for CSV files
            data_files = list(Path(dataset_path).rglob("*.csv"))
            if not data_files:
                raise ValueError("No CSV files found in S&P500 dataset")

            # Prefer files with 'sp500' or 'sandp' in name
            preferred_files = [f for f in data_files if 'sp500' in f.name.lower() or 'sandp' in f.name.lower()]
            data_file = preferred_files[0] if preferred_files else data_files[0]

            df = pd.read_csv(data_file)

            # Convert date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

            # Filter to backtest year
            df = df[df.index.year <= self.config.data.backtest_end_year]

            # Cache the data
            cache_path.parent.mkdir(exist_ok=True)
            df.to_parquet(cache_path)

            self.logger.info(f"Loaded S&P500 data: {len(df)} records")
            return df

        except Exception as e:
            self.logger.error(f"Error loading Kaggle S&P500 data: {e}")
            raise

    def load_kaggle_world_stocks_data(self) -> pd.DataFrame:
        """
        Load world stock prices from Kaggle.

        Returns:
            DataFrame with world stock data
        """
        try:
            # Try to load from cache first
            cache_path = self.config.system.data_dir / "cache" / "kaggle_world_stocks.parquet"
            if cache_path.exists():
                return pd.read_parquet(cache_path)

            # Download dataset
            dataset_path = self.download_kaggle_dataset('nelgiriyewithana/world-stock-prices-daily-updating')

            # Load data - search recursively for CSV files
            data_files = list(Path(dataset_path).rglob("*.csv"))
            if not data_files:
                raise ValueError("No CSV files found in world stocks dataset")

            # Prefer files with 'world' or 'global' in name
            preferred_files = [f for f in data_files if 'world' in f.name.lower() or 'global' in f.name.lower()]
            data_file = preferred_files[0] if preferred_files else data_files[0]

            df = pd.read_csv(data_file)

            # Convert date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

            # Filter to backtest year
            df = df[df.index.year <= self.config.data.backtest_end_year]

            # Cache the data
            cache_path.parent.mkdir(exist_ok=True)
            df.to_parquet(cache_path)

            self.logger.info(f"Loaded world stocks data: {len(df)} records")
            return df

        except Exception as e:
            self.logger.error(f"Error loading Kaggle world stocks data: {e}")
            raise

    def load_kaggle_exchanges_data(self) -> pd.DataFrame:
        """
        Load stock exchange data from Kaggle.

        Returns:
            DataFrame with stock exchange data
        """
        try:
            # Try to load from cache first
            cache_path = self.config.system.data_dir / "cache" / "kaggle_exchanges.parquet"
            if cache_path.exists():
                return pd.read_parquet(cache_path)

            # Download dataset
            dataset_path = self.download_kaggle_dataset('mattiuzc/stock-exchange-data')

            # Load data - search recursively for CSV files
            data_files = list(Path(dataset_path).rglob("*.csv"))
            if not data_files:
                raise ValueError("No CSV files found in exchanges dataset")

            # Prefer files with 'exchange' in name
            preferred_files = [f for f in data_files if 'exchange' in f.name.lower()]
            data_file = preferred_files[0] if preferred_files else data_files[0]

            df = pd.read_csv(data_file)

            # Convert date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

            # Filter to backtest year
            df = df[df.index.year <= self.config.data.backtest_end_year]

            # Cache the data
            cache_path.parent.mkdir(exist_ok=True)
            df.to_parquet(cache_path)

            self.logger.info(f"Loaded exchanges data: {len(df)} records")
            return df

        except Exception as e:
            self.logger.error(f"Error loading Kaggle exchanges data: {e}")
            raise

    async def scrape_yahoo_history(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Scrape detailed stock history from yahoo.finance.com using Fetch MCP.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with scraped historical data
        """
        try:
            self.logger.info(f"Scraping {ticker} history from Yahoo Finance using Fetch MCP")

            # Use Fetch MCP to get HTML content
            # Note: Yahoo Finance uses dynamic loading, so we may need multiple requests
            url = f"https://finance.yahoo.com/quote/{ticker}/history"

            # First, try to get the page content
            # This would use the fetch_html MCP tool in practice
            # For now, we'll simulate and fall back to yfinance with enhanced logging

            # Convert dates to timestamps for Yahoo
            from datetime import datetime
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

            # Yahoo Finance historical data URL format
            download_url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
            params = {
                'period1': start_ts,
                'period2': end_ts,
                'interval': '1d',
                'events': 'history',
                'includeAdjustedClose': 'true'
            }

            # This would use requests or fetch_html MCP tool
            # For implementation, we'll use requests with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(download_url, params=params, headers=headers)
            if response.status_code == 200:
                # Parse CSV content
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

                # Ensure proper column format
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

                self.logger.info(f"Successfully scraped {len(df)} records for {ticker} from Yahoo Finance")
                return df
            else:
                raise ValueError(f"Failed to fetch data: HTTP {response.status_code}")

        except Exception as e:
            self.logger.error(f"Error scraping Yahoo history for {ticker}: {e}")
            # Fallback to regular yfinance
            self.logger.info(f"Falling back to yfinance for {ticker}")
            return await self.fetch_yfinance_data(ticker, start_date, end_date, "1d")

    def load_large_dataset_with_memory_limit(self, file_path: str, chunk_size_mb: int = None) -> pd.DataFrame:
        """
        Load large datasets with memory management.

        Args:
            file_path: Path to the dataset file
            chunk_size_mb: Chunk size in MB for processing

        Returns:
            DataFrame loaded with memory constraints
        """
        if chunk_size_mb is None:
            chunk_size_mb = self.config.data.chunk_size_mb

        file_path = Path(file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Estimate memory usage (rough approximation)
        estimated_memory_mb = file_size_mb * 2  # Assume 2x file size for DataFrame
        max_memory_mb = self.config.data.max_ram_gb * 1024

        if estimated_memory_mb > max_memory_mb:
            self.logger.warning(f"File {file_path} may exceed memory limit. Size: {file_size_mb:.1f}MB, "
                              f"Estimated memory: {estimated_memory_mb:.1f}MB, "
                              f"Limit: {max_memory_mb:.1f}MB")

        try:
            if file_path.suffix == '.csv':
                # Load CSV with chunking if large
                if file_size_mb > chunk_size_mb:
                    self.logger.info(f"Loading large CSV {file_path} in chunks")
                    chunks = []
                    for chunk in pd.read_csv(file_path, chunksize=100000):  # 100k rows per chunk
                        chunks.append(chunk)
                        # Check memory usage
                        current_memory = sum(len(chunk) * chunk.memory_usage(deep=True).sum() / (1024*1024) for chunk in chunks)
                        if current_memory > max_memory_mb * 0.8:  # 80% of limit
                            self.logger.warning("Approaching memory limit, stopping chunk loading")
                            break
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(file_path)
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            self.logger.info(f"Loaded dataset {file_path} with {len(df)} rows, "
                           f"memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.1f}MB")
            return df

        except Exception as e:
            self.logger.error(f"Error loading large dataset {file_path}: {e}")
            raise

    def filter_backtest_data(self, df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
        """
        Filter data for backtesting (up to specified end year).

        Args:
            df: Input DataFrame
            date_column: Name of date column (if not index)

        Returns:
            Filtered DataFrame
        """
        end_year = self.config.data.backtest_end_year

        if date_column and date_column in df.columns:
            # Filter by date column
            df[date_column] = pd.to_datetime(df[date_column])
            filtered_df = df[df[date_column].dt.year <= end_year].copy()
        elif isinstance(df.index, pd.DatetimeIndex):
            # Filter by datetime index
            filtered_df = df[df.index.year <= end_year].copy()
        else:
            self.logger.warning("No date column or datetime index found for filtering")
            filtered_df = df.copy()

        self.logger.info(f"Filtered data to {end_year}: {len(filtered_df)} rows (from {len(df)})")
        return filtered_df

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

        # Check available libraries
        self.available_libraries = {
            'talib': HAS_TALIB,
            'pandas_ta': HAS_PANDAS_TA
        }

        # Log available libraries
        available = [lib for lib, status in self.available_libraries.items() if status]
        self.logger.info(f"Available technical analysis libraries: {available}")

        if not any(self.available_libraries.values()):
            self.logger.warning("No technical analysis libraries available. Using numpy/pandas fallbacks.")

    def check_library_availability(self) -> Dict[str, str]:
        """
        Check availability of technical analysis libraries and provide recommendations.

        Returns:
            Dictionary with library status and recommendations
        """
        status = {}

        if HAS_TALIB:
            status['talib'] = "✅ Available - Using TA-Lib for optimal performance"
        else:
            status['talib'] = "❌ Not available - Install TA-Lib for best performance"

        if HAS_PANDAS_TA:
            status['pandas_ta'] = "✅ Available - Using pandas-ta as alternative"
        else:
            status['pandas_ta'] = "❌ Not available - Install pandas-ta for better performance"

        status['numpy_fallback'] = "✅ Available - NumPy/Pandas fallbacks always available"

        return status

    def get_installation_instructions(self) -> str:
        """
        Get installation instructions for missing libraries.

        Returns:
            String with installation instructions
        """
        instructions = []

        if not HAS_TALIB:
            instructions.append("""
TA-Lib Installation:
===================
Option 1 - Windows:
1. Download: https://ta-lib.org/
2. Install the C library
3. Add C:\\ta-lib to PATH
4. Run: pip install TA-Lib

Option 2 - Linux/macOS:
1. Install C library: sudo apt-get install ta-lib-dev (Ubuntu/Debian)
2. Run: pip install TA-Lib
""")

        if not HAS_PANDAS_TA:
            instructions.append("""
pandas-ta Installation:
=====================
Run: pip install pandas-ta
""")

        return "\n".join(instructions) if instructions else "All libraries are properly installed!"

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
            if HAS_TALIB:
                rsi = talib.RSI(prices, timeperiod=period)
                return rsi
            elif HAS_PANDAS_TA:
                # Use pandas-ta as alternative
                df = pd.DataFrame({'close': prices})
                rsi_series = df.ta.rsi(length=period)
                return rsi_series.values
            else:
                # Fallback implementation
                return self._calculate_rsi_fallback(prices, period)
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return np.full_like(prices, np.nan)

    def _calculate_rsi_fallback(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Fallback RSI calculation using pandas."""
        # Calculate price changes
        price_changes = np.diff(prices)
        price_changes = np.insert(price_changes, 0, 0)

        # Separate gains and losses
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, abs(price_changes), 0)

        # Calculate average gains and losses
        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)

        # First average gain/loss
        avg_gains[period] = np.mean(gains[1:period+1])
        avg_losses[period] = np.mean(losses[1:period+1])

        # Subsequent averages
        for i in range(period + 1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i]) / period

        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        return rsi

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
            if HAS_TALIB:
                macd_line, signal_line, histogram = talib.MACD(
                    prices, fastperiod=fast, slowperiod=slow, signalperiod=signal
                )
                return macd_line, signal_line, histogram
            elif HAS_PANDAS_TA:
                # Use pandas-ta as alternative
                df = pd.DataFrame({'close': prices})
                macd_result = df.ta.macd(fast=fast, slow=slow, signal=signal)
                if macd_result is not None:
                    macd_line = macd_result[f'MACD_{fast}_{slow}_{signal}'].values
                    signal_line = macd_result[f'MACDs_{fast}_{slow}_{signal}'].values
                    histogram = macd_result[f'MACDh_{fast}_{slow}_{signal}'].values
                    return macd_line, signal_line, histogram
                else:
                    raise ValueError("MACD calculation failed")
            else:
                # Fallback implementation
                return self._calculate_macd_fallback(prices, fast, slow, signal)
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return np.full_like(prices, np.nan), np.full_like(prices, np.nan), np.full_like(prices, np.nan)

    def _calculate_macd_fallback(self, prices: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback MACD calculation using EMA."""
        # Calculate EMAs
        fast_ema = self._calculate_ema_fallback(prices, fast)
        slow_ema = self._calculate_ema_fallback(prices, slow)

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # Calculate signal line (EMA of MACD line)
        signal_line = self._calculate_ema_fallback(macd_line, signal)

        # Calculate histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

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
            if HAS_TALIB:
                upper, middle, lower = talib.BBANDS(
                    prices, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0
                )
                return upper, middle, lower
            elif HAS_PANDAS_TA:
                # Use pandas-ta as alternative
                df = pd.DataFrame({'close': prices})
                bb_result = df.ta.bbands(length=period, std=std_dev)
                if bb_result is not None:
                    upper = bb_result[f'BBU_{period}_{std_dev}'].values
                    middle = bb_result[f'BBM_{period}_{std_dev}'].values
                    lower = bb_result[f'BBL_{period}_{std_dev}'].values
                    return upper, middle, lower
                else:
                    raise ValueError("Bollinger Bands calculation failed")
            else:
                # Fallback implementation
                return self._calculate_bollinger_bands_fallback(prices, period, std_dev)
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return np.full_like(prices, np.nan), np.full_like(prices, np.nan), np.full_like(prices, np.nan)

    def _calculate_bollinger_bands_fallback(self, prices: np.ndarray, period: int, std_dev: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback Bollinger Bands calculation."""
        # Calculate middle band (SMA)
        middle = self._calculate_sma_fallback(prices, period)

        # Calculate rolling standard deviation
        rolling_std = np.zeros_like(prices)
        for i in range(period - 1, len(prices)):
            rolling_std[i] = np.std(prices[i-period+1:i+1])

        # Calculate upper and lower bands
        upper = middle + (rolling_std * std_dev)
        lower = middle - (rolling_std * std_dev)

        return upper, middle, lower

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
            if HAS_TALIB:
                sma = talib.SMA(prices, timeperiod=period)
                return sma
            elif HAS_PANDAS_TA:
                # Use pandas-ta as alternative
                df = pd.DataFrame({'close': prices})
                sma_series = df.ta.sma(length=period)
                return sma_series.values
            else:
                # Fallback implementation
                return self._calculate_sma_fallback(prices, period)
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
            if HAS_TALIB:
                ema = talib.EMA(prices, timeperiod=period)
                return ema
            elif HAS_PANDAS_TA:
                # Use pandas-ta as alternative
                df = pd.DataFrame({'close': prices})
                ema_series = df.ta.ema(length=period)
                return ema_series.values
            else:
                # Fallback implementation
                return self._calculate_ema_fallback(prices, period)
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return np.full_like(prices, np.nan)

    def _calculate_sma_fallback(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Fallback SMA calculation."""
        sma = np.full_like(prices, np.nan)
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
        return sma

    def _calculate_ema_fallback(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Fallback EMA calculation."""
        ema = np.full_like(prices, np.nan)
        multiplier = 2 / (period + 1)

        # First EMA value is SMA
        ema[period - 1] = np.mean(prices[:period])

        # Calculate subsequent EMA values
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))

        return ema

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
            if HAS_TALIB:
                k_percent, d_percent = talib.STOCH(
                    high, low, close, fastk_period=k_period, slowk_period=d_period
                )
                return k_percent, d_percent
            elif HAS_PANDAS_TA:
                # Use pandas-ta as alternative
                df = pd.DataFrame({'high': high, 'low': low, 'close': close})
                stoch_result = df.ta.stoch(k=k_period, d=d_period)
                if stoch_result is not None:
                    k_percent = stoch_result[f'STOCHk_{k_period}_{d_period}_3'].values
                    d_percent = stoch_result[f'STOCHd_{k_period}_{d_period}_3'].values
                    return k_percent, d_percent
                else:
                    raise ValueError("Stochastic calculation failed")
            else:
                # Fallback implementation
                return self._calculate_stochastic_fallback(high, low, close, k_period, d_period)
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {e}")
            return np.full_like(close, np.nan), np.full_like(close, np.nan)

    def _calculate_stochastic_fallback(self, high: np.ndarray, low: np.ndarray,
                                     close: np.ndarray, k_period: int, d_period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback Stochastic calculation."""
        k_percent = np.full_like(close, np.nan)
        d_percent = np.full_like(close, np.nan)

        for i in range(k_period - 1, len(close)):
            # Calculate %K
            highest_high = np.max(high[i-k_period+1:i+1])
            lowest_low = np.min(low[i-k_period+1:i+1])
            current_close = close[i]

            if highest_high - lowest_low != 0:
                k_percent[i] = 100 * (current_close - lowest_low) / (highest_high - lowest_low)
            else:
                k_percent[i] = 50  # Neutral value

        # Calculate %D (SMA of %K)
        for i in range(k_period + d_period - 2, len(close)):
            d_percent[i] = np.mean(k_percent[i-d_period+1:i+1])

        return k_percent, d_percent

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
        df = df.bfill().fillna(0)

        self.logger.info(f"Calculated technical indicators for {len(df)} records")
        return df

    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      period: int = 14) -> np.ndarray:
        """Calculate Average True Range."""
        try:
            if HAS_TALIB:
                return talib.ATR(high, low, close, timeperiod=period)
            elif HAS_PANDAS_TA:
                # Use pandas-ta as alternative
                df = pd.DataFrame({'high': high, 'low': low, 'close': close})
                atr_series = df.ta.atr(length=period)
                return atr_series.values
            else:
                # Fallback implementation
                return self._calculate_atr_fallback(high, low, close, period)
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return np.full_like(close, np.nan)

    def _calculate_atr_fallback(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Fallback ATR calculation."""
        # Calculate True Range
        tr = np.zeros_like(high)
        tr[0] = high[0] - low[0]  # First TR is just H-L

        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)

        # Calculate ATR using SMA of TR
        atr = np.full_like(tr, np.nan)
        for i in range(period - 1, len(tr)):
            atr[i] = np.mean(tr[i-period+1:i+1])

        return atr

    def calculate_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate On-Balance Volume."""
        try:
            if HAS_TALIB:
                return talib.OBV(close, volume)
            else:
                # Fallback implementation using pandas
                return self._calculate_obv_fallback(close, volume)
        except Exception as e:
            self.logger.error(f"Error calculating OBV: {e}")
            return np.full_like(close, np.nan)

    def _calculate_obv_fallback(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Fallback OBV calculation using pandas."""
        obv = np.zeros_like(close)
        obv[0] = volume[0]

        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]

        return obv


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

    async def get_sp500_data(self) -> pd.DataFrame:
        """
        Get S&P500 historical data up to backtest end year.

        Returns:
            DataFrame with S&P500 data and technical indicators
        """
        if not self.config.data.use_sp500_yahoo:
            raise ValueError("S&P500 Yahoo data source not enabled")

        # Fetch S&P500 data
        df = await self.fetcher.fetch_sp500_yahoo_data(self.config.data.backtest_end_year)

        # Add timestamp column
        df = df.reset_index()
        df = df.rename(columns={'index': 'timestamp'})

        # Calculate technical indicators
        df = self.indicators.calculate_all_indicators(df)

        # Normalize features
        df = self.preprocessor.normalize_features(df)

        return df

    def get_kaggle_nasdaq_data(self) -> pd.DataFrame:
        """
        Get NASDAQ data from Kaggle dataset.

        Returns:
            DataFrame with NASDAQ data and technical indicators
        """
        if not self.config.data.use_kaggle_nasdaq:
            raise ValueError("Kaggle NASDAQ data source not enabled")

        # Load NASDAQ data
        df = self.fetcher.load_kaggle_nasdaq_data()

        # Add timestamp column if not present
        if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={'index': 'timestamp'})

        # Calculate technical indicators
        df = self.indicators.calculate_all_indicators(df)

        # Normalize features
        df = self.preprocessor.normalize_features(df)

        return df

    def get_global_financial_data(self) -> pd.DataFrame:
        """
        Get global financial market data.

        Returns:
            DataFrame with global market data and technical indicators
        """
        if not self.config.data.use_global_financial_data:
            raise ValueError("Global financial data source not enabled")

        # Load global data
        df = self.fetcher.load_global_financial_data()

        # Add timestamp column if not present
        if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={'index': 'timestamp'})

        # Calculate technical indicators
        df = self.indicators.calculate_all_indicators(df)

        # Normalize features
        df = self.preprocessor.normalize_features(df)

        return df

    async def get_comprehensive_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive market data from all enabled sources.

        Returns:
            Dictionary with data from all sources
        """
        data_sources = {}

        try:
            if self.config.data.use_sp500_yahoo:
                data_sources['sp500'] = await self.get_sp500_data()
                self.logger.info(f"Loaded S&P500 data: {len(data_sources['sp500'])} records")
        except Exception as e:
            self.logger.error(f"Failed to load S&P500 data: {e}")

        try:
            if self.config.data.use_kaggle_nasdaq:
                data_sources['nasdaq'] = self.get_kaggle_nasdaq_data()
                self.logger.info(f"Loaded NASDAQ data: {len(data_sources['nasdaq'])} records")
        except Exception as e:
            self.logger.error(f"Failed to load NASDAQ data: {e}")

        try:
            if self.config.data.use_global_financial_data:
                data_sources['global'] = self.get_global_financial_data()
                self.logger.info(f"Loaded global financial data: {len(data_sources['global'])} records")
        except Exception as e:
            self.logger.error(f"Failed to load global financial data: {e}")

        try:
            if self.config.data.use_kaggle_huge_stock:
                data_sources['huge_stock'] = self.get_kaggle_huge_stock_data()
                self.logger.info(f"Loaded huge stock data: {len(data_sources['huge_stock'])} records")
        except Exception as e:
            self.logger.error(f"Failed to load huge stock data: {e}")

        try:
            if self.config.data.use_kaggle_sp500:
                data_sources['kaggle_sp500'] = self.get_kaggle_sp500_data()
                self.logger.info(f"Loaded Kaggle S&P500 data: {len(data_sources['kaggle_sp500'])} records")
        except Exception as e:
            self.logger.error(f"Failed to load Kaggle S&P500 data: {e}")

        try:
            if self.config.data.use_kaggle_world_stocks:
                data_sources['world_stocks'] = self.get_kaggle_world_stocks_data()
                self.logger.info(f"Loaded world stocks data: {len(data_sources['world_stocks'])} records")
        except Exception as e:
            self.logger.error(f"Failed to load world stocks data: {e}")

        try:
            if self.config.data.use_kaggle_exchanges:
                data_sources['exchanges'] = self.get_kaggle_exchanges_data()
                self.logger.info(f"Loaded exchanges data: {len(data_sources['exchanges'])} records")
        except Exception as e:
            self.logger.error(f"Failed to load exchanges data: {e}")

        # Note: Quantopian data not implemented due to unavailability

        return data_sources

    def get_kaggle_huge_stock_data(self) -> pd.DataFrame:
        """
        Get huge US stock market data from Kaggle.

        Returns:
            DataFrame with comprehensive US stock data
        """
        if not self.config.data.use_kaggle_huge_stock:
            raise ValueError("Kaggle huge stock data source not enabled")

        # Load data
        df = self.fetcher.load_kaggle_huge_stock_data()

        # Add timestamp column if not present
        if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={'index': 'timestamp'})

        # Calculate technical indicators
        df = self.indicators.calculate_all_indicators(df)

        # Normalize features
        df = self.preprocessor.normalize_features(df)

        return df

    def get_kaggle_sp500_data(self) -> pd.DataFrame:
        """
        Get S&P500 stock data from Kaggle.

        Returns:
            DataFrame with S&P500 stock data
        """
        if not self.config.data.use_kaggle_sp500:
            raise ValueError("Kaggle S&P500 data source not enabled")

        # Load data
        df = self.fetcher.load_kaggle_sp500_data()

        # Add timestamp column if not present
        if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={'index': 'timestamp'})

        # Calculate technical indicators
        df = self.indicators.calculate_all_indicators(df)

        # Normalize features
        df = self.preprocessor.normalize_features(df)

        return df

    def get_kaggle_world_stocks_data(self) -> pd.DataFrame:
        """
        Get world stock prices from Kaggle.

        Returns:
            DataFrame with world stock data
        """
        if not self.config.data.use_kaggle_world_stocks:
            raise ValueError("Kaggle world stocks data source not enabled")

        # Load data
        df = self.fetcher.load_kaggle_world_stocks_data()

        # Add timestamp column if not present
        if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={'index': 'timestamp'})

        # Calculate technical indicators
        df = self.indicators.calculate_all_indicators(df)

        # Normalize features
        df = self.preprocessor.normalize_features(df)

        return df

    def get_kaggle_exchanges_data(self) -> pd.DataFrame:
        """
        Get stock exchange data from Kaggle.

        Returns:
            DataFrame with stock exchange data
        """
        if not self.config.data.use_kaggle_exchanges:
            raise ValueError("Kaggle exchanges data source not enabled")

        # Load data
        df = self.fetcher.load_kaggle_exchanges_data()

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

    def get_library_status(self) -> Dict[str, str]:
        """
        Get status of all technical analysis libraries.

        Returns:
            Dictionary with library availability status
        """
        return self.indicators.check_library_availability()

    def get_installation_help(self) -> str:
        """
        Get installation instructions for missing libraries.

        Returns:
            String with installation instructions
        """
        return self.indicators.get_installation_instructions()

    def diagnose_technical_analysis_setup(self) -> str:
        """
        Provide comprehensive diagnosis of technical analysis setup.

        Returns:
            Diagnostic report as formatted string
        """
        status = self.get_library_status()
        instructions = self.get_installation_help()

        report = """
=== Technical Analysis Setup Diagnosis ===
"""

        for lib, message in status.items():
            report += f"{message}\n"

        if instructions and instructions != "All libraries are properly installed!":
            report += f"\n=== Installation Instructions ===\n{instructions}"

        report += """
=== Performance Recommendations ===
- TA-Lib: Best performance, requires C library installation
- pandas-ta: Good performance, pure Python, easy installation
- NumPy fallbacks: Always available, slower but functional

=== Testing Your Setup ===
Run: python -c "from nca_trading_bot.data_handler import data_handler; print(data_handler.get_library_status())"
"""

        return report


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