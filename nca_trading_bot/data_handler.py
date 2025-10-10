"""
Data handling module for NCA Trading Bot
Handles loading, preprocessing, and feature engineering of financial data
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from typing import Dict, List, Tuple, Optional, Union, Any
import jax.numpy as jnp
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .config import Config


class DataHandler:
    """Handles loading and preprocessing of financial data"""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = None
        self.feature_columns = None

    def load_kaggle_dataset(self, dataset_name: str, tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load dataset from Kaggle
        Args:
            dataset_name: Name of dataset (e.g., 'kaggle_stock_market')
            tickers: Specific tickers to load, if None load all
        Returns:
            Dictionary of DataFrames by ticker symbol
        """
        dataset_config = self.config.datasets.get(dataset_name)
        if not dataset_config:
            raise ValueError(f"Dataset {dataset_name} not configured")

        base_path = Path(dataset_config["path"])

        if dataset_name == "kaggle_stock_market":
            return self._load_kaggle_stock_market(base_path, tickers)
        elif dataset_name == "sp500_data":
            return self._load_sp500_dataset(base_path, tickers)
        elif dataset_name == "yahoo_finance":
            return self.load_yfinance_data(tickers or self.config.top_tickers)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _load_kaggle_stock_market(self, base_path: Path, tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load Kaggle stock market dataset with separate stocks/etfs folders"""
        data = {}

        # Load stocks
        stocks_path = base_path / "stocks"
        if stocks_path.exists():
            stock_files = list(stocks_path.glob("*.csv"))
            if tickers:
                stock_files = [f for f in stock_files if f.stem in tickers]

            for file_path in stock_files:
                try:
                    ticker = file_path.stem
                    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                    df = self._clean_and_validate_dataframe(df, ticker)
                    if df is not None:
                        data[ticker] = df
                except Exception as e:
                    print(f"Error loading {ticker}: {e}")
                    continue

        # Load ETFs
        etfs_path = base_path / "etfs"
        if etfs_path.exists():
            etf_files = list(etfs_path.glob("*.csv"))
            if tickers:
                etf_files = [f for f in etf_files if f.stem in tickers]

            for file_path in etf_files:
                try:
                    ticker = file_path.stem
                    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                    df = self._clean_and_validate_dataframe(df, ticker)
                    if df is not None:
                        data[ticker] = df
                except Exception as e:
                    print(f"Error loading ETF {ticker}: {e}")
                    continue

        return data

    def _load_sp500_dataset(self, base_path: Path, tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load S&P 500 dataset"""
        data = {}

        # First try the main consolidated file
        all_stocks_file = base_path / "all_stocks_5yr.csv"
        if all_stocks_file.exists():
            print("Loading S&P 500 data from all_stocks_5yr.csv...")
            try:
                df = pd.read_csv(all_stocks_file)

                # Filter by tickers if specified
                if tickers:
                    df = df[df['Name'].isin(tickers)]

                # Process each ticker
                for ticker in df['Name'].unique():
                    ticker_df = df[df['Name'] == ticker].copy()
                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
                    ticker_df.set_index('Date', inplace=True)
                    ticker_df.drop('Name', axis=1, inplace=True)

                    # Clean and validate
                    ticker_df = self._clean_and_validate_dataframe(ticker_df, ticker)
                    if ticker_df is not None:
                        data[ticker] = ticker_df

                print(f"Loaded {len(data)} tickers from S&P 500 dataset")
                return data

            except Exception as e:
                print(f"Error loading all_stocks_5yr.csv: {e}")

        # Fallback to individual stock files
        individual_stocks_path = base_path / "individual_stocks_5yr"
        if individual_stocks_path.exists():
            print("Loading S&P 500 data from individual stock files...")
            stock_files = list(individual_stocks_path.glob("*.csv"))

            if tickers:
                stock_files = [f for f in stock_files if f.stem in tickers]

            for file_path in stock_files[:100]:  # Limit to 100 files for performance
                try:
                    ticker = file_path.stem
                    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                    df = self._clean_and_validate_dataframe(df, ticker)
                    if df is not None:
                        data[ticker] = df
                except Exception as e:
                    print(f"Error loading {ticker}: {e}")
                    continue

        return data

    def load_yfinance_data(self, tickers: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load data using yfinance API
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        Returns:
            Dictionary of DataFrames by ticker symbol
        """
        start_date = start_date or self.config.data_start_date
        end_date = end_date or self.config.data_end_date

        data = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)

                if df.empty:
                    print(f"No data found for {ticker}")
                    continue

                df = self._clean_and_validate_dataframe(df, ticker)
                if df is not None:
                    data[ticker] = df

            except Exception as e:
                print(f"Error loading {ticker} from yfinance: {e}")
                continue

        return data

    def _clean_and_validate_dataframe(self, df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """Clean and validate DataFrame"""
        if df.empty:
            return None

        # Standardize column names
        df.columns = [col.strip().lower() for col in df.columns]

        # Required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            print(f"Missing required columns for {ticker}")
            return None

        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing values
        df = df.ffill().bfill()

        # Remove outliers (prices that changed by >50% in one day)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                returns = df[col].pct_change()
                outliers = abs(returns) > 0.5
                df.loc[outliers, col] = df[col].shift(1)[outliers]

        # Filter by date range
        end_date = pd.to_datetime(self.config.data_end_date)
        df = df.loc[:end_date]

        # Remove zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df = df[df[col] > 0]

        # Ensure minimum data length
        min_length = 252  # One year of trading days
        if len(df) < min_length:
            print(f"Insufficient data for {ticker}: {len(df)} days")
            return None

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to DataFrame
        Args:
            df: OHLCV DataFrame
        Returns:
            DataFrame with technical indicators added
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Momentum indicators
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['stoch_k'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        df['stoch_d'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch_signal()

        # Trend indicators
        for period in [5, 20, 50]:
            df[f'sma_{period}'] = ta.trend.SMAIndicator(df['close'], window=period).sma_indicator()

        for period in [12, 26]:
            df[f'ema_{period}'] = ta.trend.EMAIndicator(df['close'], window=period).ema_indicator()

        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
        df['macd_histogram'] = ta.trend.MACD(df['close']).macd_diff()

        # Volatility indicators
        df['bollinger_hband'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
        df['bollinger_lband'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
        df['bollinger_mband'] = ta.volatility.BollingerBands(df['close']).bollinger_mband()
        df['bollinger_width'] = (df['bollinger_hband'] - df['bollinger_lband']) / df['bollinger_mband']

        df['atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma_20'] = ta.volume.VolumeSMA(df['close'], df['volume']).volume_sma()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).vwap

        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']

        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()

        # Price position
        df['price_position_20'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min())

        return df

    def create_sequences(self, data: Dict[str, pd.DataFrame], sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for NCA training
        Args:
            data: Dictionary of DataFrames by ticker
            sequence_length: Length of input sequences
        Returns:
            Tuple of (sequences, targets)
        """
        sequence_length = sequence_length or self.config.data_sequence_length
        all_sequences = []
        all_targets = []

        for ticker, df in data.items():
            # Select features
            feature_cols = self._select_feature_columns(df)
            df_features = df[feature_cols].copy()

            # Normalize features
            df_features = self._normalize_features(df_features)

            # Create sequences
            for i in range(len(df_features) - sequence_length - self.config.data_prediction_horizon):
                sequence = df_features.iloc[i:i + sequence_length].values
                target = df_features.iloc[i + sequence_length:i + sequence_length + self.config.data_prediction_horizon]['close'].values

                all_sequences.append(sequence)
                all_targets.append(target)

        return np.array(all_sequences), np.array(all_targets)

    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Select relevant feature columns"""
        base_features = ['open', 'high', 'low', 'close', 'volume']
        tech_features = [
            'rsi_14', 'macd', 'macd_signal', 'bollinger_width', 'atr_14',
            'returns', 'log_returns', 'high_low_ratio', 'price_position_20',
            'returns_mean_20', 'returns_std_20'
        ]

        selected_cols = []
        for col in base_features + tech_features:
            if col in df.columns:
                selected_cols.append(col)

        self.feature_columns = selected_cols
        return selected_cols

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using MinMax scaling"""
        from sklearn.preprocessing import MinMaxScaler

        if self.scaler is None:
            self.scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(
                self.scaler.fit_transform(df),
                index=df.index,
                columns=df.columns
            )
        else:
            df_scaled = pd.DataFrame(
                self.scaler.transform(df),
                index=df.index,
                columns=df.columns
            )

        return df_scaled

    def create_nca_grid(self, sequence: np.ndarray, grid_size: Tuple[int, int] = None) -> jnp.ndarray:
        """
        Convert sequence to NCA grid format
        Args:
            sequence: Input sequence [sequence_length, features]
            grid_size: Target grid size (height, width)
        Returns:
            NCA grid [height, width, channels]
        """
        grid_size = grid_size or self.config.nca_grid_size
        height, width = grid_size

        # Pad or truncate sequence to fit grid
        sequence_length, features = sequence.shape
        grid_cells = height * width

        if sequence_length > grid_cells:
            # Take the most recent data
            sequence = sequence[-grid_cells:]
        elif sequence_length < grid_cells:
            # Pad with zeros
            padding = grid_cells - sequence_length
            sequence = np.pad(sequence, ((padding, 0), (0, 0)), mode='constant')

        # Reshape sequence to grid
        grid = sequence.reshape(height, width, features)

        # Pad or truncate features to match NCA channels
        if features < self.config.nca_channels:
            # Pad with zeros
            padding = self.config.nca_channels - features
            grid = np.pad(grid, ((0, 0), (0, 0), (0, padding)), mode='constant')
        elif features > self.config.nca_channels:
            # Truncate features
            grid = grid[:, :, :self.config.nca_channels]

        return jnp.array(grid, dtype=jnp.float32)

    def create_target_pattern(self, returns: np.ndarray, grid_size: Tuple[int, int] = None) -> jnp.ndarray:
        """
        Create target pattern for NCA training based on returns
        Args:
            returns: Future returns [prediction_horizon]
            grid_size: Target grid size
        Returns:
            Target pattern [height, width, channels]
        """
        grid_size = grid_size or self.config.nca_grid_size
        height, width = grid_size

        # Create pattern based on returns
        target_return = np.mean(returns)

        # Create RGB pattern based on return direction and magnitude
        if target_return > 0:
            # Green for positive returns
            intensity = min(abs(target_return) * 10, 1.0)  # Scale and cap
            rgb = [0, intensity, 0]
        else:
            # Red for negative returns
            intensity = min(abs(target_return) * 10, 1.0)
            rgb = [intensity, 0, 0]

        # Create target pattern
        target = np.zeros((height, width, self.config.nca_channels))
        target[:, :, :3] = rgb  # RGB channels
        target[:, :, 3] = 1.0   # Alpha channel (fully visible)

        return jnp.array(target, dtype=jnp.float32)

    def get_data_splits(self, sequences: np.ndarray, targets: np.ndarray) -> Tuple:
        """
        Split data into train, validation, and test sets
        Args:
            sequences: Input sequences
            targets: Target patterns
        Returns:
            Tuple of (train_x, train_y, val_x, val_y, test_x, test_y)
        """
        n_samples = len(sequences)
        val_size = int(n_samples * self.config.data_validation_split)
        test_size = int(n_samples * self.config.data_test_split)
        train_size = n_samples - val_size - test_size

        train_x = sequences[:train_size]
        train_y = targets[:train_size]

        val_x = sequences[train_size:train_size + val_size]
        val_y = targets[train_size:train_size + val_size]

        test_x = sequences[train_size + val_size:]
        test_y = targets[train_size + val_size:]

        return train_x, train_y, val_x, val_y, test_x, test_y

    def load_realtime_data(self, ticker: str, period: str = "1d") -> pd.DataFrame:
        """
        Load realtime data for a ticker
        Args:
            ticker: Ticker symbol
            period: Period for yfinance data
        Returns:
            DataFrame with latest data
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if not df.empty:
                df = self._clean_and_validate_dataframe(df, ticker)
                df = self.add_technical_indicators(df)
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            print(f"Error loading realtime data for {ticker}: {e}")
            return pd.DataFrame()

    def get_top_tickers(self, n: int = 10) -> List[str]:
        """
        Get top N tickers by market cap
        Args:
            n: Number of top tickers to return
        Returns:
            List of ticker symbols
        """
        return self.config.top_tickers[:n]

    def create_synthetic_data(self, n_samples: int = 1000, complexity: str = "simple") -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic financial data for testing
        Args:
            n_samples: Number of samples to generate
            complexity: 'simple', 'medium', or 'complex'
        Returns:
            Tuple of (sequences, targets)
        """
        sequence_length = self.config.data_sequence_length
        n_features = 10  # OHLCV + basic indicators

        sequences = []
        targets = []

        for i in range(n_samples):
            # Generate synthetic price series
            if complexity == "simple":
                # Simple random walk
                returns = np.random.normal(0.001, 0.02, sequence_length)
            elif complexity == "medium":
                # Add some trend and volatility clustering
                trend = np.sin(np.linspace(0, 4*np.pi, sequence_length)) * 0.01
                volatility = 0.01 + 0.02 * np.abs(np.sin(np.linspace(0, 2*np.pi, sequence_length)))
                returns = trend + np.random.normal(0, volatility)
            else:  # complex
                # Add multiple frequency components and regime changes
                trend1 = np.sin(np.linspace(0, 4*np.pi, sequence_length)) * 0.01
                trend2 = np.sin(np.linspace(0, 10*np.pi, sequence_length)) * 0.005
                regime_change = np.sin(np.linspace(0, 0.5*np.pi, sequence_length)) > 0
                volatility = np.where(regime_change, 0.03, 0.01)
                returns = trend1 + trend2 + np.random.normal(0, volatility)

            prices = 100 * np.exp(np.cumsum(returns))

            # Create synthetic features
            sequence = np.zeros((sequence_length, n_features))
            sequence[:, 0] = prices  # close price
            sequence[:, 1] = prices * (1 + np.random.normal(0, 0.001, sequence_length))  # high
            sequence[:, 2] = prices * (1 + np.random.normal(0, 0.001, sequence_length))  # low
            sequence[:, 3] = prices * (1 + np.random.normal(0, 0.001, sequence_length))  # open
            sequence[:, 4] = np.random.uniform(1000000, 10000000, sequence_length)  # volume

            # Add synthetic indicators
            sequence[:, 5] = self._synthetic_rsi(prices)
            sequence[:, 6] = self._synthetic_moving_average(prices, 20)
            sequence[:, 7] = self._synthetic_moving_average(prices, 50)
            sequence[:, 8] = np.roll(returns, 1)  # previous returns
            sequence[:, 9] = np.abs(returns)  # volatility

            sequences.append(sequence)

            # Create target based on future returns
            future_return = np.mean(returns[-5:])  # Average of last 5 returns
            targets.append(future_return)

        return np.array(sequences), np.array(targets)

    def _synthetic_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Generate synthetic RSI values"""
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Use pandas for rolling operations
        gains_series = pd.Series(gains)
        losses_series = pd.Series(losses)

        avg_gain = gains_series.rolling(window=period, min_periods=1).mean().bfill().fillna(0).values
        avg_loss = losses_series.rolling(window=period, min_periods=1).mean().bfill().fillna(0).values

        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _synthetic_moving_average(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Generate synthetic moving average"""
        return pd.Series(prices).rolling(window=period, min_periods=1).mean().bfill().fillna(prices).values

    def save_data_cache(self, data: Dict[str, pd.DataFrame], cache_path: str):
        """Save processed data to cache"""
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    def load_data_cache(self, cache_path: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load processed data from cache"""
        import pickle
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None