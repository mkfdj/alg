"""
Utilities module for NCA Trading Bot.

This module provides essential utilities including caching decorators,
logging helpers, DDP utilities, risk calculators, and performance monitoring.
"""

import logging
import time
import functools
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import hashlib
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.distributed as dist
from datetime import datetime, timedelta
import psutil
import GPUtil
import json

from .config import get_config


class Cache:
    """
    High-performance caching system with TTL and size limits.

    Provides in-memory and persistent caching with automatic cleanup.
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached items
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()

    def _generate_key(self, func: Callable, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key from function and arguments."""
        key_data = {
            'func': func.__name__,
            'args': args,
            'kwargs': kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached item or None if not found/expired
        """
        with self.lock:
            if key not in self.cache:
                return None

            item, timestamp = self.cache[key]
            if time.time() - timestamp > self.ttl:
                del self.cache[key]
                del self.access_times[key]
                return None

            self.access_times[key] = time.time()
            return item

    def set(self, key: str, value: Any):
        """
        Set item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            current_time = time.time()

            # Remove expired items
            expired_keys = [
                k for k, (_, timestamp) in self.cache.items()
                if current_time - timestamp > self.ttl
            ]
            for expired_key in expired_keys:
                del self.cache[expired_key]
                del self.access_times[expired_key]

            # Remove least recently used items if cache is full
            if len(self.cache) >= self.max_size:
                lru_key = min(self.access_times, key=self.access_times.get)
                del self.cache[lru_key]
                del self.access_times[lru_key]

            self.cache[key] = (value, current_time)
            self.access_times[key] = current_time

    def clear(self):
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

    def stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': len([k for k in self.cache.keys() if k in self.access_times]) / max(len(self.cache), 1)
            }


class PersistentCache:
    """
    Persistent caching system with file-based storage.

    Provides disk-based caching for large objects and results.
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize persistent cache.

        Args:
            cache_dir: Directory for cache files
        """
        self.config = get_config()
        self.cache_dir = Path(cache_dir) if cache_dir else self.config.system.data_dir / "persistent_cache"
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from persistent cache.

        Args:
            key: Cache key

        Returns:
            Cached item or None if not found
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)

            # Check TTL
            if time.time() - cached_data['timestamp'] > self.config.system.cache_ttl:
                cache_path.unlink()
                return None

            return cached_data['data']

        except Exception:
            return None

    def set(self, key: str, value: Any):
        """
        Set item in persistent cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        cache_path = self._get_cache_path(key)

        try:
            cache_data = {
                'data': value,
                'timestamp': time.time()
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

        except Exception as e:
            print(f"Failed to cache {key}: {e}")

    def clear(self):
        """Clear all cached files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


def cache_result(ttl: int = 3600, persistent: bool = False):
    """
    Decorator for caching function results.

    Args:
        ttl: Time to live in seconds
        persistent: Whether to use persistent caching

    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        # Use in-memory cache by default
        cache = Cache(ttl=ttl)

        if persistent:
            persistent_cache = PersistentCache()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = cache._generate_key(func, args, kwargs)

            # Try in-memory cache first
            result = cache.get(key)
            if result is not None:
                return result

            # Try persistent cache if enabled
            if persistent:
                result = persistent_cache.get(key)
                if result is not None:
                    cache.set(key, result)  # Update in-memory cache
                    return result

            # Compute result
            result = func(*args, **kwargs)

            # Cache result
            cache.set(key, result)
            if persistent:
                persistent_cache.set(key, result)

            return result

        return wrapper
    return decorator


def time_function(func: Callable):
    """
    Decorator for timing function execution.

    Args:
        func: Function to time

    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")

        return result
    return wrapper


def log_function(func: Callable):
    """
    Decorator for logging function calls.

    Args:
        func: Function to log

    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)

        logger.info(f"Calling {func.__name__}")
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            end_time = time.time()

            logger.info(f"{func.__name__} completed in {end_time - start_time:.4f} seconds")
            return result

        except Exception as e:
            end_time = time.time()
            logger.error(f"{func.__name__} failed after {end_time - start_time:.4f} seconds: {e}")
            raise

    return wrapper


class DDPUtils:
    """
    Utilities for Distributed Data Parallel training.

    Provides helper functions for DDP setup and management.
    """

    @staticmethod
    def setup_ddp(rank: int, world_size: int):
        """
        Set up DDP environment.

        Args:
            rank: Process rank
            world_size: Number of processes
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )

        torch.cuda.set_device(rank)

    @staticmethod
    def cleanup_ddp():
        """Clean up DDP environment."""
        dist.destroy_process_group()

    @staticmethod
    def is_main_process() -> bool:
        """Check if current process is main process."""
        return dist.get_rank() == 0 if dist.is_initialized() else True

    @staticmethod
    def barrier():
        """Synchronize all processes."""
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def all_reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        All-reduce tensor across all processes.

        Args:
            tensor: Tensor to reduce

        Returns:
            Reduced tensor
        """
        if not dist.is_initialized():
            return tensor

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
        return tensor


class RiskCalculator:
    """
    Advanced risk calculation utilities.

    Provides comprehensive risk assessment and position sizing calculations.
    """

    def __init__(self, config):
        """
        Initialize risk calculator.

        Args:
            config: Trading configuration
        """
        self.config = config

    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Array of historical returns
            confidence_level: Confidence level for VaR

        Returns:
            Value at Risk
        """
        if len(returns) == 0:
            return 0.0

        sorted_returns = np.sort(returns)
        var_index = int((1 - confidence_level) * len(sorted_returns))
        var_index = max(0, min(var_index, len(sorted_returns) - 1))

        return abs(sorted_returns[var_index])

    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).

        Args:
            returns: Array of historical returns
            confidence_level: Confidence level for CVaR

        Returns:
            Conditional Value at Risk
        """
        if len(returns) == 0:
            return 0.0

        var = self.calculate_var(returns, confidence_level)
        cvar_returns = returns[returns <= -var]  # Losses beyond VaR
        return abs(np.mean(cvar_returns)) if len(cvar_returns) > 0 else var

    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Args:
            portfolio_values: Array of portfolio values

        Returns:
            Maximum drawdown as percentage
        """
        if len(portfolio_values) == 0:
            return 0.0

        peak = portfolio_values[0]
        max_drawdown = 0

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def calculate_position_size(self, capital: float, risk_per_trade: float,
                              stop_loss_pct: float, entry_price: float) -> int:
        """
        Calculate optimal position size.

        Args:
            capital: Available capital
            risk_per_trade: Risk per trade as fraction
            stop_loss_pct: Stop loss percentage
            entry_price: Entry price

        Returns:
            Optimal position size in shares
        """
        risk_amount = capital * risk_per_trade
        risk_per_share = entry_price * stop_loss_pct
        position_size = risk_amount / risk_per_share

        return int(position_size)

    def calculate_kelly_position_size(self, win_probability: float,
                                    win_loss_ratio: float) -> float:
        """
        Calculate position size using Kelly criterion.

        Args:
            win_probability: Probability of winning trade
            win_loss_ratio: Ratio of average win to average loss

        Returns:
            Kelly position size as fraction of capital
        """
        kelly_fraction = win_probability - ((1 - win_probability) / win_loss_ratio)
        return max(0, kelly_fraction * self.config.trading.kelly_fraction)


class PerformanceMonitor:
    """
    Performance monitoring and system metrics collection.

    Provides real-time monitoring of system resources and trading performance.
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}
        self.start_time = time.time()

    def get_system_metrics(self) -> Dict[str, float]:
        """
        Get current system metrics.

        Returns:
            Dictionary with system metrics
        """
        metrics = {}

        # CPU metrics
        metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
        metrics['memory_percent'] = psutil.virtual_memory().percent

        # GPU metrics (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                metrics['gpu_utilization'] = gpu.load * 100
                metrics['gpu_memory_percent'] = (gpu.memoryUsed / gpu.memoryTotal) * 100
        except:
            metrics['gpu_utilization'] = 0
            metrics['gpu_memory_percent'] = 0

        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics['disk_percent'] = disk.percent

        # Network metrics
        network = psutil.net_io_counters()
        metrics['network_sent'] = network.bytes_sent
        metrics['network_recv'] = network.bytes_recv

        return metrics

    def get_trading_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """
        Get trading performance metrics.

        Args:
            trades: List of trade records

        Returns:
            Dictionary with trading metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }

        # Calculate metrics
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        total_profit = sum(t.get('pnl', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Calculate returns for Sharpe ratio
        returns = [t.get('pnl', 0) for t in trades]
        sharpe_ratio = self._calculate_sharpe_ratio(returns)

        # Calculate drawdown
        cumulative = np.cumsum(returns)
        max_drawdown = self._calculate_max_drawdown(cumulative)

        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_profit': total_profit,
            'total_loss': total_loss
        }

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio for returns.

        Args:
            returns: List of returns
            risk_free_rate: Risk-free rate

        Returns:
            Sharpe ratio
        """
        if not returns or np.std(returns) == 0:
            return 0.0

        excess_returns = np.array(returns) - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Args:
            cumulative_returns: Cumulative returns array

        Returns:
            Maximum drawdown as percentage
        """
        if len(cumulative_returns) == 0:
            return 0.0

        peak = cumulative_returns[0]
        max_drawdown = 0

        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / (peak + 1e-8)  # Avoid division by zero
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics to appropriate destinations.

        Args:
            metrics: Metrics dictionary
            step: Training step (optional)
        """
        # Update internal metrics
        self.metrics.update(metrics)
        self.metrics['timestamp'] = time.time()

        # Print to console
        if step is not None:
            print(f"Step {step}: {metrics}")
        else:
            print(f"Metrics: {metrics}")


class LoggerUtils:
    """
    Logging utilities for consistent logging across the application.

    Provides structured logging with different levels and formats.
    """

    @staticmethod
    def setup_logger(name: str, level: str = "INFO", log_file: str = None) -> logging.Logger:
        """
        Set up logger with consistent configuration.

        Args:
            name: Logger name
            level: Logging level
            log_file: Log file path

        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        if logger.handlers:
            return logger

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))

        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        console_handler.setFormatter(formatter)
        if log_file:
            file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        if log_file:
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def log_trading_action(action: str, symbol: str, quantity: int, price: float):
        """
        Log trading action.

        Args:
            action: Trading action (buy/sell)
            symbol: Stock symbol
            quantity: Quantity traded
            price: Trade price
        """
        logger = logging.getLogger('trading')
        logger.info(f"{action.upper()} {quantity} {symbol} @ ${price:.2f}")

    @staticmethod
    def log_performance_metrics(metrics: Dict[str, float]):
        """
        Log performance metrics.

        Args:
            metrics: Performance metrics dictionary
        """
        logger = logging.getLogger('performance')
        logger.info(f"Performance: {metrics}")

    @staticmethod
    def log_error(error: Exception, context: str = ""):
        """
        Log error with context.

        Args:
            error: Exception object
            context: Error context
        """
        logger = logging.getLogger('errors')
        logger.error(f"Error {context}: {str(error)}")


class DataUtils:
    """
    Data processing and manipulation utilities.

    Provides helper functions for data cleaning, transformation, and analysis.
    """

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(0)

        # Remove outliers (optional)
        # df = DataUtils.remove_outliers(df)

        return df

    @staticmethod
    def normalize_data(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Normalize DataFrame columns.

        Args:
            df: Input DataFrame
            columns: Columns to normalize

        Returns:
            Normalized DataFrame
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for column in columns:
            if column in df.columns:
                df[column] = (df[column] - df[column].mean()) / df[column].std()

        return df

    @staticmethod
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """
        Calculate returns from price series.

        Args:
            prices: Price array

        Returns:
            Returns array
        """
        return np.diff(prices) / prices[:-1]

    @staticmethod
    def calculate_rolling_metrics(data: np.ndarray, window: int = 20) -> Dict[str, np.ndarray]:
        """
        Calculate rolling metrics.

        Args:
            data: Input data array
            window: Rolling window size

        Returns:
            Dictionary with rolling metrics
        """
        return {
            'mean': pd.Series(data).rolling(window).mean().values,
            'std': pd.Series(data).rolling(window).std().values,
            'min': pd.Series(data).rolling(window).min().values,
            'max': pd.Series(data).rolling(window).max().values
        }


# Global utility instances
cache = Cache()
persistent_cache = PersistentCache()
risk_calculator = None
performance_monitor = PerformanceMonitor()


def initialize_utils(config):
    """Initialize global utility instances."""
    global risk_calculator
    risk_calculator = RiskCalculator(config)


if __name__ == "__main__":
    # Example usage
    print("NCA Trading Bot - Utils Module Demo")
    print("=" * 40)

    # Test caching
    @cache_result(ttl=10)
    def expensive_computation(x, y):
        time.sleep(1)  # Simulate expensive operation
        return x + y

    print("Testing caching...")
    result1 = expensive_computation(5, 3)
    result2 = expensive_computation(5, 3)  # Should be cached
    print(f"Results: {result1}, {result2}")

    # Test risk calculator
    config = get_config()
    risk_calc = RiskCalculator(config)

    returns = np.random.normal(0.001, 0.02, 1000)
    var = risk_calc.calculate_var(returns)
    sharpe = risk_calc.calculate_sharpe_ratio(returns)

    print(f"VaR (95%): {var:.4f}")
    print(f"Sharpe ratio: {sharpe:.4f}")

    # Test performance monitor
    trades = [
        {'pnl': 100}, {'pnl': -50}, {'pnl': 75}, {'pnl': -25}, {'pnl': 200}
    ]

    metrics = performance_monitor.get_trading_metrics(trades)
    print(f"Trading metrics: {metrics}")

    print("Utils module demo completed successfully!")