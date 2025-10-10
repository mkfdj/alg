"""
Utility functions for NCA Trading Bot
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit, lax
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .config import Config


def setup_tpu_environment(config: Config):
    """Setup TPU environment for distributed computing"""
    try:
        # Initialize JAX distributed system
        jax.distributed.initialize()

        # Create device mesh
        devices = jax.devices()
        mesh = jax.sharding.Mesh(jax.device_mesh((len(devices),)), ('tp',))

        print(f"TPU setup successful: {len(devices)} devices")
        return mesh, devices

    except Exception as e:
        print(f"TPU setup failed: {e}")
        return None, jax.devices()


def shard_array(array: jnp.ndarray, mesh) -> jnp.ndarray:
    """Shard array across TPU devices"""
    from jax.sharding import NamedSharding, PartitionSpec as P

    # Create sharding specification
    sharding = NamedSharding(mesh, P('tp', None))

    # Shard the array
    sharded_array = jax.device_put(array, sharding)

    return sharded_array


def create_time_series_features(prices: np.ndarray, sequence_length: int = 200) -> np.ndarray:
    """Create time series features from price data"""
    features = []

    for i in range(sequence_length, len(prices)):
        price_sequence = prices[i-sequence_length:i]

        # Returns
        returns = np.diff(price_sequence) / price_sequence[:-1]

        # Technical indicators
        features_sequence = []

        # Price features
        features_sequence.append(price_sequence[-1])  # Current price
        features_sequence.append(np.mean(price_sequence))  # Mean price
        features_sequence.append(np.std(price_sequence))   # Price volatility

        # Return features
        if len(returns) > 0:
            features_sequence.append(returns[-1])  # Latest return
            features_sequence.append(np.mean(returns))  # Mean return
            features_sequence.append(np.std(returns))   # Return volatility
            features_sequence.append(np.max(returns))   # Max return
            features_sequence.append(np.min(returns))   # Min return
        else:
            features_sequence.extend([0, 0, 0, 0, 0])

        # Momentum features
        if len(price_sequence) >= 20:
            ma_20 = np.mean(price_sequence[-20:])
            ma_50 = np.mean(price_sequence[-50:]) if len(price_sequence) >= 50 else ma_20
            features_sequence.append(price_sequence[-1] / ma_20 - 1)  # 20-day momentum
            features_sequence.append(price_sequence[-1] / ma_50 - 1)  # 50-day momentum
        else:
            features_sequence.extend([0, 0])

        # RSI calculation
        if len(returns) > 14:
            gains = returns[returns > 0]
            losses = -returns[returns < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
            features_sequence.append(rsi)
        else:
            features_sequence.append(50)  # Neutral RSI

        features.append(features_sequence)

    return np.array(features)


def calculate_portfolio_metrics(portfolio_values: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive portfolio metrics"""
    if len(portfolio_values) < 2:
        return {}

    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Basic metrics
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1

    # Risk metrics
    volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    # Drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)

    # Win rate
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0

    # Profit factor
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    profit_factor = np.sum(gains) / (-np.sum(losses)) if len(losses) > 0 and np.sum(losses) < 0 else 0

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }


def plot_training_results(metrics: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Results', fontsize=16)

    # Loss plot
    if 'loss' in metrics:
        axes[0, 0].plot(metrics['loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)

    # Reward plot
    if 'reward' in metrics:
        axes[0, 1].plot(metrics['reward'])
        axes[0, 1].set_title('Average Reward')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True)

    # Sharpe ratio plot
    if 'sharpe_ratio' in metrics:
        axes[1, 0].plot(metrics['sharpe_ratio'])
        axes[1, 0].set_title('Sharpe Ratio')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].grid(True)

    # Drawdown plot
    if 'max_drawdown' in metrics:
        axes[1, 1].plot(metrics['max_drawdown'])
        axes[1, 1].set_title('Maximum Drawdown')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Drawdown')
        axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_portfolio_performance(portfolio_values: np.ndarray, benchmark_values: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None):
    """Plot portfolio performance"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Portfolio value plot
    dates = pd.date_range(start='2020-01-01', periods=len(portfolio_values), freq='D')
    ax1.plot(dates, portfolio_values, label='Portfolio', linewidth=2)

    if benchmark_values is not None:
        ax1.plot(dates, benchmark_values, label='Benchmark', alpha=0.7)

    ax1.set_title('Portfolio Performance')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)

    # Drawdown plot
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak * 100
    ax2.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
    ax2.plot(dates, drawdown, color='red', linewidth=1)
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def create_trading_report(results: Dict[str, Any], config: Config, save_path: Optional[str] = None) -> str:
    """Create comprehensive trading report"""
    report = f"""
# NCA Trading Bot Performance Report

## Configuration
- Initial Balance: ${config.trading_initial_balance:,.2f}
- Training Period: {config.data_start_date} to {config.data_end_date}
- Top Tickers: {', '.join(config.top_tickers[:5])}
- NCA Grid Size: {config.nca_grid_size}
- NCA Evolution Steps: {config.nca_evolution_steps}
- RL Episodes: {results.get('num_episodes', 'N/A')}

## Performance Metrics
- **Total Return**: {results.get('avg_return', 0):.2f}%
- **Annualized Return**: {results.get('annualized_return', 0):.2f}%
- **Sharpe Ratio**: {results.get('avg_sharpe', 0):.2f}
- **Maximum Drawdown**: {results.get('avg_drawdown', 0):.2f}%
- **Win Rate**: {results.get('win_rate', 0):.2%}
- **Profit Factor**: {results.get('profit_factor', 0):.2f}
- **Volatility**: {results.get('volatility', 0):.2f}%

## Risk Analysis
- **Risk-Adjusted Return**: {results.get('sharpe_ratio', 0):.2f}
- **Maximum Loss**: {results.get('max_drawdown', 0):.2f}%
- **Recovery Factor**: {results.get('profit_factor', 0):.2f}

## Trading Statistics
- **Total Trades**: {results.get('num_trades', 'N/A')}
- **Average Trade Duration**: {results.get('avg_trade_duration', 'N/A')} days
- **Best Trade**: {results.get('best_trade', 'N/A')}%
- **Worst Trade**: {results.get('worst_trade', 'N/A')}%

## Model Performance
- **NCA Loss**: {results.get('nca_loss', 'N/A')}
- **PPO Loss**: {results.get('ppo_loss', 'N/A')}
- **Convergence Iterations**: {results.get('convergence_iterations', 'N/A')}

## Recommendations
"""

    # Add recommendations based on performance
    sharpe = results.get('avg_sharpe', 0)
    drawdown = results.get('avg_drawdown', 0)
    win_rate = results.get('win_rate', 0)

    if sharpe > 2.0:
        report += "- ✅ Excellent risk-adjusted returns\n"
    elif sharpe > 1.5:
        report += "- ✅ Good risk-adjusted returns\n"
    elif sharpe > 1.0:
        report += "- ⚠️  Moderate risk-adjusted returns\n"
    else:
        report += "- ❌ Poor risk-adjusted returns\n"

    if drawdown < 0.10:
        report += "- ✅ Low drawdown\n"
    elif drawdown < 0.20:
        report += "- ✅ Moderate drawdown\n"
    else:
        report += "- ❌ High drawdown\n"

    if win_rate > 0.60:
        report += "- ✅ High win rate\n"
    elif win_rate > 0.50:
        report += "- ✅ Good win rate\n"
    else:
        report += "- ❌ Low win rate\n"

    report += f"""
## Disclaimer
This report is generated by an AI trading system and is for informational purposes only.
Past performance does not guarantee future results. Trading involves substantial risk.

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {save_path}")

    return report


def validate_data_quality(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Validate data quality and return statistics"""
    validation_results = {
        'total_tickers': len(data),
        'tickers_with_issues': [],
        'data_quality_score': 0,
        'missing_data_percentage': 0,
        'date_range': None,
        'avg_data_points': 0
    }

    if not data:
        return validation_results

    all_dates = []
    total_missing = 0
    total_data_points = 0

    for ticker, df in data.items():
        if df.empty:
            validation_results['tickers_with_issues'].append(f"{ticker}: Empty DataFrame")
            continue

        # Check for missing data
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        total_missing += missing_pct
        total_data_points += len(df)

        # Check for price anomalies
        if 'close' in df.columns:
            price_changes = df['close'].pct_change()
            extreme_changes = (abs(price_changes) > 0.5).sum()
            if extreme_changes > 0:
                validation_results['tickers_with_issues'].append(
                    f"{ticker}: {extreme_changes} extreme price changes"
                )

        # Check for zero or negative prices
        if 'close' in df.columns:
            zero_prices = (df['close'] <= 0).sum()
            if zero_prices > 0:
                validation_results['tickers_with_issues'].append(
                    f"{ticker}: {zero_prices} zero/negative prices"
                )

        all_dates.extend(df.index.tolist())

    # Calculate overall statistics
    if all_dates:
        validation_results['date_range'] = (min(all_dates), max(all_dates))

    validation_results['missing_data_percentage'] = total_missing / len(data) if data else 0
    validation_results['avg_data_points'] = total_data_points / len(data) if data else 0

    # Calculate quality score (0-100)
    score = 100
    score -= validation_results['missing_data_percentage'] * 0.5
    score -= len(validation_results['tickers_with_issues']) * 2
    score -= max(0, (30 - validation_results['avg_data_points']) * 0.1)  # Penalty for short time series
    validation_results['data_quality_score'] = max(0, score)

    return validation_results


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    import logging

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )

    return logging.getLogger(__name__)


def create_checkpoint_directory(config: Config):
    """Create checkpoint directory if it doesn't exist"""
    import os
    checkpoint_dir = config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def save_model_checkpoint(model_state: Any, config: Config, iteration: int, metrics: Dict[str, float]):
    """Save model checkpoint with metadata"""
    import os
    import pickle

    checkpoint_dir = create_checkpoint_directory(config)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pkl")

    checkpoint_data = {
        'model_state': model_state,
        'config': config,
        'iteration': iteration,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_model_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load model checkpoint with metadata"""
    import pickle

    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)

    print(f"Checkpoint loaded: {checkpoint_path}")
    return checkpoint_data


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get path to latest checkpoint in directory"""
    import os
    import glob

    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_*.pkl")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if not checkpoint_files:
        return None

    # Sort by iteration number
    def get_iteration(file_path):
        filename = os.path.basename(file_path)
        return int(filename.split('_')[1].split('.')[0])

    latest_checkpoint = max(checkpoint_files, key=get_iteration)
    return latest_checkpoint


# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor training and inference performance"""

    def __init__(self):
        self.metrics = {}
        self.start_times = {}

    def start_timer(self, name: str):
        """Start timing a process"""
        self.start_times[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End timing and return duration"""
        if name not in self.start_times:
            return 0

        duration = time.time() - self.start_times[name]
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)

        return duration

    def get_average_time(self, name: str) -> float:
        """Get average time for a process"""
        if name not in self.metrics or not self.metrics[name]:
            return 0
        return np.mean(self.metrics[name])

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics"""
        summary = {}
        for name, times in self.metrics.items():
            if times:
                summary[f"{name}_avg"] = np.mean(times)
                summary[f"{name}_min"] = np.min(times)
                summary[f"{name}_max"] = np.max(times)
                summary[f"{name}_total"] = np.sum(times)
        return summary


import time