"""
Main application entry point for NCA Trading Bot
"""

import argparse
import os
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from nca_trading_bot import (
    Config, AdaptiveNCA, DataHandler, TradingEnvironment,
    PPOAgent, TradingBot, CombinedTrainer
)


def setup_jax_environment(config: Config):
    """Setup JAX environment for TPU/GPU/CPU"""
    print("Setting up JAX environment...")

    # Configure JAX
    jax.config.update("jax_enable_x64", config.jax_enable_x64)
    jax.config.update("jax_platform_name", config.jax_platform)
    jax.config.update("jax_debug_nans", config.jax_debug_nans)

    # Set memory fraction
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config.jax_memory_fraction)

    # Check available devices
    devices = jax.devices()
    print(f"Available devices: {len(devices)} x {devices[0].device_kind}")

    if config.jax_platform == "tpu":
        print(f"TPU configuration: {config.tpu_device_count} chips")
        # Initialize distributed training if needed
        try:
            jax.distributed.initialize()
            print("Distributed training initialized")
        except Exception as e:
            print(f"Distributed training initialization failed: {e}")

    return devices


def load_data(config: Config, datasets: Optional[List[str]] = None) -> dict:
    """Load training data"""
    print("Loading training data...")

    data_handler = DataHandler(config)
    data = {}

    if datasets is None:
        datasets = ["kaggle_stock_market", "yahoo_finance"]

    for dataset_name in datasets:
        try:
            print(f"Loading dataset: {dataset_name}")
            dataset_data = data_handler.load_kaggle_dataset(dataset_name)
            data.update(dataset_data)
            print(f"Loaded {len(dataset_data)} tickers from {dataset_name}")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")

    # Add technical indicators
    print("Adding technical indicators...")
    for ticker in data:
        try:
            data[ticker] = data_handler.add_technical_indicators(data[ticker])
        except Exception as e:
            print(f"Error adding indicators for {ticker}: {e}")

    print(f"Total data loaded: {len(data)} tickers")
    return data, data_handler


def run_training(config: Config, args):
    """Run training pipeline"""
    print("=== NCA Trading Bot Training ===")

    # Setup environment
    devices = setup_jax_environment(config)

    # Validate configuration
    if not config.validate():
        print("Configuration validation failed!")
        return

    # Load data
    data, data_handler = load_data(config, args.datasets)

    if not data:
        print("No data loaded. Please check dataset configuration.")
        return

    # Create combined trainer
    trainer = CombinedTrainer(config)

    # Run training
    trainer.train(
        nca_iterations=args.nca_iterations,
        ppo_iterations=args.ppo_iterations
    )

    # Evaluate final model
    results = trainer.evaluate(num_episodes=args.eval_episodes)
    print("\n=== Final Evaluation Results ===")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    print("Training completed!")


def run_backtesting(config: Config, args):
    """Run backtesting with trained model"""
    print("=== NCA Trading Bot Backtesting ===")

    # Setup environment
    setup_jax_environment(config)

    # Load data
    data, data_handler = load_data(config, args.datasets)

    # Create trading bot
    bot = TradingBot(config)

    # Load checkpoint if specified
    if args.checkpoint:
        try:
            bot.ppo_trainer.load_checkpoint(args.checkpoint)
            print(f"Loaded checkpoint: {args.checkpoint}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # Run backtesting
    results = bot.evaluate(num_episodes=args.eval_episodes)

    print("\n=== Backtesting Results ===")
    print(f"Average Return: {results['avg_return']:.2f}%")
    print(f"Average Sharpe Ratio: {results['avg_sharpe']:.2f}")
    print(f"Average Max Drawdown: {results['avg_drawdown']:.2f}")
    print(f"Win Rate: {results.get('win_rate', 0):.2%}")


def run_live_trading(config: Config, args):
    """Run live trading"""
    print("=== NCA Trading Bot Live Trading ===")

    # Safety check for live trading
    if not args.paper_mode:
        confirm = input("⚠️  WARNING: This will start LIVE trading with REAL money!\n"
                       "Type 'I UNDERSTAND THE RISKS' to continue: ")
        if confirm != "I UNDERSTAND THE RISKS":
            print("Live trading cancelled for your safety.")
            return

    # Setup environment
    setup_jax_environment(config)

    # Load checkpoint
    if not args.checkpoint:
        print("Error: Checkpoint required for live trading")
        return

    # Create and configure trading bot
    bot = TradingBot(config)

    try:
        bot.ppo_trainer.load_checkpoint(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Start live trading
    bot.start_live_trading(paper_mode=args.paper_mode)


def run_data_analysis(config: Config, args):
    """Run data analysis and visualization"""
    print("=== Data Analysis ===")

    # Load data
    data, data_handler = load_data(config, args.datasets)

    if not data:
        print("No data loaded")
        return

    # Print data statistics
    print(f"\nData Statistics:")
    print(f"Total tickers: {len(data)}")

    total_data_points = sum(len(df) for df in data.values())
    print(f"Total data points: {total_data_points:,}")

    # Date range
    all_dates = []
    for ticker, df in data.items():
        all_dates.extend(df.index.tolist())

    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        print(f"Date range: {min_date} to {max_date}")

    # Sample tickers info
    print(f"\nSample tickers:")
    for ticker in list(data.keys())[:5]:
        df = data[ticker]
        print(f"  {ticker}: {len(df)} days, "
              f"${df['close'].iloc[-1]:.2f} (latest), "
              f"{((df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100):.2f}% return")

    # Create sequences for NCA
    sequences, targets = data_handler.create_sequences(data)
    print(f"\nNCA Sequences:")
    print(f"  Number of sequences: {len(sequences):,}")
    print(f"  Sequence length: {config.data_sequence_length}")
    print(f"  Features per sequence: {sequences.shape[2] if len(sequences) > 0 else 0}")

    print("Data analysis completed!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="NCA Trading Bot")
    parser.add_argument("--mode", choices=["train", "backtest", "live", "analyze"],
                       default="train", help="Running mode")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--datasets", nargs="+",
                       default=["kaggle_stock_market", "yahoo_finance"],
                       help="Datasets to use")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to trade")
    parser.add_argument("--nca-iterations", type=int, default=1000,
                       help="Number of NCA training iterations")
    parser.add_argument("--ppo-iterations", type=int, default=1000,
                       help="Number of PPO training iterations")
    parser.add_argument("--eval-episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path to load")
    parser.add_argument("--paper-mode", action="store_true", default=True,
                       help="Use paper trading (default)")
    parser.add_argument("--live-mode", action="store_true",
                       help="Use live trading (requires explicit confirmation)")

    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Override with command line arguments
    if args.tickers:
        config.top_tickers = args.tickers

    if args.live_mode:
        args.paper_mode = False

    # Set environment variables for API keys
    if not os.getenv("ALPACA_PAPER_API_KEY"):
        print("Warning: ALPACA_PAPER_API_KEY environment variable not set")
        print("Set it using: export ALPACA_PAPER_API_KEY='your_key'")

    if not os.getenv("ALPACA_PAPER_SECRET_KEY"):
        print("Warning: ALPACA_PAPER_SECRET_KEY environment variable not set")
        print("Set it using: export ALPACA_PAPER_SECRET_KEY='your_secret'")

    try:
        if args.mode == "train":
            run_training(config, args)
        elif args.mode == "backtest":
            run_backtesting(config, args)
        elif args.mode == "live":
            run_live_trading(config, args)
        elif args.mode == "analyze":
            run_data_analysis(config, args)
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()