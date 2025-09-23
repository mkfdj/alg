#!/usr/bin/env python3
"""
Main CLI interface for NCA Trading Bot.

This module provides a comprehensive command-line interface for running
the Neural Cellular Automata trading system with support for backtesting,
live trading, training, and various utility functions.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import os

from config import get_config, ConfigManager
from data_handler import DataHandler, fetch_sample_data
from nca_model import create_nca_model, load_nca_model
from trader import create_trading_environment, create_trading_agent, TradingAgent
from trainer import TrainingManager
from utils import initialize_utils, LoggerUtils, PerformanceMonitor, cache


class NCACommandLineInterface:
    """
    Command-line interface for NCA Trading Bot.

    Provides comprehensive CLI functionality for all trading bot operations.
    """

    def __init__(self):
        """Initialize CLI interface."""
        self.config = get_config()
        self.config_manager = ConfigManager()
        self.data_handler = DataHandler()
        self.training_manager = TrainingManager(self.config)
        self.performance_monitor = PerformanceMonitor()

        # Initialize utilities
        initialize_utils(self.config)

        # Setup logging
        self.logger = LoggerUtils.setup_logger(
            'nca_cli',
            self.config.system.log_level,
            str(self.config.system.log_dir / 'cli.log')
        )

        # CLI state
        self.is_running = False

    def run(self):
        """Run the CLI interface."""
        parser = self._create_parser()
        args = parser.parse_args()

        # Set up signal handlers for graceful shutdown
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            # Execute command
            if hasattr(args, 'func'):
                args.func(args)
            else:
                parser.print_help()

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.logger.error(f"Error running command: {e}")
            sys.exit(1)

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with all commands."""
        parser = argparse.ArgumentParser(
            description="NCA Trading Bot - Neural Cellular Automata Trading System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Backtest with default settings
  python main.py backtest --tickers AAPL MSFT GOOGL --days 30

  # Train model with custom configuration
  python main.py train --config config.yaml --epochs 100

  # Start live trading
  python main.py live --symbols SPY QQQ --paper

  # Fetch and analyze data
  python main.py data --tickers AAPL MSFT --interval 1d

  # Show system status
  python main.py status
            """
        )

        # Global options
        parser.add_argument(
            '--config',
            type=str,
            help='Path to configuration file'
        )
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Set logging level'
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )

        # Create subparsers for commands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Backtest command
        self._add_backtest_parser(subparsers)

        # Train command
        self._add_train_parser(subparsers)

        # Live trading command
        self._add_live_parser(subparsers)

        # Data command
        self._add_data_parser(subparsers)

        # Model command
        self._add_model_parser(subparsers)

        # Status command
        self._add_status_parser(subparsers)

        # Config command
        self._add_config_parser(subparsers)

        return parser

    def _add_backtest_parser(self, subparsers):
        """Add backtest command parser."""
        parser = subparsers.add_parser(
            'backtest',
            help='Run backtesting simulation'
        )
        parser.set_defaults(func=self._run_backtest)

        parser.add_argument(
            '--tickers',
            nargs='+',
            default=self.config.data.tickers[:5],
            help='Stock tickers to backtest'
        )
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Number of days to backtest'
        )
        parser.add_argument(
            '--interval',
            choices=['1m', '5m', '15m', '1h', '1d'],
            default='1d',
            help='Data interval'
        )
        parser.add_argument(
            '--initial-balance',
            type=float,
            default=100000,
            help='Initial portfolio balance'
        )
        parser.add_argument(
            '--output',
            type=str,
            help='Output file for results'
        )

    def _add_train_parser(self, subparsers):
        """Add train command parser."""
        parser = subparsers.add_parser(
            'train',
            help='Train NCA model'
        )
        parser.set_defaults(func=self._run_train)

        parser.add_argument(
            '--mode',
            choices=['offline', 'live'],
            default='offline',
            help='Training mode'
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=100,
            help='Number of training epochs'
        )
        parser.add_argument(
            '--model-path',
            type=str,
            help='Path to save/load model'
        )
        parser.add_argument(
            '--symbols',
            nargs='+',
            help='Symbols for live training'
        )
        parser.add_argument(
            '--resume',
            action='store_true',
            help='Resume training from checkpoint'
        )

    def _add_live_parser(self, subparsers):
        """Add live trading command parser."""
        parser = subparsers.add_parser(
            'live',
            help='Start live trading'
        )
        parser.set_defaults(func=self._run_live)

        parser.add_argument(
            '--symbols',
            nargs='+',
            required=True,
            help='Stock symbols to trade'
        )
        parser.add_argument(
            '--paper',
            action='store_true',
            help='Use paper trading mode'
        )
        parser.add_argument(
            '--risk-limit',
            type=float,
            default=0.02,
            help='Maximum risk per trade'
        )
        parser.add_argument(
            '--max-position',
            type=float,
            default=0.1,
            help='Maximum position size'
        )

    def _add_data_parser(self, subparsers):
        """Add data command parser."""
        parser = subparsers.add_parser(
            'data',
            help='Data management operations'
        )
        parser.set_defaults(func=self._run_data)

        parser.add_argument(
            '--tickers',
            nargs='+',
            required=True,
            help='Stock tickers'
        )
        parser.add_argument(
            '--start-date',
            type=str,
            help='Start date (YYYY-MM-DD)'
        )
        parser.add_argument(
            '--end-date',
            type=str,
            help='End date (YYYY-MM-DD)'
        )
        parser.add_argument(
            '--interval',
            choices=['1m', '5m', '15m', '1h', '1d'],
            default='1d',
            help='Data interval'
        )
        parser.add_argument(
            '--output',
            type=str,
            help='Output file path'
        )
        parser.add_argument(
            '--analyze',
            action='store_true',
            help='Analyze fetched data'
        )

    def _add_model_parser(self, subparsers):
        """Add model command parser."""
        parser = subparsers.add_parser(
            'model',
            help='Model management operations'
        )
        parser.set_defaults(func=self._run_model)

        parser.add_argument(
            'action',
            choices=['create', 'load', 'save', 'info', 'list'],
            help='Model action'
        )
        parser.add_argument(
            '--model-path',
            type=str,
            help='Path to model file'
        )
        parser.add_argument(
            '--output-path',
            type=str,
            help='Output path for saved model'
        )

    def _add_status_parser(self, subparsers):
        """Add status command parser."""
        parser = subparsers.add_parser(
            'status',
            help='Show system status'
        )
        parser.set_defaults(func=self._run_status)

        parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed status'
        )

    def _add_config_parser(self, subparsers):
        """Add config command parser."""
        parser = subparsers.add_parser(
            'config',
            help='Configuration management'
        )
        parser.set_defaults(func=self._run_config)

        parser.add_argument(
            'action',
            choices=['show', 'save', 'load', 'reset'],
            help='Config action'
        )
        parser.add_argument(
            '--config-path',
            type=str,
            help='Path to configuration file'
        )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False

    def _run_backtest(self, args):
        """Run backtesting simulation."""
        self.logger.info("Starting backtest...")

        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)

        self.logger.info(f"Fetching data for {args.tickers} from {start_date.date()} to {end_date.date()}")

        try:
            data = asyncio.run(self.data_handler.get_multiple_tickers_data(
                args.tickers, start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'), args.interval
            ))

            if not data:
                self.logger.error("No data fetched for backtesting")
                return

            # Create model
            model = self.training_manager.create_model()

            # Create trading agent
            trading_agent = create_trading_agent(model, self.config)

            # Run backtest for each ticker
            results = {}
            for ticker, ticker_data in data.items():
                self.logger.info(f"Backtesting {ticker}...")

                # Create environment
                env = create_trading_environment(ticker_data, self.config)

                # Run simulation
                obs, info = env.reset()
                done = False
                total_reward = 0

                while not done:
                    # Get action from agent
                    decision = trading_agent.make_decision(ticker_data.iloc[:env.current_step + 60])

                    # Convert decision to environment action
                    if decision['action'] == 'buy':
                        action = [1, 5]  # Buy with 50% position
                    elif decision['action'] == 'sell':
                        action = [2, 5]  # Sell with 50% position
                    else:
                        action = [0, 0]  # Hold

                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward

                # Store results
                results[ticker] = {
                    'total_reward': total_reward,
                    'final_balance': info['portfolio_value'],
                    'total_trades': len(env.trades),
                    'win_rate': len([t for t in env.trades if t.get('pnl', 0) > 0]) / len(env.trades) if env.trades else 0
                }

                self.logger.info(f"{ticker} backtest completed: Reward={total_reward:.2f}, Balance=${info['portfolio_value']:.2f}")

            # Display results
            self._display_backtest_results(results)

            # Save results if requested
            if args.output:
                self._save_backtest_results(results, args.output)

        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")

    def _run_train(self, args):
        """Run training."""
        self.logger.info(f"Starting {args.mode} training...")

        try:
            if args.mode == 'offline':
                # Fetch training data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)  # 3 months of data

                self.logger.info("Fetching training data...")
                data = asyncio.run(self.data_handler.get_multiple_tickers_data(
                    self.config.data.tickers[:3],  # Use first 3 tickers for training
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    '1d'
                ))

                if not data:
                    self.logger.error("No training data available")
                    return

                # Combine data for training
                combined_data = list(data.values())[0]  # Use first ticker's data
                for ticker_data in list(data.values())[1:]:
                    combined_data = combined_data.append(ticker_data)

                # Start offline training
                asyncio.run(self.training_manager.train_offline(combined_data, args.epochs))

            elif args.mode == 'live':
                if not args.symbols:
                    self.logger.error("Symbols required for live training")
                    return

                # Create model and trading agent
                model = self.training_manager.create_model()
                trading_agent = create_trading_agent(model, self.config)

                # Start live training
                asyncio.run(self.training_manager.train_live(args.symbols, trading_agent))

        except Exception as e:
            self.logger.error(f"Training failed: {e}")

    def _run_live(self, args):
        """Run live trading."""
        self.logger.info("Starting live trading...")

        try:
            # Create model
            model = self.training_manager.create_model()

            # Create trading agent
            trading_agent = create_trading_agent(model, self.config)

            # Configure trading parameters
            if args.paper:
                self.logger.info("Using paper trading mode")

            # Start live trading engine
            from trader import LiveTradingEngine

            trading_engine = LiveTradingEngine(trading_agent, self.config)
            asyncio.run(trading_engine.start_trading(args.symbols))

        except Exception as e:
            self.logger.error(f"Live trading failed: {e}")

    def _run_data(self, args):
        """Run data operations."""
        self.logger.info(f"Fetching data for {args.tickers}...")

        try:
            # Set default dates if not provided
            if not args.start_date:
                args.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if not args.end_date:
                args.end_date = datetime.now().strftime('%Y-%m-%d')

            # Fetch data
            data = asyncio.run(self.data_handler.get_multiple_tickers_data(
                args.tickers, args.start_date, args.end_date, args.interval
            ))

            if not data:
                self.logger.error("No data fetched")
                return

            # Display data summary
            for ticker, ticker_data in data.items():
                self.logger.info(f"{ticker}: {len(ticker_data)} records, {len(ticker_data.columns)} features")

            # Analyze data if requested
            if args.analyze:
                self._analyze_data(data)

            # Save data if requested
            if args.output:
                self.data_handler.save_data(data, args.output)

        except Exception as e:
            self.logger.error(f"Data operation failed: {e}")

    def _run_model(self, args):
        """Run model operations."""
        try:
            if args.action == 'create':
                model = self.training_manager.create_model()
                self.logger.info("Model created successfully")

            elif args.action == 'load':
                if not args.model_path:
                    self.logger.error("Model path required for loading")
                    return
                model = self.training_manager.load_model(args.model_path)
                self.logger.info(f"Model loaded from {args.model_path}")

            elif args.action == 'save':
                if not args.output_path:
                    self.logger.error("Output path required for saving")
                    return
                self.training_manager.save_model(args.output_path)
                self.logger.info(f"Model saved to {args.output_path}")

            elif args.action == 'info':
                model = self.training_manager.model
                if model is None:
                    self.logger.error("No model loaded")
                    return

                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                self.logger.info("Model Information:")
                self.logger.info(f"  Total parameters: {total_params:,}")
                self.logger.info(f"  Trainable parameters: {trainable_params:,}")
                self.logger.info(f"  Device: {next(model.parameters()).device}")

            elif args.action == 'list':
                model_dir = self.config.system.model_dir
                if model_dir.exists():
                    models = list(model_dir.glob("*.pt"))
                    self.logger.info(f"Available models ({len(models)}):")
                    for model_file in models:
                        self.logger.info(f"  {model_file.name}")
                else:
                    self.logger.info("No models found")

        except Exception as e:
            self.logger.error(f"Model operation failed: {e}")

    def _run_status(self, args):
        """Show system status."""
        self.logger.info("NCA Trading Bot Status")
        self.logger.info("=" * 40)

        # System information
        self.logger.info("System Information:")
        self.logger.info(f"  Python version: {sys.version}")
        self.logger.info(f"  Working directory: {os.getcwd()}")
        self.logger.info(f"  Configuration file: {self.config_manager.config_path}")

        # Configuration status
        self.logger.info("Configuration Status:")
        config_summary = self.config_manager.get_config_summary()
        for section, params in config_summary.items():
            self.logger.info(f"  {section}: {params}")

        # Data status
        self.logger.info("Data Status:")
        data_dir = self.config.system.data_dir
        if data_dir.exists():
            data_files = list(data_dir.glob("*"))
            self.logger.info(f"  Data files: {len(data_files)}")
            for data_file in data_files[:5]:  # Show first 5 files
                self.logger.info(f"    {data_file.name}")
        else:
            self.logger.info("  No data directory found")

        # Model status
        self.logger.info("Model Status:")
        if self.training_manager.model is not None:
            total_params = sum(p.numel() for p in self.training_manager.model.parameters())
            self.logger.info(f"  Model loaded: Yes ({total_params:,} parameters)")
        else:
            self.logger.info("  Model loaded: No")

        # Performance metrics
        if args.detailed:
            self.logger.info("Performance Metrics:")
            system_metrics = self.performance_monitor.get_system_metrics()
            for key, value in system_metrics.items():
                self.logger.info(f"  {key}: {value}")

    def _run_config(self, args):
        """Run configuration operations."""
        global config
        try:
            if args.action == 'show':
                config_summary = self.config_manager.get_config_summary()
                print(json.dumps(config_summary, indent=2))

            elif args.action == 'save':
                config_path = args.config_path or "config.yaml"
                self.config_manager.save_config(config_path)
                self.logger.info(f"Configuration saved to {config_path}")

            elif args.action == 'load':
                if not args.config_path:
                    self.logger.error("Config path required for loading")
                    return
                config = ConfigManager(args.config_path)
                self.logger.info(f"Configuration loaded from {args.config_path}")

            elif args.action == 'reset':
                config = ConfigManager()
                self.logger.info("Configuration reset to defaults")

        except Exception as e:
            self.logger.error(f"Config operation failed: {e}")

    def _display_backtest_results(self, results: Dict):
        """Display backtest results."""
        self.logger.info("Backtest Results:")
        self.logger.info("-" * 40)

        total_reward = 0
        total_trades = 0

        for ticker, result in results.items():
            self.logger.info(f"{ticker}:")
            self.logger.info(f"  Total Reward: {result['total_reward']:.2f}")
            self.logger.info(f"  Final Balance: ${result['final_balance']:.2f}")
            self.logger.info(f"  Total Trades: {result['total_trades']}")
            self.logger.info(f"  Win Rate: {result['win_rate']:.2%}")

            total_reward += result['total_reward']
            total_trades += result['total_trades']

        self.logger.info("Summary:")
        self.logger.info(f"  Overall Reward: {total_reward:.2f}")
        self.logger.info(f"  Overall Trades: {total_trades}")
        self.logger.info(f"  Average Reward per Ticker: {total_reward / len(results):.2f}")

    def _save_backtest_results(self, results: Dict, output_path: str):
        """Save backtest results to file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'results': results,
                    'summary': {
                        'total_reward': sum(r['total_reward'] for r in results.values()),
                        'total_trades': sum(r['total_trades'] for r in results.values()),
                        'num_tickers': len(results)
                    }
                }, f, indent=2)

            self.logger.info(f"Backtest results saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def _analyze_data(self, data: Dict):
        """Analyze fetched data."""
        self.logger.info("Data Analysis:")
        self.logger.info("-" * 40)

        for ticker, ticker_data in data.items():
            self.logger.info(f"{ticker} Analysis:")

            # Basic statistics
            self.logger.info(f"  Records: {len(ticker_data)}")
            self.logger.info(f"  Date range: {ticker_data['timestamp'].min()} to {ticker_data['timestamp'].max()}")

            # Price statistics
            if 'Close' in ticker_data.columns:
                prices = ticker_data['Close']
                self.logger.info(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
                self.logger.info(f"  Mean price: ${prices.mean():.2f}")
                self.logger.info(f"  Volatility: {prices.std():.4f}")

            # Technical indicators
            if len(ticker_data) > 20:
                tech_columns = [col for col in ticker_data.columns
                               if col not in ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
                self.logger.info(f"  Technical indicators: {len(tech_columns)}")


def main():
    """Main entry point."""
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run CLI
    cli = NCACommandLineInterface()
    cli.run()


if __name__ == "__main__":
    main()