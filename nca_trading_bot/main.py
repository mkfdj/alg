"""
NCA Trading Bot Main Entry Point

This module provides the main entry point for the Neural Cellular Automata
trading bot, including command-line interface, configuration management,
and training/inference orchestration.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import ConfigManager, get_config, reload_config, detect_tpu_availability, get_tpu_device_count
from data_handler import DataHandler
from nca_model import create_nca_model, load_nca_model, save_nca_model
from trainer import TrainingManager, PPOTrainer, JAXPPOTrainer, RealTimeStreamingTrainer
from trader import TradingAgent, TradingEnvironment
from utils import PerformanceMonitor, RiskCalculator, LoggerUtils

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nca_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NCATradingBot:
    """
    Main NCA Trading Bot class.

    Orchestrates all components of the trading system including data handling,
    model training, and live trading.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize NCA Trading Bot.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = ConfigManager(config_path)
        self.logger = LoggerUtils.setup_logger('NCA_Trading_Bot', self.config.system.log_level)

        # Initialize components
        self.data_handler = DataHandler()
        self.model = None
        self.trainer = None
        self.trading_agent = None
        self.performance_monitor = PerformanceMonitor()
        self.risk_calculator = RiskCalculator(self.config)

        # Training state
        self.is_training = False
        self.is_trading = False

        # Initialize hardware
        self._initialize_hardware()

        self.logger.info("NCA Trading Bot initialized")

    def _initialize_hardware(self):
        """Initialize hardware configuration and optimizations."""
        # Detect available hardware
        device = self.config.system.device
        if device == "auto":
            if detect_tpu_availability():
                device = "tpu"
                self.config.system.device = "tpu"
                self.logger.info(f"TPU detected with {get_tpu_device_count()} cores")
            elif torch.cuda.is_available():
                device = "cuda"
                self.config.system.device = "cuda"
                self.logger.info(f"CUDA detected with {torch.cuda.device_count()} GPUs")
            else:
                device = "cpu"
                self.config.system.device = "cpu"
                self.logger.info("Using CPU for computation")

        # Set up mixed precision if available
        if device == "cuda" and torch.cuda.is_available():
            self.config.training.use_mixed_precision = True  # TPU supports bfloat16
            self.logger.info("Mixed precision enabled for CUDA")
        elif device == "tpu":
            self.config.training.use_mixed_precision = True  # TPU supports bfloat16
            self.logger.info("Mixed precision enabled for TPU")

    def create_model(self, adaptive: bool = False, load_path: Optional[str] = None):
        """
        Create or load NCA model.

        Args:
            adaptive: Whether to create an adaptive model
            load_path: Path to load model from
        """
        if load_path and os.path.exists(load_path):
            self.logger.info(f"Loading model from {load_path}")
            self.model = load_nca_model(load_path, self.config)
        else:
            self.logger.info("Creating new NCA model")
            self.model = create_nca_model(self.config, adaptive=adaptive)

        self.logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")

    def setup_trainer(self):
        """Set up training manager."""
        self.trainer = TrainingManager(self.config)
        self.trainer.create_model(adaptive=True)
        self.model = self.trainer.model

        # Initialize JAX PPO for real-time learning
        self.trainer.initialize_jax_ppo(
            observation_dim=self.config.data.sequence_length,
            action_dim=3  # buy, hold, sell
        )

        self.logger.info("Training manager initialized")

    def setup_trading_agent(self):
        """Set up trading agent."""
        if self.model is None:
            raise ValueError("Model must be created before setting up trading agent")

        self.trading_agent = TradingAgent(self.model, self.config)
        self.logger.info("Trading agent initialized")

    async def train_offline(self, data_path: Optional[str] = None, num_epochs: int = 100):
        """
        Perform offline training on historical data.

        Args:
            data_path: Path to training data
            num_epochs: Number of training epochs
        """
        if self.trainer is None:
            self.setup_trainer()

        # Load training data
        if data_path:
            data = pd.read_csv(data_path)
        else:
            # Fetch data using data handler
            data = await self.data_handler.get_multiple_tickers_data(
                self.config.data.tickers,
                start_date='2020-01-01',
                end_date='2023-01-01'
            )

        self.logger.info(f"Loaded training data with shape {data.shape}")

        # Start offline training
        self.is_training = True
        await self.trainer.train_offline(data, num_epochs)
        self.is_training = False

        self.logger.info("Offline training completed")

    async def train_streaming(self, symbols: List[str]):
        """
        Perform real-time streaming training.

        Args:
            symbols: List of symbols to stream and train on
        """
        if self.trainer is None:
            self.setup_trainer()

        self.logger.info(f"Starting streaming training for symbols: {symbols}")

        # Start streaming training
        self.is_training = True
        await self.trainer.train_streaming(symbols)
        self.is_training = False

        self.logger.info("Streaming training completed")

    async def trade_live(self, symbols: List[str], duration_minutes: int = 60):
        """
        Perform live trading.

        Args:
            symbols: List of symbols to trade
            duration_minutes: Duration of trading session in minutes
        """
        if self.trading_agent is None:
            self.setup_trading_agent()

        self.logger.info(f"Starting live trading for symbols: {symbols}")

        # Start live trading
        self.is_trading = True
        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        while datetime.now() < end_time and self.is_trading:
            try:
                # Make trading decisions for each symbol
                for symbol in symbols:
                    # Get recent market data
                    market_data = await self.data_handler.get_historical_data(
                        symbol,
                        start_date=(datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d'),
                        interval='5m'
                    )

                    if not market_data.empty:
                        # Make trading decision
                        decision = self.trading_agent.make_decision(market_data)

                        # Execute trade if confidence is high enough
                        if decision['confidence'] > self.config.trading.confidence_threshold:
                            await self.trading_agent.execute_trade(symbol, decision)

                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in live trading: {e}")
                await asyncio.sleep(10)  # Wait before retry

        self.is_trading = False
        self.logger.info("Live trading completed")

    async def evaluate_model(self, test_data_path: Optional[str] = None):
        """
        Evaluate model performance on test data.

        Args:
            test_data_path: Path to test data
        """
        if self.model is None:
            raise ValueError("Model must be loaded before evaluation")

        # Load test data
        if test_data_path:
            test_data = pd.read_csv(test_data_path)
        else:
            # Fetch test data using data handler
            test_data = await self.data_handler.get_multiple_tickers_data(
                self.config.data.tickers,
                start_date='2023-01-01',
                end_date='2023-06-01'
            )

        self.logger.info(f"Loaded test data with shape {test_data.shape}")

        # Create evaluation environment
        env = TradingEnvironment(test_data, self.config)

        # Run evaluation
        total_reward = 0
        num_episodes = 10

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                # Get action from model
                with torch.no_grad():
                    outputs = self.model(state_tensor)
                    action_probs = torch.softmax(outputs['signal_probabilities'], dim=1)
                    action = torch.argmax(action_probs, dim=1).item()

                # Take action in environment
                next_state, reward, terminated, truncated, _ = env.step([action, 5])
                done = terminated or truncated

                episode_reward += reward
                state = next_state

            total_reward += episode_reward
            self.logger.debug(f"Episode {episode} reward: {episode_reward:.2f}")

        avg_reward = total_reward / num_episodes
        self.logger.info(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")

        return avg_reward

    async def run_backtest(self, start_date: str, end_date: str, symbols: List[str]):
        """
        Run backtest on historical data.

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            symbols: List of symbols to backtest
        """
        if self.trading_agent is None:
            self.setup_trading_agent()

        self.logger.info(f"Running backtest from {start_date} to {end_date}")

        # Load historical data
        backtest_data = await self.data_handler.get_multiple_tickers_data(
            symbols,
            start_date=start_date,
            end_date=end_date
        )

        # Run backtest
        results = await self.trading_agent.run_backtest(backtest_data)

        # Log results
        self.logger.info(f"Backtest completed with results: {results}")

        return results

    def save_model(self, path: Optional[str] = None):
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.config.system.model_dir / f"nca_model_{timestamp}.pt"

        save_nca_model(self.model, path)
        self.logger.info(f"Model saved to {path}")

    def get_performance_report(self) -> Dict:
        """
        Get performance report.

        Returns:
            Performance report dictionary
        """
        if self.trading_agent:
            return self.trading_agent.get_performance_report()
        elif self.trainer:
            return self.trainer.get_training_summary()
        else:
            return {"status": "No trading or training activity"}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NCA Trading Bot')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'trade', 'evaluate', 'backtest'],
                        default='train', help='Operation mode')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--symbols', type=str, nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                        help='Symbols to trade or train on')
    parser.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--duration', type=int, default=60, help='Trading duration in minutes')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive model')
    parser.add_argument('--streaming', action='store_true', help='Use streaming training')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    return parser.parse_args()


async def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create bot
    bot = NCATradingBot(args.config)

    try:
        if args.mode == 'train':
            # Load or create model
            bot.create_model(adaptive=args.adaptive, load_path=args.model)

            # Set up trainer
            bot.setup_trainer()

            if args.streaming:
                # Streaming training
                await bot.train_streaming(args.symbols)
            else:
                # Offline training
                await bot.train_offline(args.data, args.epochs)

            # Save model
            bot.save_model()

        elif args.mode == 'trade':
            # Load model
            bot.create_model(load_path=args.model)

            # Set up trading agent
            bot.setup_trading_agent()

            # Start live trading
            await bot.trade_live(args.symbols, args.duration)

            # Print performance report
            report = bot.get_performance_report()
            print("Performance Report:")
            for key, value in report.items():
                print(f"  {key}: {value}")

        elif args.mode == 'evaluate':
            # Load model
            bot.create_model(load_path=args.model)

            # Evaluate model
            avg_reward = await bot.evaluate_model(args.data)
            print(f"Average reward: {avg_reward:.2f}")

        elif args.mode == 'backtest':
            # Load model
            bot.create_model(load_path=args.model)

            # Set up trading agent
            bot.setup_trading_agent()

            # Run backtest
            if not args.start_date or not args.end_date:
                print("Start date and end date are required for backtest")
                return

            results = await bot.run_backtest(args.start_date, args.end_date, args.symbols)
            print("Backtest Results:")
            for key, value in results.items():
                print(f"  {key}: {value}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        # Clean up
        if bot.is_training:
            bot.trainer.stop_training()
        if bot.is_trading:
            bot.is_trading = False

        logger.info("NCA Trading Bot shutdown complete")


if __name__ == "__main__":
    # Run main function
    asyncio.run(main())