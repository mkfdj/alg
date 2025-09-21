"""
Unit tests for main CLI module.

Tests command-line interface functionality and argument parsing.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from ..main import NCACommandLineInterface


class TestNCACommandLineInterface(unittest.TestCase):
    """Test cases for NCACommandLineInterface class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cli = NCACommandLineInterface()

    def test_initialization(self):
        """Test CLI initialization."""
        self.assertIsNotNone(self.cli.config)
        self.assertIsNotNone(self.cli.config_manager)
        self.assertIsNotNone(self.cli.data_handler)
        self.assertIsNotNone(self.cli.training_manager)
        self.assertIsNotNone(self.cli.performance_monitor)
        self.assertIsNotNone(self.cli.logger)

    @patch('argparse.ArgumentParser.parse_args')
    def test_run_with_backtest_command(self, mock_parse_args):
        """Test running backtest command."""
        # Mock arguments
        mock_args = Mock()
        mock_args.func = self.cli._run_backtest
        mock_args.tickers = ['AAPL', 'MSFT']
        mock_args.days = 30
        mock_args.interval = '1d'
        mock_args.output = None
        mock_parse_args.return_value = mock_args

        # Mock the backtest method
        with patch.object(self.cli, '_run_backtest') as mock_backtest:
            self.cli.run()
            mock_backtest.assert_called_once_with(mock_args)

    @patch('argparse.ArgumentParser.parse_args')
    def test_run_with_train_command(self, mock_parse_args):
        """Test running train command."""
        # Mock arguments
        mock_args = Mock()
        mock_args.func = self.cli._run_train
        mock_args.mode = 'offline'
        mock_args.epochs = 100
        mock_args.symbols = None
        mock_parse_args.return_value = mock_args

        # Mock the train method
        with patch.object(self.cli, '_run_train') as mock_train:
            self.cli.run()
            mock_train.assert_called_once_with(mock_args)

    @patch('argparse.ArgumentParser.parse_args')
    def test_run_with_live_command(self, mock_parse_args):
        """Test running live command."""
        # Mock arguments
        mock_args = Mock()
        mock_args.func = self.cli._run_live
        mock_args.symbols = ['SPY', 'QQQ']
        mock_args.paper = True
        mock_parse_args.return_value = mock_args

        # Mock the live method
        with patch.object(self.cli, '_run_live') as mock_live:
            self.cli.run()
            mock_live.assert_called_once_with(mock_args)

    @patch('argparse.ArgumentParser.parse_args')
    def test_run_with_data_command(self, mock_parse_args):
        """Test running data command."""
        # Mock arguments
        mock_args = Mock()
        mock_args.func = self.cli._run_data
        mock_args.tickers = ['AAPL', 'MSFT']
        mock_args.analyze = True
        mock_parse_args.return_value = mock_args

        # Mock the data method
        with patch.object(self.cli, '_run_data') as mock_data:
            self.cli.run()
            mock_data.assert_called_once_with(mock_args)

    @patch('argparse.ArgumentParser.parse_args')
    def test_run_with_model_command(self, mock_parse_args):
        """Test running model command."""
        # Mock arguments
        mock_args = Mock()
        mock_args.func = self.cli._run_model
        mock_args.action = 'create'
        mock_parse_args.return_value = mock_args

        # Mock the model method
        with patch.object(self.cli, '_run_model') as mock_model:
            self.cli.run()
            mock_model.assert_called_once_with(mock_args)

    @patch('argparse.ArgumentParser.parse_args')
    def test_run_with_status_command(self, mock_parse_args):
        """Test running status command."""
        # Mock arguments
        mock_args = Mock()
        mock_args.func = self.cli._run_status
        mock_args.detailed = False
        mock_parse_args.return_value = mock_args

        # Mock the status method
        with patch.object(self.cli, '_run_status') as mock_status:
            self.cli.run()
            mock_status.assert_called_once_with(mock_args)

    @patch('argparse.ArgumentParser.parse_args')
    def test_run_with_config_command(self, mock_parse_args):
        """Test running config command."""
        # Mock arguments
        mock_args = Mock()
        mock_args.func = self.cli._run_config
        mock_args.action = 'show'
        mock_parse_args.return_value = mock_args

        # Mock the config method
        with patch.object(self.cli, '_run_config') as mock_config:
            self.cli.run()
            mock_config.assert_called_once_with(mock_args)

    def test_signal_handler(self):
        """Test signal handler."""
        # Mock signal handler
        with patch('signal.signal') as mock_signal:
            # Create CLI instance (this will set up signal handlers)
            cli = NCACommandLineInterface()

            # Check that signal.signal was called
            self.assertEqual(mock_signal.call_count, 2)  # SIGINT and SIGTERM

    @patch('builtins.print')
    def test_display_backtest_results(self, mock_print):
        """Test backtest results display."""
        results = {
            'AAPL': {
                'total_reward': 100.5,
                'final_balance': 100500.0,
                'total_trades': 10,
                'win_rate': 0.6
            },
            'MSFT': {
                'total_reward': 75.25,
                'final_balance': 100375.0,
                'total_trades': 8,
                'win_rate': 0.5
            }
        }

        self.cli._display_backtest_results(results)

        # Check that print was called multiple times
        self.assertGreater(mock_print.call_count, 0)

    def test_save_backtest_results(self):
        """Test backtest results saving."""
        results = {
            'AAPL': {
                'total_reward': 100.5,
                'final_balance': 100500.0,
                'total_trades': 10,
                'win_rate': 0.6
            }
        }

        # Test saving
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            with patch('json.dump') as mock_json:
                self.cli._save_backtest_results(results, 'test_results.json')

                mock_file.assert_called_once()
                mock_json.assert_called_once()

    def test_analyze_data(self):
        """Test data analysis."""
        data = {
            'AAPL': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10),
                'Close': [100 + i for i in range(10)],
                'Volume': [1000 + i * 100 for i in range(10)]
            })
        }

        # Should not raise exception
        with patch('builtins.print') as mock_print:
            self.cli._analyze_data(data)
            self.assertGreater(mock_print.call_count, 0)

    @patch('sys.exit')
    def test_error_handling(self, mock_exit):
        """Test error handling in run method."""
        # Mock argument parsing to raise exception
        with patch('argparse.ArgumentParser.parse_args', side_effect=Exception("Test error")):
            cli = NCACommandLineInterface()
            cli.run()

            mock_exit.assert_called_once_with(1)

    def test_parser_creation(self):
        """Test argument parser creation."""
        parser = self.cli._create_parser()

        # Test that parser has expected arguments
        args = parser.parse_args(['status'])
        self.assertEqual(args.command, 'status')

        # Test help
        help_output = parser.format_help()
        self.assertIn('NCA Trading Bot', help_output)
        self.assertIn('backtest', help_output)
        self.assertIn('train', help_output)
        self.assertIn('live', help_output)


if __name__ == '__main__':
    unittest.main()