# NCA Trading Bot

A sophisticated neural network-based trading bot with reinforcement learning capabilities for automated financial trading.

## Features

- **Neural Network Architecture**: Advanced NCA (Neural Cellular Automata) based trading model
- **Reinforcement Learning**: Self-improving trading strategies using RL algorithms
- **Technical Analysis**: Comprehensive technical indicators with multiple library support
- **Multi-Asset Support**: Trade stocks, ETFs, and other financial instruments
- **Real-time Processing**: Live market data integration with Alpaca and Yahoo Finance
- **Risk Management**: Advanced position sizing and risk calculation utilities
- **Performance Monitoring**: Real-time metrics and comprehensive backtesting

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Basic Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nca-trading-bot
```

2. Install core dependencies:
```bash
pip install -r requirements.txt
```

## Technical Analysis Libraries

The bot supports multiple technical analysis libraries with automatic fallback. Choose one of the following options:

### Option 1: TA-Lib (Recommended for Performance)

TA-Lib is the fastest and most reliable option for technical analysis.

#### Installation on Windows:

1. Download TA-Lib from: https://ta-lib.org/
2. Install the C library:
   - Run the downloaded installer
   - Add `C:\ta-lib` to your system PATH

3. Install Python wrapper:
```bash
pip install TA-Lib
```

#### Installation on Linux/macOS:

```bash
# Install TA-Lib C library
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr/local
make
sudo make install

# Install Python wrapper
pip install TA-Lib
```

### Option 2: pandas-ta (Easiest Installation)

pandas-ta is a pure Python library that works without C dependencies.

```bash
pip install pandas-ta
```

### Option 3: NumPy/Pandas Fallbacks (No Installation Required)

If no technical analysis library is available, the bot will automatically use pure NumPy/Pandas implementations.

## Configuration

1. Copy the configuration template:
```bash
cp config.example.py config.py
```

2. Edit `config.py` with your API keys and preferences:
```python
# API Configuration
api:
  alpaca_api_key: "YOUR_ALPACA_API_KEY"
  alpaca_secret_key: "YOUR_ALPACA_SECRET_KEY"
  alpaca_base_url: "https://paper-api.alpaca.markets"

# Trading Configuration
trading:
  tickers: ["AAPL", "MSFT", "GOOGL"]
  initial_balance: 100000
  max_position_size: 0.1

# Technical Analysis Configuration
data:
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bb_period: 20
  bb_std: 2.0
```

## Usage

### Basic Usage

```python
from nca_trading_bot.main import TradingBot

# Initialize bot
bot = TradingBot()

# Run backtest
results = bot.backtest("AAPL", "2023-01-01", "2023-12-31")

# Start live trading (with caution!)
# bot.start_trading()
```

### Command Line Interface

```bash
# Run backtest
python -m nca_trading_bot.main backtest --ticker AAPL --start-date 2023-01-01 --end-date 2023-12-31

# Train model
python -m nca_trading_bot.main train --episodes 1000

# Start live trading
python -m nca_trading_bot.main trade --live
```

## Technical Indicators

The bot calculates the following technical indicators:

- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands**
- **Simple Moving Averages** (SMA)
- **Exponential Moving Averages** (EMA)
- **Stochastic Oscillator**
- **Average True Range** (ATR)
- **On-Balance Volume** (OBV)

### Library Priority

The bot automatically selects the best available library:

1. **TA-Lib** (if installed) - Fastest and most accurate
2. **pandas-ta** (if installed) - Good performance, easy installation
3. **NumPy/Pandas fallbacks** - Always available, slower but functional

## Troubleshooting

### TA-Lib Installation Issues

**Problem**: `ImportError: cannot import name 'talib'`

**Solution**: Install TA-Lib properly:

1. Ensure TA-Lib C library is installed
2. Install Python wrapper: `pip install TA-Lib`
3. On Windows, make sure `ta-lib.dll` is in your PATH

**Problem**: `talib` package not found on PyPI

**Solution**: Use `TA-Lib` (capital letters with hyphen) or install manually.

### Performance Issues

- Use TA-Lib for best performance
- Consider using GPU acceleration for neural networks
- Monitor memory usage with large datasets

### API Connection Issues

- Verify API keys in configuration
- Check internet connection
- Ensure API endpoints are accessible

## Development

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_data_handler.py

# Run with coverage
python -m pytest --cov=nca_trading_bot
```

### Code Quality

```bash
# Format code
black nca_trading_bot/

# Check style
flake8 nca_trading_bot/

# Type checking
mypy nca_trading_bot/
```

## Architecture

- **Data Handler**: Fetches and processes market data
- **Technical Indicators**: Calculates trading signals
- **NCA Model**: Neural network for trading decisions
- **Trading Engine**: Executes trades and manages positions
- **Risk Manager**: Monitors and controls trading risk

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.

**Always test thoroughly with paper trading before using real money.**