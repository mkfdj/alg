"""
Alpaca API Client for NCA Trading Bot
Uses environment variables for authentication
"""

import os
import requests
from typing import Dict, Optional, Any
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd


class AlpacaClient:
    """
    Alpaca API client with both HTTP and SDK methods
    Uses environment variables for authentication
    """

    def __init__(self):
        """Initialize client with environment variables"""
        from .config import config

        self.api_key = os.environ.get("ALPACA_PAPER_API_KEY")
        self.secret_key = os.environ.get("ALPACA_PAPER_SECRET_KEY")
        self.base_url = config.alpaca_base_url

        if not self.api_key:
            raise ValueError("ALPACA_PAPER_API_KEY environment variable not found")
        if not self.secret_key:
            raise ValueError("ALPACA_PAPER_SECRET_KEY environment variable not found")

        self.trading_client = None
        self.data_client = None

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests"""
        return {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        }

    def test_connection(self) -> Dict[str, Any]:
        """Test API connection and return status"""
        results = {
            'http_api': False,
            'sdk_api': False,
            'market_data': False,
            'account_info': None
        }

        # Test HTTP API
        try:
            response = requests.get(
                f"{self.base_url}/account",
                headers=self._get_headers(),
                timeout=30
            )
            if response.status_code == 200:
                results['http_api'] = True
                results['account_info'] = response.json()
        except Exception as e:
            print(f"HTTP API test failed: {e}")

        # Test SDK API
        try:
            if not self.trading_client:
                self.trading_client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper=True
                )

            account = self.trading_client.get_account()
            results['sdk_api'] = True
            if not results['account_info']:
                results['account_info'] = {
                    'id': account.id,
                    'status': account.account_status,
                    'buying_power': str(account.buying_power),
                    'portfolio_value': str(account.portfolio_value)
                }

            # Test market data
            if not self.data_client:
                self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)

            request_params = StockLatestTradeRequest(symbol_or_symbols="AAPL")
            latest_trade = self.data_client.get_stock_latest_trade(request_params)

            if latest_trade and 'AAPL' in latest_trade:
                results['market_data'] = True

        except Exception as e:
            print(f"SDK API test failed: {e}")

        return results

    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            if not self.trading_client:
                self.trading_client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper=True
                )

            account = self.trading_client.get_account()
            return {
                'id': account.id,
                'status': account.account_status,
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'cash': float(account.cash),
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked
            }
        except Exception as e:
            print(f"Failed to get account info: {e}")
            return {}

    def get_market_data(self, symbol: str, timeframe: TimeFrame = TimeFrame.Day,
                       limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical market data"""
        try:
            if not self.data_client:
                self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)

            from alpaca.data.requests import StockBarsRequest
            from datetime import datetime, timedelta

            end_date = datetime.now()
            start_date = end_date - timedelta(days=limit * 2)  # Rough estimate

            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )

            bars = self.data_client.get_stock_bars(request_params)

            if symbol in bars.data:
                df = pd.DataFrame([bar.__dict__ for bar in bars.data[symbol]])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            else:
                return None

        except Exception as e:
            print(f"Failed to get market data for {symbol}: {e}")
            return None

    def place_order(self, symbol: str, qty: int, side: OrderSide) -> Dict[str, Any]:
        """Place a market order"""
        try:
            if not self.trading_client:
                self.trading_client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper=True
                )

            market_order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )

            order = self.trading_client.submit_order(market_order_data)

            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'status': order.status,
                'created_at': order.created_at
            }

        except Exception as e:
            print(f"Failed to place order for {symbol}: {e}")
            return {'error': str(e)}

    def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        try:
            if not self.trading_client:
                self.trading_client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper=True
                )

            positions = self.trading_client.get_all_positions()

            return [{
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'market_value': float(pos.market_value),
                'cost_basis': float(pos.cost_basis),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'side': pos.side
            } for pos in positions]

        except Exception as e:
            print(f"Failed to get positions: {e}")
            return []


# Convenience function for easy access
def get_alpaca_client() -> AlpacaClient:
    """Get an initialized Alpaca client"""
    return AlpacaClient()


# Quick test function
def quick_test() -> bool:
    """Quick test of Alpaca API connection"""
    try:
        client = AlpacaClient()
        results = client.test_connection()

        success = results['http_api'] or results['sdk_api']

        if success:
            print("✅ Alpaca API connection successful!")
            if results['account_info']:
                info = results['account_info']
                print(f"   Account ID: {info.get('id', 'N/A')}")
                print(f"   Status: {info.get('status', 'N/A')}")
                print(f"   Buying Power: ${info.get('buying_power', '0.00')}")
        else:
            print("❌ Alpaca API connection failed!")

        return success
    except Exception as e:
        print(f"❌ Alpaca API test failed: {e}")
        return False


if __name__ == "__main__":
    quick_test()