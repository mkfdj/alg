"""
Dataset download script for NCA Trading Bot
Downloads and prepares datasets for training
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta


def download_kaggle_datasets():
    """Download datasets from Kaggle"""
    print("Downloading Kaggle datasets...")

    datasets = [
        "jacksoncrow/stock-market-dataset",
        "jakewright/9000-tickers-of-stock-market-data-full-history",
        "cameronn/securities-database"
    ]

    for dataset in datasets:
        try:
            print(f"Downloading {dataset}...")
            cmd = f"kaggle datasets download -d {dataset}"
            subprocess.run(cmd, shell=True, check=True)

            # Extract the dataset
            zip_file = dataset.split('/')[-1] + '.zip'
            if os.path.exists(zip_file):
                print(f"Extracting {zip_file}...")
                subprocess.run(f"unzip {zip_file}", shell=True, check=True)
                os.remove(zip_file)

        except subprocess.CalledProcessError as e:
            print(f"Failed to download {dataset}: {e}")
        except FileNotFoundError:
            print("Kaggle CLI not found. Please install it first.")
            print("pip install kaggle")
            return False

    return True


def download_yfinance_data(tickers, start_date="1990-01-01", end_date="2021-12-31"):
    """Download data using yfinance"""
    print(f"Downloading yfinance data for {len(tickers)} tickers...")

    data_dir = Path("yfinance_data")
    data_dir.mkdir(exist_ok=True)

    successful_downloads = []
    failed_downloads = []

    for ticker in tickers:
        try:
            print(f"Downloading {ticker}...")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if not df.empty:
                # Save to CSV
                file_path = data_dir / f"{ticker}.csv"
                df.to_csv(file_path)
                successful_downloads.append(ticker)
                print(f"✅ {ticker}: {len(df)} days of data")
            else:
                failed_downloads.append(ticker)
                print(f"❌ {ticker}: No data available")

        except Exception as e:
            failed_downloads.append(ticker)
            print(f"❌ {ticker}: Error - {e}")

    print(f"\nDownload Summary:")
    print(f"✅ Successful: {len(successful_downloads)} tickers")
    print(f"❌ Failed: {len(failed_downloads)} tickers")

    return successful_downloads, failed_downloads


def download_sp500_data():
    """Download S&P 500 constituent data"""
    print("Downloading S&P 500 data...")

    try:
        # Get S&P 500 tickers from Wikipedia
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(sp500_url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()

        # Remove dots from tickers (e.g., BRK.B -> BRK-B)
        tickers = [ticker.replace('.', '-') for ticker in tickers]

        print(f"Found {len(tickers)} S&P 500 tickers")

        # Download data for top 50 tickers by market cap
        top_tickers = tickers[:50]

        successful, failed = download_yfinance_data(
            top_tickers,
            start_date="2010-01-01",
            end_date="2021-12-31"
        )

        return successful, failed

    except Exception as e:
        print(f"Failed to download S&P 500 data: {e}")
        return [], []


def create_dataset_metadata():
    """Create metadata file for datasets"""
    print("Creating dataset metadata...")

    metadata = {
        "created_at": datetime.now().isoformat(),
        "datasets": {
            "kaggle_stock_market": {
                "description": "Historical daily prices for NASDAQ-traded stocks and ETFs",
                "source": "https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset",
                "format": "CSV",
                "columns": ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            },
            "9000_tickers": {
                "description": "Historical stock market data covering over 9,000 tickers from 1962 to present",
                "source": "https://www.kaggle.com/datasets/jakewright/9000-tickers-of-stock-market-data-full-history",
                "format": "CSV",
                "columns": ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            },
            "yfinance_data": {
                "description": "Real-time and historical data from Yahoo Finance",
                "source": "https://finance.yahoo.com/",
                "format": "CSV",
                "tickers": "Downloaded separately",
                "columns": ["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
            }
        },
        "preprocessing": {
            "technical_indicators": [
                "RSI", "MACD", "Bollinger Bands", "ATR", "Moving Averages"
            ],
            "normalization": "MinMax scaling applied to all features",
            "sequence_length": 200,
            "prediction_horizon": 20
        }
    }

    # Save metadata
    import json
    with open("dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Dataset metadata created")


def validate_datasets():
    """Validate downloaded datasets"""
    print("Validating datasets...")

    validation_results = {
        "total_tickers": 0,
        "valid_tickers": 0,
        "total_data_points": 0,
        "date_ranges": {},
        "issues": []
    }

    # Check Kaggle datasets
    kaggle_paths = ["stocks", "etfs"]
    for path in kaggle_paths:
        if os.path.exists(path):
            csv_files = list(Path(path).glob("*.csv"))
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if len(df) > 0:
                        validation_results["total_tickers"] += 1

                        # Check data quality
                        if df.isnull().sum().sum() / len(df) < 0.1:  # Less than 10% missing
                            validation_results["valid_tickers"] += 1
                            validation_results["total_data_points"] += len(df)

                        # Get date range
                        if 'Date' in df.columns:
                            min_date = df['Date'].min()
                            max_date = df['Date'].max()
                            ticker = csv_file.stem
                            validation_results["date_ranges"][ticker] = (min_date, max_date)

                except Exception as e:
                    validation_results["issues"].append(f"Error reading {csv_file}: {e}")

    # Check yfinance data
    yfinance_path = Path("yfinance_data")
    if yfinance_path.exists():
        csv_files = list(yfinance_path.glob("*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                if len(df) > 0:
                    validation_results["total_tickers"] += 1
                    validation_results["valid_tickers"] += 1
                    validation_results["total_data_points"] += len(df)

                    ticker = csv_file.stem
                    validation_results["date_ranges"][ticker] = (df.index.min(), df.index.max())

            except Exception as e:
                validation_results["issues"].append(f"Error reading {csv_file}: {e}")

    # Print validation results
    print(f"✅ Validation Results:")
    print(f"   Total tickers: {validation_results['total_tickers']}")
    print(f"   Valid tickers: {validation_results['valid_tickers']}")
    print(f"   Total data points: {validation_results['total_data_points']:,}")

    if validation_results["issues"]:
        print(f"⚠️  Issues found:")
        for issue in validation_results["issues"]:
            print(f"   - {issue}")

    return validation_results


def main():
    parser = argparse.ArgumentParser(description="Download datasets for NCA Trading Bot")
    parser.add_argument("--kaggle", action="store_true", help="Download Kaggle datasets")
    parser.add_argument("--yfinance", action="store_true", help="Download yfinance data")
    parser.add_argument("--sp500", action="store_true", help="Download S&P 500 data")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to download")
    parser.add_argument("--validate", action="store_true", help="Validate downloaded datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")

    args = parser.parse_args()

    if args.all or not any([args.kaggle, args.yfinance, args.sp500, args.validate]):
        args.kaggle = True
        args.yfinance = True
        args.sp500 = True
        args.validate = True

    print("🚀 NCA Trading Bot Dataset Downloader")
    print("=" * 50)

    # Download Kaggle datasets
    if args.kaggle:
        print("\n📥 Downloading Kaggle datasets...")
        success = download_kaggle_datasets()
        if not success:
            print("❌ Failed to download Kaggle datasets")

    # Download yfinance data
    if args.yfinance:
        print("\n📈 Downloading yfinance data...")
        top_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "JNJ", "V",
            "PG", "UNH", "HD", "MA", "BAC", "XOM", "PFE", "CSCO", "KO", "PEP"
        ]

        if args.tickers:
            top_tickers = args.tickers

        successful, failed = download_yfinance_data(top_tickers)
        print(f"✅ Successfully downloaded: {len(successful)} tickers")
        print(f"❌ Failed to download: {len(failed)} tickers")

    # Download S&P 500 data
    if args.sp500:
        print("\n📊 Downloading S&P 500 data...")
        successful, failed = download_sp500_data()
        print(f"✅ Successfully downloaded: {len(successful)} tickers")
        print(f"❌ Failed to download: {len(failed)} tickers")

    # Create metadata
    create_dataset_metadata()

    # Validate datasets
    if args.validate:
        print("\n🔍 Validating datasets...")
        validation_results = validate_datasets()

    print("\n✅ Dataset download completed!")
    print("\nNext steps:")
    print("1. Run: python nca_trading_bot/main.py --mode analyze")
    print("2. Run: python nca_trading_bot/main.py --mode train")
    print("3. Run: python nca_trading_bot/main.py --mode backtest")


if __name__ == "__main__":
    main()