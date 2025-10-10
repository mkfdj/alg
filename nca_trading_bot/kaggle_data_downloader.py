"""
Kaggle Data Downloader for NCA Trading Bot
Automatically downloads and processes real financial data from Kaggle
"""

import os
import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time

def check_kaggle_credentials():
    """Check if Kaggle credentials are set"""
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')

    if not username or not key:
        print("❌ Kaggle credentials not found!")
        print("Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        return False

    print(f"✅ Kaggle credentials found for user: {username}")
    return True

def download_kaggle_dataset(dataset_name: str, download_path: str = "/kaggle/working") -> bool:
    """Download a Kaggle dataset"""
    try:
        print(f"📥 Downloading {dataset_name}...")

        # Create download command
        cmd = f"kaggle datasets download -d {dataset_name} -p {download_path}"

        # Run download
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"✅ Successfully downloaded {dataset_name}")
            return True
        else:
            print(f"❌ Failed to download {dataset_name}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"❌ Download timeout for {dataset_name}")
        return False
    except Exception as e:
        print(f"❌ Error downloading {dataset_name}: {e}")
        return False

def extract_dataset(zip_path: str, extract_path: str) -> bool:
    """Extract downloaded dataset"""
    try:
        print(f"📂 Extracting {zip_path}...")

        # Create extract command
        cmd = f"unzip -q {zip_path} -d {extract_path}"

        # Run extraction
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print(f"✅ Successfully extracted {zip_path}")
            return True
        else:
            print(f"❌ Failed to extract {zip_path}: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error extracting {zip_path}: {e}")
        return False

def download_and_process_datasets(config, target_tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Download and process Kaggle datasets"""
    print("🚀 Starting Kaggle data download and processing...")

    # Check credentials
    if not check_kaggle_credentials():
        return {}

    # Define datasets to download
    datasets = [
        {
            "name": "jacksoncrow/stock-market-dataset",
            "zip_name": "stock-market-dataset.zip",
            "description": "NASDAQ stocks and ETFs data"
        },
        {
            "name": "camnugent/sandp500",
            "zip_name": "sandp500.zip",
            "description": "S&P 500 stock data"
        }
    ]

    working_dir = Path("/kaggle/working")
    data_dir = working_dir / "downloaded_data"
    data_dir.mkdir(exist_ok=True)

    all_data = {}

    for dataset in datasets:
        print(f"\n📊 Processing {dataset['description']}...")

        # Download dataset
        if download_kaggle_dataset(dataset["name"], str(data_dir)):
            zip_path = data_dir / dataset["zip_name"]

            # Extract if zip exists
            if zip_path.exists():
                extract_path = data_dir / dataset["name"].split("/")[-1]
                if extract_dataset(str(zip_path), str(extract_path)):

                    # Process the extracted data
                    processed_data = process_extracted_data(extract_path, dataset["name"], target_tickers)
                    all_data.update(processed_data)

                    # Clean up zip file
                    zip_path.unlink()
                else:
                    print(f"❌ Failed to extract {dataset['description']}")
            else:
                print(f"❌ Zip file not found: {zip_path}")
        else:
            print(f"❌ Failed to download {dataset['description']}")

    print(f"\n✅ Data download and processing completed!")
    print(f"📈 Total tickers loaded: {len(all_data)}")

    return all_data

def process_extracted_data(extract_path: Path, dataset_name: str, target_tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Process extracted dataset files"""
    data = {}

    try:
        if "stock-market-dataset" in dataset_name:
            # Process stock market dataset (stocks and etfs folders)
            for folder_name in ["stocks", "etfs"]:
                folder_path = extract_path / folder_name
                if folder_path.exists():
                    print(f"  📁 Processing {folder_name} folder...")

                    csv_files = list(folder_path.glob("*.csv"))
                    if target_tickers:
                        csv_files = [f for f in csv_files if f.stem in target_tickers]

                    # Limit to first 100 files for performance
                    csv_files = csv_files[:100]

                    for csv_file in csv_files:
                        try:
                            ticker = csv_file.stem
                            df = pd.read_csv(csv_file, index_col='Date', parse_dates=True)

                            # Clean and validate
                            df = clean_stock_data(df, ticker)
                            if df is not None and len(df) > 100:  # At least 100 days of data
                                data[ticker] = df

                        except Exception as e:
                            print(f"    ⚠️  Error processing {csv_file.name}: {e}")
                            continue

                    print(f"    ✅ Loaded {len([f for f in folder_path.glob('*.csv') if f.stem in data])} tickers from {folder_name}")

        elif "sandp500" in dataset_name:
            # Process S&P 500 dataset
            all_stocks_file = extract_path / "all_stocks_5yr.csv"
            individual_folder = extract_path / "individual_stocks_5yr"

            if all_stocks_file.exists():
                print("  📁 Processing all_stocks_5yr.csv...")
                df = pd.read_csv(all_stocks_file)

                # Filter by target tickers if specified
                if target_tickers:
                    df = df[df['Name'].isin(target_tickers)]

                # Process each ticker
                for ticker in df['Name'].unique():
                    ticker_df = df[df['Name'] == ticker].copy()
                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
                    ticker_df.set_index('Date', inplace=True)
                    ticker_df.drop('Name', axis=1, inplace=True)

                    # Clean and validate
                    ticker_df = clean_stock_data(ticker_df, ticker)
                    if ticker_df is not None and len(ticker_df) > 100:
                        data[ticker] = ticker_df

                print(f"    ✅ Loaded {len(data)} tickers from S&P 500 dataset")

            elif individual_folder.exists():
                print("  📁 Processing individual stock files...")
                csv_files = list(individual_folder.glob("*.csv"))

                if target_tickers:
                    csv_files = [f for f in csv_files if f.stem in target_tickers]

                # Limit to first 50 files
                csv_files = csv_files[:50]

                for csv_file in csv_files:
                    try:
                        ticker = csv_file.stem
                        df = pd.read_csv(csv_file, index_col='Date', parse_dates=True)

                        # Clean and validate
                        df = clean_stock_data(df, ticker)
                        if df is not None and len(df) > 100:
                            data[ticker] = df

                    except Exception as e:
                        print(f"    ⚠️  Error processing {csv_file.name}: {e}")
                        continue

                print(f"    ✅ Loaded {len(data)} tickers from individual files")

    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {e}")

    return data

def clean_stock_data(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Clean and validate stock data"""
    try:
        # Standardize column names
        df.columns = [col.strip().lower() for col in df.columns]

        # Required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
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

        # Remove zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df = df[df[col] > 0]

        # Ensure minimum data length
        if len(df) < 100:
            return None

        # Sort by date
        df = df.sort_index()

        return df

    except Exception as e:
        print(f"Error cleaning data for {ticker}: {e}")
        return None

def print_data_statistics(data: Dict[str, pd.DataFrame]):
    """Print comprehensive data statistics"""
    print("\n📊 Data Statistics:")
    print("=" * 50)

    if not data:
        print("❌ No data available")
        return

    total_tickers = len(data)
    total_data_points = sum(len(df) for df in data.values())

    print(f"Total tickers: {total_tickers}")
    print(f"Total data points: {total_data_points:,}")

    # Date range
    all_dates = []
    for ticker, df in data.items():
        all_dates.extend(df.index.tolist())

    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        print(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        print(f"Total days: {(max_date - min_date).days}")

    # Sample tickers
    print(f"\nSample tickers ({min(10, total_tickers)}):")
    for i, (ticker, df) in enumerate(list(data.items())[:10]):
        if 'close' in df.columns:
            latest_price = df['close'].iloc[-1]
            first_price = df['close'].iloc[0]
            return_pct = ((latest_price / first_price - 1) * 100)
            print(f"  {ticker}: ${latest_price:.2f} ({return_pct:+.1f}%) - {len(df)} days")
        else:
            print(f"  {ticker}: {len(df)} days")

    print("=" * 50)

if __name__ == "__main__":
    print("🧬 Kaggle Data Downloader - NCA Trading Bot")
    print("=" * 50)

    # Test download
    from nca_trading_bot.config import Config
    config = Config()

    data = download_and_process_datasets(config, config.top_tickers[:5])
    print_data_statistics(data)