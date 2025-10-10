#!/usr/bin/env python3
"""
Setup script for NCA Trading Bot on Kaggle
Installs required dependencies and configures the environment
"""

import subprocess
import sys
import os

def install_package(package_name):
    """Install a package using pip"""
    try:
        print(f"📦 Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--quiet"])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def install_kaggle():
    """Install kaggle package"""
    return install_package("kaggle>=1.5.0")

def verify_kaggle_credentials():
    """Verify Kaggle credentials"""
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')

    if username and key:
        print(f"✅ Kaggle credentials found for user: {username}")
        return True

    print("⚠️  Kaggle credentials not found!")
    print("\nTo set up Kaggle credentials:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json")
    print("4. Run one of these commands:")
    print("\nOption A - Set environment variables:")
    print("export KAGGLE_USERNAME='your_username'")
    print("export KAGGLE_KEY='your_api_key'")
    print("\nOption B - Upload kaggle.json to /root/.kaggle/")
    print("mkdir -p ~/.kaggle")
    print("mv kaggle.json ~/.kaggle/")

    return False

def test_kaggle_download():
    """Test kaggle download with a small dataset"""
    try:
        print("🧪 Testing Kaggle download...")
        # Test with a small dataset
        result = subprocess.run(
            ["kaggle", "datasets", "list", "--page-size", "1"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print("✅ Kaggle download test successful")
            return True
        else:
            print(f"❌ Kaggle test failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Kaggle test timed out")
        return False
    except Exception as e:
        print(f"❌ Kaggle test error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 NCA Trading Bot - Kaggle Setup")
    print("=" * 60)

    # Required packages for trading bot
    required_packages = [
        "jax[tpu]>=0.4.8",
        "flax>=0.7.0",
        "optax>=0.1.4",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "yfinance>=0.2.0",
        "ta>=0.10.0",
        "scikit-learn>=1.1.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0"
    ]

    print("📦 Installing required packages...")
    failed_packages = []

    for package in required_packages:
        if not install_package(package):
            failed_packages.append(package)

    if failed_packages:
        print(f"\n⚠️  Warning: Failed to install {len(failed_packages)} packages:")
        for package in failed_packages:
            print(f"  - {package}")

    # Install kaggle
    if not install_kaggle():
        print("❌ Setup failed at kaggle installation")
        return False

    # Set environment variables for TPU
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Verify credentials
    if not verify_kaggle_credentials():
        print("⚠️  Please set up Kaggle credentials and run this script again")
        return False

    # Test download
    if not test_kaggle_download():
        print("⚠️  Kaggle test failed, but basic setup is complete")

    print("\n✅ Kaggle setup completed!")
    print("\nNow you can run the NCA Trading Bot:")
    print("python nca_trading_bot/kaggle_main.py --mode train")
    print("or")
    print("python nca_trading_bot/kaggle_main.py --mode demo")

    return True

if __name__ == "__main__":
    main()