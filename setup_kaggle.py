#!/usr/bin/env python3
"""
Setup script for NCA Trading Bot on Kaggle
"""

import os
import subprocess
import sys

def install_kaggle():
    """Install kaggle package"""
    try:
        print("📦 Installing Kaggle API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle>=1.5.0"])
        print("✅ Kaggle API installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Kaggle API: {e}")
        return False

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
    print("=" * 50)

    # Install kaggle
    if not install_kaggle():
        print("❌ Setup failed at kaggle installation")
        return False

    # Verify credentials
    if not verify_kaggle_credentials():
        print("⚠️  Please set up Kaggle credentials and run this script again")
        return False

    # Test download
    if not test_kaggle_download():
        print("⚠️  Kaggle test failed, but basic setup is complete")

    print("\n✅ Kaggle setup completed!")
    print("\nNow you can run the NCA Trading Bot:")
    print("python run_nca_bot.py")
    print("or")
    print("python nca_trading_bot/main.py --mode train --kaggle")

    return True

if __name__ == "__main__":
    main()