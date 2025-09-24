#!/usr/bin/env python3
"""
Integration test for NCA Trading Bot components.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

async def test_data_handler():
    """Test DataHandler integration."""
    print("Testing DataHandler...")
    from data_handler import DataHandler

    dh = DataHandler()
    print("✓ DataHandler created successfully")

    # Test enriched market data
    enriched = await dh.get_enriched_market_data('AAPL', '2023-01-01', '2023-01-31')
    print(f"✓ Enriched market data keys: {list(enriched.keys())}")

    return True

def test_nca_model():
    """Test NCA model creation."""
    print("Testing NCA model...")
    from nca_model import create_nca_model
    from config import get_config

    config = get_config()
    model = create_nca_model(config)
    print("✓ NCA model created successfully")

    return True

def test_trainer():
    """Test trainer creation."""
    print("Testing trainer...")
    from trainer import PPOTrainer
    from nca_model import create_nca_model
    from config import get_config

    config = get_config()
    model = create_nca_model(config)
    trainer = PPOTrainer(model, config)
    print("✓ Trainer created successfully")

    return True

def test_utils():
    """Test utils initialization."""
    print("Testing utils...")
    from utils import initialize_utils
    from config import get_config

    config = get_config()
    utils = initialize_utils(config)
    print("✓ Utils initialized successfully")

    return True

async def main():
    """Run all integration tests."""
    print("NCA Trading Bot Integration Tests")
    print("=" * 40)

    tests = [
        test_nca_model,
        test_trainer,
        test_utils,
        test_data_handler,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                result = await test()
            else:
                result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)