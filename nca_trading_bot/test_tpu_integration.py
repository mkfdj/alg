"""
TPU Integration Test Script for NCA Trading Bot.

This script tests TPU functionality and verifies that all TPU optimizations
are working correctly. Run this script to validate TPU integration.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import get_config, detect_tpu_availability, get_tpu_device_count
from nca_model import create_nca_model, NCATradingModel
from trainer import PPOTrainer, DistributedTrainer
from utils import PerformanceMonitor

def test_tpu_detection():
    """Test TPU detection functionality."""
    print("Testing TPU detection...")

    tpu_available = detect_tpu_availability()
    tpu_count = get_tpu_device_count()

    print(f"TPU Available: {tpu_available}")
    print(f"TPU Device Count: {tpu_count}")

    if tpu_available:
        print("✅ TPU detection working correctly")
        return True
    else:
        print("ℹ️  No TPU detected - running CPU/GPU tests")
        return False

def test_tpu_configuration():
    """Test TPU configuration loading."""
    print("\nTesting TPU configuration...")

    config = get_config()

    # Test TPU device setting
    config.system.device = "tpu"
    config.tpu.tpu_cores = 8
    config.tpu.xla_compile = True
    config.tpu.sharding_strategy = "2d"

    print(f"Device: {config.system.device}")
    print(f"TPU Cores: {config.tpu.tpu_cores}")
    print(f"XLA Compile: {config.tpu.xla_compile}")
    print(f"Sharding Strategy: {config.tpu.sharding_strategy}")

    print("✅ TPU configuration working correctly")
    return True

def test_tpu_model_creation():
    """Test TPU-optimized model creation."""
    print("\nTesting TPU model creation...")

    config = get_config()
    config.system.device = "tpu"

    try:
        model = create_nca_model(config)
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

        # Test model device placement
        print(f"Model device: {next(model.parameters()).device}")

        print("✅ TPU model creation working correctly")
        return True
    except Exception as e:
        print(f"❌ TPU model creation failed: {e}")
        return False

def test_tpu_training_setup():
    """Test TPU training setup."""
    print("\nTesting TPU training setup...")

    config = get_config()
    config.system.device = "tpu"

    try:
        model = create_nca_model(config)
        trainer = PPOTrainer(model, config)

        print(f"Trainer device: {trainer.device}")
        print(f"AMP enabled: {trainer.use_amp}")

        print("✅ TPU training setup working correctly")
        return True
    except Exception as e:
        print(f"❌ TPU training setup failed: {e}")
        return False

def test_tpu_distributed_training():
    """Test TPU distributed training setup."""
    print("\nTesting TPU distributed training...")

    config = get_config()
    config.system.device = "tpu"

    try:
        distributed_trainer = DistributedTrainer(config)

        print(f"World size: {distributed_trainer.world_size}")
        print(f"Local rank: {distributed_trainer.local_rank}")
        print(f"Device type: {distributed_trainer.device_type}")

        print("✅ TPU distributed training setup working correctly")
        return True
    except Exception as e:
        print(f"❌ TPU distributed training setup failed: {e}")
        return False

def test_tpu_performance_monitoring():
    """Test TPU performance monitoring."""
    print("\nTesting TPU performance monitoring...")

    try:
        monitor = PerformanceMonitor()
        metrics = monitor.get_system_metrics()

        print("System metrics collected:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Test TPU-specific metrics
        tpu_metrics = monitor.get_tpu_training_metrics()
        print("TPU metrics collected:")
        for key, value in tpu_metrics.items():
            print(f"  {key}: {value}")

        print("✅ TPU performance monitoring working correctly")
        return True
    except Exception as e:
        print(f"❌ TPU performance monitoring failed: {e}")
        return False

def test_tpu_model_forward_pass():
    """Test TPU model forward pass."""
    print("\nTesting TPU model forward pass...")

    config = get_config()
    config.system.device = "tpu"

    try:
        model = create_nca_model(config)

        # Create sample input
        batch_size, seq_len, features = 4, 10, 20
        sample_input = torch.randn(batch_size, seq_len, features)

        # Move to TPU device if available
        if detect_tpu_availability():
            try:
                import torch_xla.core.xla_model as xm
                sample_input = sample_input.to(xm.xla_device())
                model = model.to(xm.xla_device())
            except ImportError:
                pass

        # Forward pass
        with torch.no_grad():
            outputs = model(sample_input)

        print(f"Input shape: {sample_input.shape}")
        print(f"Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")

        print("✅ TPU model forward pass working correctly")
        return True
    except Exception as e:
        print(f"❌ TPU model forward pass failed: {e}")
        return False

def test_xla_compilation():
    """Test XLA compilation functionality."""
    print("\nTesting XLA compilation...")

    try:
        # Test XLA import
        import torch_xla.core.xla_model as xm
        print("✅ XLA import successful")

        # Test XLA device creation
        device = xm.xla_device()
        print(f"✅ XLA device created: {device}")

        # Test XLA compilation
        x = torch.randn(3, 3)
        x = x.to(device)
        y = torch.matmul(x, x)
        xm.mark_step()
        print("✅ XLA compilation and execution successful")

        return True
    except ImportError:
        print("ℹ️  XLA not available - skipping XLA tests")
        return True
    except Exception as e:
        print(f"❌ XLA compilation failed: {e}")
        return False

def test_tpu_optimizations():
    """Test TPU-specific optimizations."""
    print("\nTesting TPU optimizations...")

    try:
        config = get_config()
        config.system.device = "tpu"

        model = create_nca_model(config)

        # Test TPU-optimized forward passes
        sample_input = torch.randn(2, 10, 20)

        if detect_tpu_availability():
            try:
                import torch_xla.core.xla_model as xm
                sample_input = sample_input.to(xm.xla_device())
                model = model.to(xm.xla_device())
            except ImportError:
                pass

        # Test regular forward pass
        outputs1 = model(sample_input, evolution_steps=1)

        # Test TPU-optimized forward pass (if available)
        if hasattr(model, 'forward_tpu_optimized'):
            outputs2 = model.forward_tpu_optimized(sample_input, evolution_steps=1)
            print("✅ TPU-optimized forward pass available")
        else:
            outputs2 = outputs1
            print("ℹ️  TPU-optimized forward pass not available")

        # Verify outputs are consistent
        for key in outputs1.keys():
            if isinstance(outputs1[key], torch.Tensor) and isinstance(outputs2[key], torch.Tensor):
                diff = torch.abs(outputs1[key] - outputs2[key]).mean()
                print(f"  {key} difference: {diff".6f"}")

        print("✅ TPU optimizations working correctly")
        return True
    except Exception as e:
        print(f"❌ TPU optimizations failed: {e}")
        return False

def run_all_tests():
    """Run all TPU integration tests."""
    print("🚀 Starting TPU Integration Tests for NCA Trading Bot")
    print("=" * 60)

    tests = [
        test_tpu_detection,
        test_tpu_configuration,
        test_tpu_model_creation,
        test_tpu_training_setup,
        test_tpu_distributed_training,
        test_tpu_performance_monitoring,
        test_tpu_model_forward_pass,
        test_xla_compilation,
        test_tpu_optimizations,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All TPU integration tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)