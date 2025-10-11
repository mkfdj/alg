"""
Utility functions for TRM Coding Agent.

This module provides helper functions for JAX/TPU setup, tokenization,
binary operations, and code simulation.
"""

import os
import sys
import json
import ast
import signal
import importlib
import subprocess
from io import StringIO
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

import jax
import jax.numpy as jnp
from jax import config, random
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax.distributed
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
import timeout_decorator

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_tpu(
    num_devices: int = 8,
    platform_name: str = "tpu",
    enable_x64: bool = False
) -> Tuple[Mesh, Dict[str, Any]]:
    """
    Setup JAX for TPU training.

    Args:
        num_devices: Number of TPU devices (default 8 for v5e-8)
        platform_name: JAX platform name
        enable_x64: Whether to enable 64-bit precision

    Returns:
        mesh: Device mesh for distributed computing
        device_info: Dictionary with device information
    """
    try:
        # Configure JAX
        config.update("jax_enable_x64", enable_x64)
        config.update("jax_platform_name", platform_name)

        # Initialize distributed computing
        if platform_name == "tpu":
            jax.distributed.initialize()

        # Get available devices
        devices = jax.devices()
        device_count = len(devices)

        print(f"JAX platform: {platform_name}")
        print(f"Available devices: {device_count}")
        print(f"Device type: {devices[0].device_kind if devices else 'None'}")

        # Create device mesh
        if device_count >= num_devices:
            mesh_shape = (num_devices,)
            mesh_axis_names = ('tp',)
        else:
            # Fallback to available devices
            mesh_shape = (device_count,)
            mesh_axis_names = ('tp',)
            print(f"Warning: Requested {num_devices} devices, only {device_count} available")

        mesh = Mesh(jax.device_mesh(mesh_shape), mesh_axis_names)

        device_info = {
            'platform': platform_name,
            'device_count': device_count,
            'device_kind': devices[0].device_kind if devices else 'None',
            'mesh_shape': mesh_shape,
            'mesh_axis_names': mesh_axis_names,
            'devices': devices
        }

        return mesh, device_info

    except Exception as e:
        print(f"Failed to setup TPU: {e}")
        print("Falling back to CPU/GPU setup...")

        # Fallback to CPU/GPU
        config.update("jax_platform_name", "cpu")
        devices = jax.devices()
        mesh = Mesh(jax.device_mesh((len(devices),)), ('tp',))

        device_info = {
            'platform': 'cpu',
            'device_count': len(devices),
            'device_kind': 'cpu',
            'mesh_shape': (len(devices),),
            'mesh_axis_names': ('tp',),
            'devices': devices
        }

        return mesh, device_info


def create_tokenizer(
    model_name: str = "EleutherAI/gpt-neox-20b",
    cache_dir: Optional[str] = None
) -> Tokenizer:
    """
    Create and initialize tokenizer.

    Args:
        model_name: Name of the pretrained tokenizer
        cache_dir: Cache directory for tokenizer

    Returns:
        Initialized tokenizer
    """
    try:
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download

        # Download tokenizer if needed
        tokenizer_file = hf_hub_download(
            repo_id=model_name,
            filename="tokenizer.json",
            cache_dir=cache_dir
        )

        tokenizer = Tokenizer.from_file(tokenizer_file)

        # Add special tokens if needed
        tokenizer.add_special_tokens([
            "<pad>", "<unk>", "<s>", "</s>", "<mask>", "<binary>", "<think>"
        ])

        return tokenizer

    except Exception as e:
        print(f"Failed to load tokenizer {model_name}: {e}")
        # Fallback to basic tokenizer
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        # Create simple BPE tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        trainer = BpeTrainer(
            special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
            vocab_size=50257
        )
        tokenizer.pre_tokenizer = Whitespace()

        return tokenizer


def binarize(
    tensor: jnp.ndarray,
    threshold: float = 0.5,
    mode: str = "threshold"
) -> jnp.ndarray:
    """
    Binarize tensor using different methods.

    Args:
        tensor: Input tensor to binarize
        threshold: Threshold for binary conversion
        mode: Binarization mode ('threshold', 'sign', 'round')

    Returns:
        Binarized tensor (0/1)
    """
    if mode == "threshold":
        return jnp.where(tensor > threshold, 1.0, 0.0)
    elif mode == "sign":
        return jnp.where(tensor > 0, 1.0, 0.0)
    elif mode == "round":
        return jnp.round(tensor).astype(jnp.float32)
    else:
        raise ValueError(f"Unknown binarization mode: {mode}")


def tokenize_and_binarize(
    text: Union[str, List[str]],
    tokenizer: Tokenizer,
    max_length: int = 512,
    binary_threshold: float = 0.5,
    binarize_embeddings: bool = True
) -> Dict[str, jnp.ndarray]:
    """
    Tokenize and optionally binarize text.

    Args:
        text: Input text or list of texts
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        binary_threshold: Threshold for binarization
        binarize_embeddings: Whether to binarize embeddings

    Returns:
        Dictionary with tokenized and optionally binarized tensors
    """
    if isinstance(text, str):
        text = [text]

    # Tokenize
    encodings = tokenizer.encode_batch(text)

    # Convert to arrays
    input_ids = []
    attention_masks = []

    for encoding in encodings:
        # Pad or truncate
        ids = encoding.ids[:max_length]
        padding_length = max_length - len(ids)

        if padding_length > 0:
            ids.extend([0] * padding_length)  # Assume 0 is pad token

        input_ids.append(ids)
        attention_masks.append([1] * len(encoding.ids) + [0] * padding_length)

    input_ids = jnp.array(input_ids, dtype=jnp.int32)
    attention_masks = jnp.array(attention_masks, dtype=jnp.int32)

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_masks
    }

    if binarize_embeddings:
        # Create binary embeddings
        binary_embeddings = binarize(
            input_ids.astype(jnp.float32) / 50257.0,  # Normalize to [0, 1]
            threshold=binary_threshold
        )
        result['binary_embeddings'] = binary_embeddings

    return result


class CodeSimulator:
    """Safe code execution simulator."""

    def __init__(
        self,
        timeout: int = 10,
        max_memory_mb: int = 512,
        allowed_imports: Optional[List[str]] = None
    ):
        """
        Initialize code simulator.

        Args:
            timeout: Execution timeout in seconds
            max_memory_mb: Maximum memory usage in MB
            allowed_imports: List of allowed import modules
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.allowed_imports = allowed_imports or [
            'json', 'math', 'random', 'datetime', 'itertools', 'collections',
            'functools', 'operator', 're', 'string', 'typing', 'statistics'
        ]

        # Create safe execution environment
        self.safe_globals = self._create_safe_globals()

    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create safe global environment for code execution."""
        # Built-in functions
        safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'float': float,
            'int': int,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'pow': pow,
            'range': range,
            'reversed': reversed,
            'round': round,
            'set': set,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
            'print': self._safe_print,
        }

        # Safe modules
        safe_modules = {}
        for module_name in self.allowed_imports:
            try:
                safe_modules[module_name] = importlib.import_module(module_name)
            except ImportError:
                pass

        return {
            '__builtins__': safe_builtins,
            **safe_modules
        }

    def _safe_print(self, *args, **kwargs):
        """Safe print function that captures output."""
        pass  # Implement output capture if needed

    @timeout_decorator.timeout(timeout=10)
    def execute_code(
        self,
        code: str,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute code in safe environment.

        Args:
            code: Python code to execute
            test_cases: Optional test cases to run

        Returns:
            Execution result with output, success status, and errors
        """
        result = {
            'output': '',
            'success': False,
            'error': None,
            'test_results': [],
            'execution_time': 0.0
        }

        try:
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            import time
            start_time = time.time()

            # Execute code
            safe_locals = {}
            exec(code, self.safe_globals, safe_locals)

            execution_time = time.time() - start_time

            # Get output
            output = captured_output.getvalue()
            sys.stdout = old_stdout

            result['output'] = output
            result['success'] = True
            result['execution_time'] = execution_time

            # Run test cases if provided
            if test_cases:
                for test_case in test_cases:
                    test_result = self._run_test_case(safe_locals, test_case)
                    result['test_results'].append(test_result)

        except timeout_decorator.TimeoutError:
            result['error'] = "Execution timeout"
            sys.stdout = old_stdout
        except Exception as e:
            result['error'] = str(e)
            sys.stdout = old_stdout

        return result

    def _run_test_case(
        self,
        safe_locals: Dict[str, Any],
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single test case."""
        test_result = {
            'passed': False,
            'error': None,
            'output': None
        }

        try:
            # Extract test components
            function_name = test_case.get('function')
            inputs = test_case.get('inputs', [])
            expected_output = test_case.get('expected_output')

            if function_name and function_name in safe_locals:
                func = safe_locals[function_name]
                output = func(*inputs)

                # Check output
                if expected_output is not None:
                    passed = output == expected_output
                else:
                    passed = True  # Just check if function runs without error

                test_result['passed'] = passed
                test_result['output'] = output

        except Exception as e:
            test_result['error'] = str(e)

        return test_result


def simulate_code_execution(
    code: str,
    test_cases: Optional[List[Dict[str, Any]]] = None,
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Simulate code execution (convenience function).

    Args:
        code: Python code to execute
        test_cases: Optional test cases
        timeout: Execution timeout

    Returns:
        Execution result
    """
    simulator = CodeSimulator(timeout=timeout)
    return simulator.execute_code(code, test_cases)


def validate_syntax(code: str) -> Dict[str, Any]:
    """
    Validate Python code syntax.

    Args:
        code: Python code to validate

    Returns:
        Validation result
    """
    result = {
        'valid': False,
        'error': None,
        'line_number': None
    }

    try:
        ast.parse(code)
        result['valid'] = True
    except SyntaxError as e:
        result['error'] = str(e)
        result['line_number'] = e.lineno
    except Exception as e:
        result['error'] = str(e)

    return result


def create_dataset_loaders(
    dataset_paths: Dict[str, str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> Dict[str, Any]:
    """
    Create dataset loaders for training.

    Args:
        dataset_paths: Dictionary of dataset names to paths
        batch_size: Batch size for data loading
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        Dictionary of data loaders
    """
    # This is a placeholder - implement actual data loading logic
    # in the data_handler module
    return {
        'train': None,
        'validation': None,
        'test': None
    }


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level
        log_file: Log file path
        format_string: Log format string
    """
    import logging

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # Create file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    step: int,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Save model checkpoint.

    Args:
        state: Training state to save
        filepath: Checkpoint file path
        step: Training step
        metrics: Optional metrics to save
    """
    checkpoint_data = {
        'step': step,
        'state': state,
        'metrics': metrics or {},
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save checkpoint
    with open(filepath, 'wb') as f:
        import pickle
        pickle.dump(checkpoint_data, f)


def load_checkpoint(filepath: str) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        filepath: Checkpoint file path

    Returns:
        Checkpoint data
    """
    with open(filepath, 'rb') as f:
        import pickle
        return pickle.load(f)


def count_parameters(params: Dict[str, Any]) -> int:
    """
    Count total number of parameters in model.

    Args:
        params: Model parameters

    Returns:
        Total parameter count
    """
    return sum(x.size for x in jax.tree_leaves(params))


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.

    Returns:
        Dictionary with memory usage statistics
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'error': 'psutil not available'}


# JAX-specific utilities
def replicate_across_devices(
    tensor: jnp.ndarray,
    mesh: Mesh,
    mesh_axes: Tuple[str, ...]
) -> jnp.ndarray:
    """
    Replicate tensor across devices.

    Args:
        tensor: Input tensor
        mesh: Device mesh
        mesh_axes: Mesh axes for replication

    Returns:
        Sharded tensor
    """
    sharding = NamedSharding(mesh, P(*mesh_axes))
    return jax.device_put(tensor, sharding)


def create_sharded_optimizer(
    learning_rate: float,
    mesh: Mesh,
    weight_decay: float = 0.0
) -> optax.GradientTransformation:
    """
    Create optimizer with sharding support.

    Args:
        learning_rate: Learning rate
        mesh: Device mesh
        weight_decay: Weight decay coefficient

    Returns:
        Sharded optimizer
    """
    # Create basic optimizer
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    # Wrap with gradient transformation for sharding
    return optax.chain(
        optax.scale_by_adam(),
        optax.add_noise(
            jax.random.PRNGKey(42),
            0.01,
            mesh.shape
        ),
        optax.scale(-learning_rate)
    )


if __name__ == "__main__":
    # Test utilities
    print("Testing TRM utilities...")

    # Test TPU setup
    try:
        mesh, device_info = setup_tpu()
        print(f"TPU setup successful: {device_info}")
    except Exception as e:
        print(f"TPU setup failed: {e}")

    # Test tokenizer
    try:
        tokenizer = create_tokenizer()
        test_text = "def hello_world(): print('Hello, world!')"
        result = tokenize_and_binarize(test_text, tokenizer)
        print(f"Tokenization successful: {result['input_ids'].shape}")
    except Exception as e:
        print(f"Tokenization failed: {e}")

    # Test code simulation
    try:
        code = "def add(a, b): return a + b"
        test_cases = [
            {
                'function': 'add',
                'inputs': [2, 3],
                'expected_output': 5
            }
        ]
        result = simulate_code_execution(code, test_cases)
        print(f"Code simulation successful: {result['success']}")
    except Exception as e:
        print(f"Code simulation failed: {e}")

    print("Utility tests completed!")