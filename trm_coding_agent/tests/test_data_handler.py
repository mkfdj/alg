"""
Tests for data handler implementation.
"""

import pytest
import tempfile
import json
import pandas as pd
from pathlib import Path

from trm_coding_agent.config import get_config
from trm_coding_agent.data_handler import (
    DataHandler, DatasetSample, HumanEvalLoader, MBPPLoader,
    AlpacaLoader, GlaiveQLoader
)


class TestDatasetLoaders:
    """Test cases for individual dataset loaders."""

    @pytest.fixture
    def config(self):
        """Get test configuration."""
        return get_config("debug")

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_human_eval_loader(self, config, temp_dir):
        """Test HumanEval CSV loader."""
        # Create sample CSV file
        csv_data = {
            'task_id': ['HumanEval/0', 'HumanEval/1'],
            'prompt': [
                'def has_close_elements(numbers, threshold):\n    """Check for close elements."""',
                'def separate_paren_groups(paren_string):\n    """Separate parentheses groups."""'
            ],
            'canonical_solution': [
                'def has_close_elements(numbers, threshold):\n    return True',
                'def separate_paren_groups(paren_string):\n    return []'
            ],
            'test': [
                'assert has_close_elements([1,2], 1) == True',
                'assert separate_paren_groups("()") == [""]'
            ],
            'entry_point': ['has_close_elements', 'separate_paren_groups']
        }

        csv_path = temp_dir / "test_humaneval.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)

        # Test loader
        loader = HumanEvalLoader(config.dataset)

        assert loader.validate_format(csv_path)
        assert loader.get_sample_count(csv_path) == 2

        samples = loader.load(csv_path)
        assert len(samples) == 2
        assert all(isinstance(s, DatasetSample) for s in samples)

        # Check first sample
        sample = samples[0]
        assert sample.prompt.startswith('def has_close_elements')
        assert sample.solution == 'def has_close_elements(numbers, threshold):\n    return True'
        assert sample.validation == 'assert has_close_elements([1,2], 1) == True'
        assert sample.metadata['dataset'] == 'humaneval'

    def test_mbpp_loader(self, config, temp_dir):
        """Test MBPP JSONL loader."""
        # Create sample JSONL file
        jsonl_data = [
            {
                "task_id": 1,
                "text": "Write a function to add two numbers",
                "code": "def add(a, b): return a + b",
                "test_list": [
                    "assert add(2, 3) == 5",
                    "assert add(0, 0) == 0"
                ],
                "test_setup_code": "",
                "challenge_test_list": []
            },
            {
                "task_id": 2,
                "text": "Write a function to multiply two numbers",
                "code": "def multiply(a, b): return a * b",
                "test_list": [
                    "assert multiply(2, 3) == 6",
                    "assert multiply(-1, 5) == -5"
                ],
                "test_setup_code": "",
                "challenge_test_list": []
            }
        ]

        jsonl_path = temp_dir / "test_mbpp.jsonl"
        with open(jsonl_path, 'w') as f:
            for item in jsonl_data:
                f.write(json.dumps(item) + '\n')

        # Test loader
        loader = MBPPLoader(config.dataset)

        assert loader.validate_format(jsonl_path)
        assert loader.get_sample_count(jsonl_path) == 2

        samples = loader.load(jsonl_path)
        assert len(samples) == 2

        # Check first sample
        sample = samples[0]
        assert sample.prompt == "Write a function to add two numbers"
        assert sample.solution == "def add(a, b): return a + b"
        assert "assert add(2, 3) == 5" in sample.validation
        assert sample.metadata['dataset'] == 'mbpp'

    def test_alpaca_loader(self, config, temp_dir):
        """Test Alpaca JSON loader."""
        # Create sample JSON file
        json_data = {
            "data": [
                {
                    "instruction": "Write a function to calculate factorial",
                    "input": "",
                    "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
                },
                {
                    "instruction": "Write a function to check if a number is prime",
                    "input": "",
                    "output": "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
                }
            ]
        }

        json_path = temp_dir / "test_alpaca.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

        # Test loader
        loader = AlpacaLoader(config.dataset)

        assert loader.validate_format(json_path)
        assert loader.get_sample_count(json_path) == 2

        samples = loader.load(json_path)
        assert len(samples) == 2

        # Check first sample
        sample = samples[0]
        assert sample.prompt == "Write a function to calculate factorial"
        assert "def factorial" in sample.solution
        assert sample.metadata['dataset'] == 'alpaca_python'

    def test_glaive_qa_loader(self, config, temp_dir):
        """Test Glaive QA CSV loader."""
        # Create sample CSV file
        csv_data = {
            'question': [
                'How do I create a list comprehension in Python?',
                'What is the difference between list and tuple?'
            ],
            'answer': [
                '[x for x in range(10)]',
                'Lists are mutable, tuples are immutable'
            ]
        }

        csv_path = temp_dir / "test_glaive.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)

        # Test loader
        loader = GlaiveQLoader(config.dataset)

        assert loader.validate_format(csv_path)
        assert loader.get_sample_count(csv_path) == 2

        samples = loader.load(csv_path)
        assert len(samples) == 2

        # Check first sample
        sample = samples[0]
        assert sample.prompt == 'How do I create a list comprehension in Python?'
        assert sample.answer == '[x for x in range(10)]'
        assert sample.metadata['dataset'] == 'glaive_qa'


class TestDataHandler:
    """Test cases for main data handler."""

    @pytest.fixture
    def config(self):
        """Get test configuration."""
        return get_config("debug")

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def data_handler(self, config):
        """Create data handler instance."""
        return DataHandler(config)

    def test_data_handler_initialization(self, data_handler):
        """Test data handler initialization."""
        assert data_handler.config is not None
        assert data_handler.loaders is not None
        assert len(data_handler.loaders) > 0

    def test_binary_decision_generation(self, data_handler):
        """Test binary decision generation."""
        # Simple prompt
        simple_prompt = "Write a hello world function"
        binary_decision = data_handler._generate_binary_decision(simple_prompt)
        assert binary_decision in [0, 1]

        # Complex prompt
        complex_prompt = "Write a complex function to optimize recursive sorting algorithms with error handling and debugging"
        binary_decision = data_handler._generate_binary_decision(complex_prompt)
        assert binary_decision in [0, 1]

    def test_tool_suggestions(self, data_handler):
        """Test tool suggestions."""
        # Sample with error
        error_sample = DatasetSample(
            prompt="Fix the error in this function",
            solution="def broken(): error",
            validation=None,
            binary_decision=1,
            metadata={}
        )

        tools = data_handler._suggest_tools(error_sample)
        assert 'syntax_checker' in tools or 'debugger' in tools

        # Sample with test
        test_sample = DatasetSample(
            prompt="Write a function with tests",
            solution="def func(): pass",
            validation="assert func() is None",
            binary_decision=0,
            metadata={}
        )

        tools = data_handler._suggest_tools(test_sample)
        assert 'test_runner' in tools

    def test_data_splitting(self, data_handler):
        """Test data splitting."""
        # Create dummy samples
        samples = [
            DatasetSample(
                prompt=f"Sample {i}",
                solution=f"def sample_{i}(): pass",
                validation=None,
                binary_decision=i % 2,
                metadata={}
            )
            for i in range(100)
        ]

        train_samples, val_samples, test_samples = data_handler._split_data(samples)

        # Check split ratios
        total = len(samples)
        expected_val = int(total * data_handler.config.dataset.validation_split)
        expected_test = int(total * data_handler.config.dataset.test_split)
        expected_train = total - expected_val - expected_test

        assert len(train_samples) == expected_train
        assert len(val_samples) == expected_val
        assert len(test_samples) == expected_test

        # Check no overlap
        all_indices = set(range(total))
        train_indices = {i for i, s in enumerate(samples) if s in train_samples}
        val_indices = {i for i, s in enumerate(samples) if s in val_samples}
        test_indices = {i for i, s in enumerate(samples) if s in test_samples}

        assert len(train_indices & val_indices) == 0
        assert len(train_indices & test_indices) == 0
        assert len(val_indices & test_indices) == 0

    def test_tokenization(self, data_handler):
        """Test tokenization functionality."""
        # Setup tokenizer
        data_handler.setup_tokenizer()

        # Create dummy samples
        samples = [
            DatasetSample(
                prompt="def hello(): print('Hello')",
                solution="def hello(): print('Hello')",
                validation=None,
                binary_decision=0,
                metadata={}
            )
        ]

        # Tokenize
        tokenized = data_handler.tokenize_samples(samples, max_length=64)

        # Check tokenized data
        assert 'prompt_input_ids' in tokenized
        assert 'prompt_attention_mask' in tokenized
        assert 'solution_input_ids' in tokenized
        assert 'solution_attention_mask' in tokenized
        assert 'binary_decisions' in tokenized

        # Check shapes
        assert tokenized['prompt_input_ids'].shape == (1, 64)
        assert tokenized['binary_decisions'].shape == (1,)

    def test_batch_creation(self, data_handler):
        """Test batch creation."""
        # Create dummy tokenized data
        tokenized = {
            'prompt_input_ids': [[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]],
            'prompt_attention_mask': [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
            'binary_decisions': [0, 1]
        }

        # Convert to JAX arrays
        import jax.numpy as jnp
        for key in tokenized:
            tokenized[key] = jnp.array(tokenized[key])

        # Create batches
        batches = data_handler.create_batches(tokenized, batch_size=2, shuffle=False)

        assert len(batches) == 1
        batch = batches[0]
        assert batch['prompt_input_ids'].shape[0] == 2

    def test_dataset_statistics(self, data_handler):
        """Test dataset statistics."""
        # Add some cached data
        data_handler.cached_data = {
            'test_dataset_10': [
                DatasetSample(
                    prompt="Short prompt",
                    solution="def short(): pass",
                    validation=None,
                    binary_decision=0,
                    metadata={}
                )
            ]
        }

        stats = data_handler.get_dataset_statistics()

        assert 'total_samples' in stats
        assert 'datasets' in stats
        assert 'binary_decision_distribution' in stats
        assert stats['total_samples'] == 1

    def test_generated_code_validation(self, data_handler):
        """Test generated code validation."""
        # Valid code
        valid_code = "def add(a, b): return a + b"
        validation = "assert add(2, 3) == 5"

        result = data_handler.validate_generated_code(
            valid_code,
            validation,
            DatasetSample("", "", "", 0, {})
        )

        assert 'syntax_valid' in result
        assert 'validation_passed' in result
        assert 'error' in result

        # Invalid code
        invalid_code = "def broken(\n    # Missing closing parenthesis"

        result = data_handler.validate_generated_code(
            invalid_code,
            None,
            DatasetSample("", "", "", 0, {})
        )

        assert not result['syntax_valid']
        assert result['error'] is not None


class TestDataHandlerIntegration:
    """Integration tests for data handler."""

    def test_full_pipeline(self):
        """Test full data loading pipeline."""
        config = get_config("debug")
        data_handler = DataHandler(config)

        # Create temporary dataset files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create sample HumanEval CSV
            csv_data = {
                'task_id': ['HumanEval/0'],
                'prompt': ['def test_func():\n    """Test function"""'],
                'canonical_solution': ['def test_func():\n    return True'],
                'test': ['assert test_func() == True'],
                'entry_point': ['test_func']
            }
            csv_path = temp_path / "test.csv"
            pd.DataFrame(csv_data).to_csv(csv_path, index=False)

            # Mock dataset path
            original_datasets = data_handler.dataset_config.datasets
            data_handler.dataset_config.datasets = [
                {
                    'name': 'test_dataset',
                    'path': 'test.csv',
                    'format': 'csv',
                    'url': '',
                    'enabled': True
                }
            ]

            # Temporarily modify dataset directory
            original_dir = data_handler.dataset_config.dataset_dir
            data_handler.dataset_config.dataset_dir = str(temp_path)

            try:
                # Load data
                train_samples, val_samples, test_samples = data_handler.load_datasets(
                    dataset_names=['test_dataset'],
                    max_samples=10
                )

                assert len(train_samples) > 0
                assert all(isinstance(s, DatasetSample) for s in train_samples)

                # Tokenize
                data_handler.setup_tokenizer()
                tokenized = data_handler.tokenize_samples(train_samples[:1])

                assert 'prompt_input_ids' in tokenized

                # Create batches
                batches = data_handler.create_batches(tokenized, batch_size=1)
                assert len(batches) > 0

            finally:
                # Restore original configuration
                data_handler.dataset_config.datasets = original_datasets
                data_handler.dataset_config.dataset_dir = original_dir


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__])