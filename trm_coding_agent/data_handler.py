"""
Data handler for TRM Coding Agent datasets.

This module provides unified loading and processing for 10+ coding datasets,
including OpenAI HumanEval, MBPP, codeparrot, Alpaca Python, and more.
Supports binary thinking augmentation and tool integration.
"""

import os
import json
import csv
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

import jax.numpy as jnp
import numpy as np
from tokenizers import Tokenizer

from .config import DatasetConfig, Config
from .utils import create_tokenizer, binarize, simulate_code_execution, validate_syntax

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class DatasetSample:
    """Single dataset sample with prompt, solution, and validation."""
    prompt: str
    solution: Optional[str]
    validation: Optional[str]
    binary_decision: int
    metadata: Dict[str, Any]


class DatasetLoader:
    """Base class for dataset loaders."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.tokenizer = None

    def load(self, path: str) -> List[DatasetSample]:
        """Load dataset from path."""
        raise NotImplementedError

    def validate_format(self, path: str) -> bool:
        """Validate dataset format."""
        raise NotImplementedError

    def get_sample_count(self, path: str) -> int:
        """Get number of samples in dataset."""
        raise NotImplementedError


class HumanEvalLoader(DatasetLoader):
    """Loader for OpenAI HumanEval datasets."""

    def load(self, path: str) -> List[DatasetSample]:
        """Load HumanEval CSV format."""
        samples = []

        try:
            df = pd.read_csv(path)

            for _, row in df.iterrows():
                prompt = row.get('prompt', '')
                solution = row.get('canonical_solution', '')
                test = row.get('test', '')
                entry_point = row.get('entry_point', '')

                # Generate binary decision based on prompt complexity
                binary_decision = 1 if len(prompt) > 200 else 0

                sample = DatasetSample(
                    prompt=prompt,
                    solution=solution,
                    validation=test,
                    binary_decision=binary_decision,
                    metadata={
                        'entry_point': entry_point,
                        'dataset': 'humaneval',
                        'format': 'csv'
                    }
                )
                samples.append(sample)

        except Exception as e:
            print(f"Error loading HumanEval CSV: {e}")

        return samples

    def validate_format(self, path: str) -> bool:
        """Validate CSV format with required columns."""
        try:
            df = pd.read_csv(path)
            required_columns = ['prompt', 'canonical_solution', 'test', 'entry_point']
            return all(col in df.columns for col in required_columns)
        except:
            return False

    def get_sample_count(self, path: str) -> int:
        """Get number of samples."""
        try:
            df = pd.read_csv(path)
            return len(df)
        except:
            return 0


class HumanEvalJSONLLoader(DatasetLoader):
    """Loader for OpenAI HumanEval JSONL format."""

    def load(self, path: str) -> List[DatasetSample]:
        """Load HumanEval JSONL format."""
        samples = []

        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)

                        prompt = data.get('prompt', '')
                        entry_point = data.get('entry_point', '')
                        test = data.get('test', '')

                        # Generate binary decision
                        binary_decision = 1 if 'complex' in prompt.lower() else 0

                        sample = DatasetSample(
                            prompt=prompt,
                            solution=None,  # No solution in original HumanEval
                            validation=test,
                            binary_decision=binary_decision,
                            metadata={
                                'entry_point': entry_point,
                                'dataset': 'humaneval_jsonl',
                                'format': 'jsonl'
                            }
                        )
                        samples.append(sample)

        except Exception as e:
            print(f"Error loading HumanEval JSONL: {e}")

        return samples

    def validate_format(self, path: str) -> bool:
        """Validate JSONL format."""
        try:
            with open(path, 'r') as f:
                first_line = f.readline()
                if first_line.strip():
                    data = json.loads(first_line)
                    required_keys = ['prompt', 'entry_point', 'test']
                    return all(key in data for key in required_keys)
        except:
            pass
        return False

    def get_sample_count(self, path: str) -> int:
        """Count lines in JSONL file."""
        try:
            with open(path, 'r') as f:
                return sum(1 for line in f if line.strip())
        except:
            return 0


class MBPPLoader(DatasetLoader):
    """Loader for MBPP (Mostly Basic Python Problems)."""

    def load(self, path: str) -> List[DatasetSample]:
        """Load MBPP JSONL format."""
        samples = []

        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)

                        text = data.get('text', '')
                        code = data.get('code', '')
                        test_list = data.get('test_list', [])

                        # Combine test cases into validation string
                        validation = '\n'.join(test_list)

                        # Generate binary decision based on problem complexity
                        binary_decision = 1 if len(text) > 150 or len(test_list) > 3 else 0

                        sample = DatasetSample(
                            prompt=text,
                            solution=code,
                            validation=validation,
                            binary_decision=binary_decision,
                            metadata={
                                'task_id': data.get('task_id'),
                                'dataset': 'mbpp',
                                'format': 'jsonl',
                                'test_count': len(test_list)
                            }
                        )
                        samples.append(sample)

        except Exception as e:
            print(f"Error loading MBPP: {e}")

        return samples

    def validate_format(self, path: str) -> bool:
        """Validate MBPP JSONL format."""
        try:
            with open(path, 'r') as f:
                first_line = f.readline()
                if first_line.strip():
                    data = json.loads(first_line)
                    required_keys = ['text', 'code', 'test_list']
                    return all(key in data for key in required_keys)
        except:
            pass
        return False

    def get_sample_count(self, path: str) -> int:
        """Count lines in JSONL file."""
        try:
            with open(path, 'r') as f:
                return sum(1 for line in f if line.strip())
        except:
            return 0


class CodeParrotLoader(DatasetLoader):
    """Loader for CodeParrot dataset (Lance format)."""

    def load(self, path: str) -> List[DatasetSample]:
        """Load CodeParrot Lance format."""
        samples = []

        try:
            import lance

            # Open Lance dataset
            dataset = lance.dataset(path)

            # Convert to pandas for processing
            df = dataset.to_table().to_pandas()

            # Process in chunks for memory efficiency
            chunk_size = 1000
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]

                for _, row in chunk.iterrows():
                    content = row.get('content', '')

                    # Create completion prompt
                    lines = content.split('\n')
                    if len(lines) > 2:
                        split_point = len(lines) // 2
                        prompt = '\n'.join(lines[:split_point])
                        solution = '\n'.join(lines[split_point:])
                    else:
                        prompt = content[:100]
                        solution = content[100:]

                    # Generate binary decision
                    binary_decision = 1 if len(content) > 500 else 0

                    sample = DatasetSample(
                        prompt=f"Complete: {prompt}",
                        solution=solution,
                        validation=None,  # No validation provided
                        binary_decision=binary_decision,
                        metadata={
                            'dataset': 'codeparrot',
                            'format': 'lance',
                            'content_length': len(content)
                        }
                    )
                    samples.append(sample)

        except Exception as e:
            print(f"Error loading CodeParrot: {e}")

        return samples

    def validate_format(self, path: str) -> bool:
        """Validate Lance format."""
        try:
            import lance
            dataset = lance.dataset(path)
            return 'content' in dataset.to_table().column_names
        except:
            return False

    def get_sample_count(self, path: str) -> int:
        """Get number of samples."""
        try:
            import lance
            dataset = lance.dataset(path)
            return dataset.count_rows()
        except:
            return 0


class AlpacaLoader(DatasetLoader):
    """Loader for Alpaca Python instruction datasets."""

    def load(self, path: str) -> List[DatasetSample]:
        """Load Alpaca JSON format."""
        samples = []

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Handle different JSON structures
            items = data
            if 'data' in data:
                items = data['data']
            elif 'items' in data:
                items = data['items']

            for item in items:
                instruction = item.get('instruction', '')
                input_text = item.get('input', '')
                output = item.get('output', '')

                # Combine instruction and input for prompt
                if input_text:
                    prompt = f"{instruction}\nInput: {input_text}"
                else:
                    prompt = instruction

                # Generate binary decision
                binary_decision = 1 if len(instruction) > 100 or 'function' in instruction.lower() else 0

                sample = DatasetSample(
                    prompt=prompt,
                    solution=output,
                    validation=None,  # No explicit validation
                    binary_decision=binary_decision,
                    metadata={
                        'dataset': 'alpaca_python',
                        'format': 'json',
                        'has_input': bool(input_text)
                    }
                )
                samples.append(sample)

        except Exception as e:
            print(f"Error loading Alpaca: {e}")

        return samples

    def validate_format(self, path: str) -> bool:
        """Validate Alpaca JSON format."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            items = data
            if 'data' in data:
                items = data['data']
            elif 'items' in data:
                items = data['items']

            if items and isinstance(items, list):
                first_item = items[0]
                required_keys = ['instruction', 'output']
                return all(key in first_item for key in required_keys)
        except:
            pass
        return False

    def get_sample_count(self, path: str) -> int:
        """Get number of samples."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            items = data
            if 'data' in data:
                items = data['data']
            elif 'items' in data:
                items = data['items']

            return len(items) if isinstance(items, list) else 0
        except:
            return 0


class GlaiveQLoader(DatasetLoader):
    """Loader for Glaive Python Code QA dataset."""

    def load(self, path: str) -> List[DatasetSample]:
        """Load Glaive QA CSV format."""
        samples = []

        try:
            df = pd.read_csv(path)

            for _, row in df.iterrows():
                question = row.get('question', '')
                answer = row.get('answer', '')

                # Generate binary decision based on question complexity
                binary_decision = 1 if len(question) > 150 or 'error' in question.lower() else 0

                sample = DatasetSample(
                    prompt=question,
                    solution=answer,
                    validation=None,
                    binary_decision=binary_decision,
                    metadata={
                        'dataset': 'glaive_qa',
                        'format': 'csv'
                    }
                )
                samples.append(sample)

        except Exception as e:
            print(f"Error loading Glaive QA: {e}")

        return samples

    def validate_format(self, path: str) -> bool:
        """Validate Glaive QA CSV format."""
        try:
            df = pd.read_csv(path)
            required_columns = ['question', 'answer']
            return all(col in df.columns for col in required_columns)
        except:
            return False

    def get_sample_count(self, path: str) -> int:
        """Get number of samples."""
        try:
            df = pd.read_csv(path)
            return len(df)
        except:
            return 0


class LiveCodeBenchLoader(DatasetLoader):
    """Loader for LiveCodeBench dataset."""

    def load(self, path: str) -> List[DatasetSample]:
        """Load LiveCodeBench JSON format."""
        samples = []

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            problems = data.get('problems', data)  # Handle both structures

            for problem in problems:
                problem_text = problem.get('problem', '')
                solution = problem.get('solution', '')
                test = problem.get('test', '')

                # Generate binary decision
                binary_decision = 1 if 'algorithm' in problem_text.lower() or len(problem_text) > 300 else 0

                sample = DatasetSample(
                    prompt=problem_text,
                    solution=solution,
                    validation=test,
                    binary_decision=binary_decision,
                    metadata={
                        'dataset': 'livecodebench',
                        'format': 'json'
                    }
                )
                samples.append(sample)

        except Exception as e:
            print(f"Error loading LiveCodeBench: {e}")

        return samples

    def validate_format(self, path: str) -> bool:
        """Validate LiveCodeBench JSON format."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            problems = data.get('problems', data)
            if problems and isinstance(problems, list):
                first_problem = problems[0]
                required_keys = ['problem', 'solution', 'test']
                return all(key in first_problem for key in required_keys)
        except:
            pass
        return False

    def get_sample_count(self, path: str) -> int:
        """Get number of samples."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            problems = data.get('problems', data)
            return len(problems) if isinstance(problems, list) else 0
        except:
            return 0


class DataHandler:
    """Main data handler for all coding datasets."""

    def __init__(self, config: Config):
        self.config = config
        self.dataset_config = config.dataset
        self.tokenizer = None
        self.loaders = self._create_loaders()
        self.cached_data = {}

    def _create_loaders(self) -> Dict[str, DatasetLoader]:
        """Create dataset loaders."""
        return {
            'humaneval': HumanEvalLoader(self.dataset_config),
            'humaneval_jsonl': HumanEvalJSONLLoader(self.dataset_config),
            'mbpp': MBPPLoader(self.dataset_config),
            'codeparrot': CodeParrotLoader(self.dataset_config),
            'alpaca': AlpacaLoader(self.dataset_config),
            'glaive_qa': GlaiveQLoader(self.dataset_config),
            'livecodebench': LiveCodeBenchLoader(self.dataset_config),
        }

    def setup_tokenizer(self, model_name: str = "EleutherAI/gpt-neox-20b"):
        """Setup tokenizer."""
        self.tokenizer = create_tokenizer(model_name)

    def load_datasets(
        self,
        dataset_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        binary_augmentation: bool = True
    ) -> Tuple[List[DatasetSample], List[DatasetSample], List[DatasetSample]]:
        """
        Load specified datasets.

        Args:
            dataset_names: List of dataset names to load
            max_samples: Maximum samples per dataset
            binary_augmentation: Whether to add binary thinking augmentation

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        if dataset_names is None:
            dataset_names = [d['name'] for d in self.dataset_config.datasets if d['enabled']]

        if self.tokenizer is None:
            self.setup_tokenizer()

        all_samples = []

        for dataset_name in dataset_names:
            dataset_info = self._get_dataset_info(dataset_name)
            if not dataset_info or not dataset_info['enabled']:
                continue

            samples = self._load_single_dataset(dataset_info, max_samples)

            if binary_augmentation:
                samples = self._add_binary_augmentation(samples)

            all_samples.extend(samples)

        # Split into train/val/test
        train_samples, val_samples, test_samples = self._split_data(all_samples)

        return train_samples, val_samples, test_samples

    def _get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get dataset configuration."""
        for dataset in self.dataset_config.datasets:
            if dataset['name'] == dataset_name:
                return dataset
        return None

    def _load_single_dataset(
        self,
        dataset_info: Dict[str, Any],
        max_samples: Optional[int] = None
    ) -> List[DatasetSample]:
        """Load a single dataset."""
        dataset_path = os.path.join(self.dataset_config.dataset_dir, dataset_info['path'])
        format_type = dataset_info['format']

        # Check cache first
        cache_key = f"{dataset_info['name']}_{max_samples}"
        if cache_key in self.cached_data:
            return self.cached_data[cache_key]

        # Get appropriate loader
        loader = self.loaders.get(dataset_info['name'])
        if loader is None:
            # Fallback by format
            if format_type == 'csv':
                loader = HumanEvalLoader(self.dataset_config)
            elif format_type == 'jsonl':
                loader = HumanEvalJSONLLoader(self.dataset_config)
            elif format_type == 'json':
                loader = AlpacaLoader(self.dataset_config)
            elif format_type == 'lance':
                loader = CodeParrotLoader(self.dataset_config)
            else:
                print(f"No loader found for {dataset_info['name']} ({format_type})")
                return []

        # Validate dataset format
        if not loader.validate_format(dataset_path):
            print(f"Invalid format for {dataset_info['name']}")
            return []

        # Load samples
        samples = loader.load(dataset_path)

        # Limit samples if specified
        if max_samples is not None:
            samples = samples[:max_samples]

        # Cache results
        self.cached_data[cache_key] = samples

        print(f"Loaded {len(samples)} samples from {dataset_info['name']}")

        return samples

    def _add_binary_augmentation(self, samples: List[DatasetSample]) -> List[DatasetSample]:
        """Add binary thinking augmentation to samples."""
        augmented_samples = []

        for sample in samples:
            # Create binary decision based on prompt complexity
            if self.dataset_config.use_synthetic_binary_decisions:
                binary_decision = self._generate_binary_decision(sample.prompt)
                sample.binary_decision = binary_decision

            # Add tool simulation metadata
            sample.metadata['tool_suggestions'] = self._suggest_tools(sample)

            augmented_samples.append(sample)

        return augmented_samples

    def _generate_binary_decision(self, prompt: str) -> int:
        """Generate binary decision for refinement logic."""
        # Decision factors
        length_factor = len(prompt) > self.dataset_config.binary_decision_threshold
        complexity_keywords = any(keyword in prompt.lower() for keyword in [
            'complex', 'difficult', 'optimize', 'debug', 'error', 'fix', 'improve'
        ])
        function_count = prompt.count('def') > 1

        # Binary decision: 1 = needs refinement, 0 = accept as is
        if length_factor or complexity_keywords or function_count:
            return 1
        else:
            return 0

    def _suggest_tools(self, sample: DatasetSample) -> List[str]:
        """Suggest tools for processing the sample."""
        tools = []

        prompt = sample.prompt.lower()
        solution = sample.solution or ""

        # Syntax checking
        if 'error' in prompt or 'fix' in prompt:
            tools.append('syntax_checker')

        # Debugging
        if 'debug' in prompt or 'error' in prompt:
            tools.append('debugger')

        # Testing
        if 'test' in prompt or sample.validation:
            tools.append('test_runner')

        # Documentation
        if 'explain' in prompt or 'document' in prompt:
            tools.append('documentation_search')

        # Import resolution
        if 'import' in solution:
            tools.append('import_resolver')

        return tools

    def _split_data(
        self,
        samples: List[DatasetSample]
    ) -> Tuple[List[DatasetSample], List[DatasetSample], List[DatasetSample]]:
        """Split data into train/val/test sets."""
        total_samples = len(samples)

        # Calculate split sizes
        val_size = int(total_samples * self.dataset_config.validation_split)
        test_size = int(total_samples * self.dataset_config.test_split)
        train_size = total_samples - val_size - test_size

        # Shuffle samples
        np.random.shuffle(samples)

        # Split
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]

        return train_samples, val_samples, test_samples

    def tokenize_samples(
        self,
        samples: List[DatasetSample],
        max_length: Optional[int] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Tokenize dataset samples.

        Args:
            samples: List of dataset samples
            max_length: Maximum sequence length

        Returns:
            Dictionary with tokenized arrays
        """
        if self.tokenizer is None:
            self.setup_tokenizer()

        if max_length is None:
            max_length = self.config.max_sequence_length

        prompts = [sample.prompt for sample in samples]
        solutions = [sample.solution or "" for sample in samples]
        binary_decisions = [sample.binary_decision for sample in samples]

        # Tokenize prompts
        prompt_data = self._tokenize_texts(prompts, max_length)

        # Tokenize solutions if available
        solution_data = self._tokenize_texts(solutions, max_length)

        # Convert binary decisions to arrays
        binary_array = jnp.array(binary_decisions, dtype=jnp.int32)

        result = {
            'prompt_input_ids': prompt_data['input_ids'],
            'prompt_attention_mask': prompt_data['attention_mask'],
            'solution_input_ids': solution_data['input_ids'],
            'solution_attention_mask': solution_data['attention_mask'],
            'binary_decisions': binary_array
        }

        # Add binary embeddings if enabled
        if self.config.trm.binary_binarization:
            result['prompt_binary_embeddings'] = binarize(
                prompt_data['input_ids'].astype(jnp.float32) / 50257.0,
                threshold=self.config.trm.binary_threshold
            )
            result['solution_binary_embeddings'] = binarize(
                solution_data['input_ids'].astype(jnp.float32) / 50257.0,
                threshold=self.config.trm.binary_threshold
            )

        return result

    def _tokenize_texts(self, texts: List[str], max_length: int) -> Dict[str, jnp.ndarray]:
        """Tokenize a list of texts."""
        # Batch tokenize
        encodings = self.tokenizer.encode_batch(texts)

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

        return {
            'input_ids': jnp.array(input_ids, dtype=jnp.int32),
            'attention_mask': jnp.array(attention_masks, dtype=jnp.int32)
        }

    def create_batches(
        self,
        tokenized_data: Dict[str, jnp.ndarray],
        batch_size: int,
        shuffle: bool = True
    ) -> List[Dict[str, jnp.ndarray]]:
        """
        Create batches from tokenized data.

        Args:
            tokenized_data: Tokenized data dictionary
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            List of batch dictionaries
        """
        num_samples = len(tokenized_data['prompt_input_ids'])
        indices = np.arange(num_samples)

        if shuffle:
            np.random.shuffle(indices)

        batches = []

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]

            batch = {}
            for key, value in tokenized_data.items():
                batch[key] = value[batch_indices]

            batches.append(batch)

        return batches

    def validate_generated_code(
        self,
        generated_code: str,
        validation: Optional[str] = None,
        sample: Optional[DatasetSample] = None
    ) -> Dict[str, Any]:
        """
        Validate generated code against validation criteria.

        Args:
            generated_code: Generated code string
            validation: Validation string or tests
            sample: Original dataset sample

        Returns:
            Validation result
        """
        result = {
            'syntax_valid': False,
            'execution_result': None,
            'validation_passed': False,
            'error': None
        }

        # Syntax validation
        syntax_result = validate_syntax(generated_code)
        result['syntax_valid'] = syntax_result['valid']
        if not syntax_result['valid']:
            result['error'] = syntax_result['error']
            return result

        # Execution validation
        try:
            if validation:
                # Create test cases from validation string
                test_cases = self._parse_validation_tests(validation)

                if test_cases:
                    execution_result = simulate_code_execution(
                        generated_code, test_cases, timeout=5
                    )
                    result['execution_result'] = execution_result
                    result['validation_passed'] = (
                        execution_result['success'] and
                        all(test['passed'] for test in execution_result['test_results'])
                    )
                else:
                    # Simple execution test
                    execution_result = simulate_code_execution(generated_code, timeout=5)
                    result['execution_result'] = execution_result
                    result['validation_passed'] = execution_result['success']

        except Exception as e:
            result['error'] = str(e)

        return result

    def _parse_validation_tests(self, validation: str) -> List[Dict[str, Any]]:
        """Parse validation string into test cases."""
        test_cases = []

        try:
            # Look for assert statements
            lines = validation.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('assert'):
                    # Simple assert parsing - could be enhanced
                    test_cases.append({
                        'type': 'assert',
                        'code': line
                    })

        except Exception as e:
            print(f"Error parsing validation tests: {e}")

        return test_cases

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded datasets."""
        stats = {
            'total_samples': 0,
            'datasets': {},
            'binary_decision_distribution': {0: 0, 1: 0},
            'average_prompt_length': 0,
            'average_solution_length': 0
        }

        total_prompt_length = 0
        total_solution_length = 0
        sample_count = 0

        for cache_key, samples in self.cached_data.items():
            dataset_name = cache_key.split('_')[0]
            stats['datasets'][dataset_name] = len(samples)
            stats['total_samples'] += len(samples)

            for sample in samples:
                stats['binary_decision_distribution'][sample.binary_decision] += 1
                total_prompt_length += len(sample.prompt)
                if sample.solution:
                    total_solution_length += len(sample.solution)
                sample_count += 1

        if sample_count > 0:
            stats['average_prompt_length'] = total_prompt_length / sample_count
            stats['average_solution_length'] = total_solution_length / sample_count

        return stats


if __name__ == "__main__":
    # Test data handler
    from .config import get_config

    config = get_config("debug")
    data_handler = DataHandler(config)

    print("Testing dataset loaders...")

    # Test individual loaders
    for name, loader in data_handler.loaders.items():
        print(f"Testing {name} loader...")

    print("Data handler tests completed!")