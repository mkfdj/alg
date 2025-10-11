"""
Actual Dataset Registry for coding datasets with correct sizes and structure
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """Information about a dataset"""
    name: str
    kaggle_link: str
    description: str
    size: str
    format: str
    structure_example: str
    kaggle_dataset: str
    local_filename: str
    load_function: str
    columns: List[str]
    prompt_field: str


class ActualDatasetRegistry:
    """Registry for actual coding datasets with correct information"""

    def __init__(self):
        self.datasets = self._initialize_datasets()

    def _initialize_datasets(self) -> Dict[str, DatasetInfo]:
        """Initialize all datasets with actual information"""
        return {
            "openai_humaneval_code_gen": DatasetInfo(
                name="OpenAI HumanEval Code Gen",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/openai-humaneval-code-gen",
                description="English task descriptions with code solutions and unittests",
                size="196.17 kB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('test.csv')\nprompts = df['prompt'].tolist()",
                kaggle_dataset="thedevastator/openai-humaneval-code-gen",
                local_filename="test.csv",
                load_function="load_csv",
                columns=["prompt", "canonical_solution", "test", "entry_point"],
                prompt_field="prompt"
            ),

            "mbpp_python_problems": DatasetInfo(
                name="MBPP Python Problems jsonl",
                kaggle_link="https://www.kaggle.com/datasets/mpwolke/mbppjsonl",
                description="1000 basic Python programming problems with tests",
                size="563.74 kB",
                format="JSONL",
                structure_example="import json\ndata = [json.loads(line) for line in open('mbpp.jsonl')]\nprompts = [item['text'] for item in data]",
                kaggle_dataset="mpwolke/mbppjsonl",
                local_filename="mbpp.jsonl",
                load_function="load_jsonl",
                columns=["text", "code", "task_id", "test_setup_code", "test_list", "challenge_test_list"],
                prompt_field="text"
            ),

            "python_code_instruction": DatasetInfo(
                name="Python Code Instruction Dataset",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/python-code-instruction-dataset",
                description="Instruction-based Python code generation tasks",
                size="25.22 MB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('train.csv')\nprompts = df['instruction'].tolist()",
                kaggle_dataset="thedevastator/python-code-instruction-dataset",
                local_filename="train.csv",
                load_function="load_csv",
                columns=["instruction", "input", "output", "prompt"],
                prompt_field="instruction"
            ),

            "python_code_instructions_18k": DatasetInfo(
                name="python_code_instructions_18k_alpaca",
                kaggle_link="https://www.kaggle.com/datasets/nikitakudriashov/python-code-instructions-18k-alpaca",
                description="18K Alpaca-style Python code instructions",
                size="11.36 MB",
                format="Parquet",
                structure_example="import pandas as pd\ndf = pd.read_parquet('train-00000-of-00001-8b6e212f3e1ece96.parquet')\nprompts = df['instruction'].tolist()",
                kaggle_dataset="nikitakudriashov/python-code-instructions-18k-alpaca",
                local_filename="train-00000-of-00001-8b6e212f3e1ece96.parquet",
                load_function="load_parquet",
                columns=["instruction", "input", "output", "prompt"],
                prompt_field="instruction"
            ),

            "alpaca_cleaned": DatasetInfo(
                name="Alpaca Cleaned",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/alpaca-language-instruction-training",
                description="52K cleaned Alpaca instructions, filterable for code",
                size="39.98 MB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('train.csv')\nprompts = df['instruction'].tolist()",
                kaggle_dataset="thedevastator/alpaca-language-instruction-training",
                local_filename="train.csv",
                load_function="load_csv",
                columns=["instruction", "output", "input"],
                prompt_field="instruction"
            ),

            "glaive_python_code_qa": DatasetInfo(
                name="Glaive Python Code QA Dataset",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/glaive-python-code-qa-dataset",
                description="140K Python code Q&A pairs",
                size="207.63 MB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('train.csv')\nprompts = df['question'].tolist()",
                kaggle_dataset="thedevastator/glaive-python-code-qa-dataset",
                local_filename="train.csv",
                load_function="load_csv",
                columns=["answer", "question"],
                prompt_field="question"
            ),

            "livecodebench": DatasetInfo(
                name="LiveCodeBench",
                kaggle_link="https://www.kaggle.com/datasets/open-benchmarks/livecodebench",
                description="Live updating LeetCode-style benchmark",
                size="4 GB",
                format="JSONL",
                structure_example="import json\ndata = [json.loads(line) for line in open('test.jsonl')]\nprompts = [item['question_content'] for item in data]",
                kaggle_dataset="open-benchmarks/livecodebench",
                local_filename="test.jsonl",
                load_function="load_jsonl",
                columns=["question_title", "question_content", "platform", "question_id", "contest_id", "contest_date", "starter_code", "difficulty", "public_test_cases", "private_test_cases", "metadata"],
                prompt_field="question_content"
            )
        }

    def get_dataset_info(self, dataset_id: str) -> DatasetInfo:
        """Get dataset information by ID"""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset '{dataset_id}' not found in registry")
        return self.datasets[dataset_id]

    def list_datasets(self) -> List[str]:
        """List all available dataset IDs"""
        return list(self.datasets.keys())

    def get_datasets_by_size(self, max_size_mb: float = None) -> List[DatasetInfo]:
        """Get datasets filtered by size"""
        datasets = []
        for info in self.datasets.values():
            size_str = info.size
            try:
                if "kB" in size_str:
                    size_mb = float(size_str.replace(" kB", "")) / 1024
                elif "MB" in size_str:
                    size_mb = float(size_str.replace(" MB", ""))
                elif "GB" in size_str:
                    size_mb = float(size_str.replace(" GB", "")) * 1024
                else:
                    continue

                if max_size_mb is None or size_mb <= max_size_mb:
                    datasets.append(info)
            except:
                continue

        return datasets

    def get_total_size_info(self) -> Dict[str, Any]:
        """Get summary information about all datasets"""
        total_size_mb = 0
        format_counts = {}
        dataset_count = 0

        for info in self.datasets.values():
            dataset_count += 1
            try:
                size_str = info.size
                if "kB" in size_str:
                    size_mb = float(size_str.replace(" kB", "")) / 1024
                elif "MB" in size_str:
                    size_mb = float(size_str.replace(" MB", ""))
                elif "GB" in size_str:
                    size_mb = float(size_str.replace(" GB", "")) * 1024
                else:
                    continue

                total_size_mb += size_mb
                format_counts[info.format] = format_counts.get(info.format, 0) + 1
            except:
                continue

        return {
            "total_datasets": dataset_count,
            "total_size_mb": round(total_size_mb, 2),
            "total_size_gb": round(total_size_mb / 1024, 2),
            "format_distribution": format_counts
        }