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


class DatasetRegistry:
    """Registry for actual coding datasets with correct information"""

    def __init__(self):
        self.datasets = self._initialize_datasets()

    def _initialize_datasets(self) -> Dict[str, DatasetInfo]:
        """Initialize all datasets with actual correct information"""
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
            ),

            "coding_questions_solutions": DatasetInfo(
                name="Coding Questions with Solutions",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/coding-questions-with-solutions",
                description="Multi-level coding questions for interviews/competitions",
                size="1.33 GB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('train.csv')\nprompts = df['question'].tolist()",
                kaggle_dataset="thedevastator/coding-questions-with-solutions",
                local_filename="train.csv",
                load_function="load_csv",
                columns=["question", "solutions", "input_output", "difficulty", "url", "starter_code"],
                prompt_field="question"
            ),

            "evol_instruct_code": DatasetInfo(
                name="Evol-Instruct-Code-80k-v1",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/evol-instruct-code-80k-v1-dataset",
                description="80K evolved code instructions for advanced gen",
                size="116.59 MB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('train.csv')\nprompts = df['instruction'].tolist()",
                kaggle_dataset="thedevastator/evol-instruct-code-80k-v1-dataset",
                local_filename="train.csv",
                load_function="load_csv",
                columns=["output", "instruction"],
                prompt_field="instruction"
            ),

            "python_programming_questions": DatasetInfo(
                name="Python Programming Questions Dataset",
                kaggle_link="https://www.kaggle.com/datasets/bhaveshmittal/python-programming-questions-dataset",
                description="150+ Python questions for ML training",
                size="3.9 MB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('Python Programming Questions Dataset.csv')\nprompts = df['Instruction'].tolist()",
                kaggle_dataset="bhaveshmittal/python-programming-questions-dataset",
                local_filename="Python Programming Questions Dataset.csv",
                load_function="load_csv",
                columns=["Instruction", "Input", "Output"],
                prompt_field="Instruction"
            ),

            "python_code_advanced": DatasetInfo(
                name="Python Code Instruction (Advanced)",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/python-code-instruction-dataset",
                description="Advanced instructions with web react/design patterns",
                size="25.22 MB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('train.csv')\nprompts = df['prompt'].tolist()",
                kaggle_dataset="thedevastator/python-code-instruction-dataset",
                local_filename="train.csv",
                load_function="load_csv",
                columns=["instruction", "input", "output", "prompt"],
                prompt_field="prompt"
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

    def get_datasets_by_size_limit(self, max_mb: float = 100) -> List[str]:
        """Get datasets that are smaller than max_mb"""
        datasets = []
        for dataset_id, info in self.datasets.items():
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

                if size_mb <= max_mb:
                    datasets.append(dataset_id)
            except:
                continue

        return datasets

    def get_medium_datasets(self, max_mb: float = 1000) -> List[str]:
        """Get datasets that are medium size (100MB - max_mb)"""
        medium_datasets = []
        for dataset_id, info in self.datasets.items():
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

                if 100 < size_mb <= max_mb:
                    medium_datasets.append(dataset_id)
            except:
                continue

        return medium_datasets

    def get_large_datasets(self, min_mb: float = 1000) -> List[str]:
        """Get datasets that are larger than min_mb"""
        large_datasets = []
        for dataset_id, info in self.datasets.items():
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

                if size_mb > min_mb:
                    large_datasets.append(dataset_id)
            except:
                continue

        return large_datasets

    def get_datasets_for_size_limit(self, max_mb: float = 500) -> Dict[str, List[str]]:
        """Get all datasets categorized by size"""
        return {
            "small": self.get_datasets_by_size_limit(100),
            "medium": self.get_medium_datasets(max_mb),
            "large": self.get_large_datasets(max_mb)
        }