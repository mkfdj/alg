"""
Dataset Registry for coding datasets
Registry containing metadata for all 18 coding datasets
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


class DatasetRegistry:
    """Registry for all coding datasets"""

    def __init__(self):
        self.datasets = self._initialize_datasets()

    def _initialize_datasets(self) -> Dict[str, DatasetInfo]:
        """Initialize all 18 coding datasets"""
        return {
            "handcrafted_code_gen": DatasetInfo(
                name="OpenAI HumanEval (Coding Challenges & Unit-tests)",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/handcrafted-dataset-for-code-generation-models",
                description="164 handcrafted Python coding problems with unit tests",
                size="500 KB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('humaneval.csv')\nprompts = df['prompt'].tolist()",
                kaggle_dataset="thedevastator/handcrafted-dataset-for-code-generation-models",
                local_filename="humaneval.csv",
                load_function="load_csv"
            ),

            "openai_human_eval": DatasetInfo(
                name="openai-human-eval",
                kaggle_link="https://www.kaggle.com/datasets/inoueu1/openai-human-eval",
                description="Original OpenAI HumanEval mirror for code evaluation",
                size="200 KB",
                format="JSONL",
                structure_example="import json\nwith open('humaneval.jsonl', 'r') as f:\n    data = [json.loads(line) for line in f]\nprompts = [item['prompt'] for item in data]",
                kaggle_dataset="inoueu1/openai-human-eval",
                local_filename="humaneval.jsonl",
                load_function="load_jsonl"
            ),

            "openai_humaneval_code_gen": DatasetInfo(
                name="OpenAI HumanEval Code Gen",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/openai-humaneval-code-gen",
                description="English task descriptions with code solutions and unittests",
                size="400 KB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('codegen.csv', index_col='id')\nprompts = df['prompt'].tolist()",
                kaggle_dataset="thedevastator/openai-humaneval-code-gen",
                local_filename="codegen.csv",
                load_function="load_csv"
            ),

            "mbpp_python_problems": DatasetInfo(
                name="MBPP Python Problems jsonl",
                kaggle_link="https://www.kaggle.com/datasets/mpwolke/mbppjsonl",
                description="1000 basic Python programming problems with tests",
                size="1 MB",
                format="JSONL",
                structure_example="import json\ndata = [json.loads(line) for line in open('mbpp.jsonl')]\nprompts = [item['text'] for item in data]",
                kaggle_dataset="mpwolke/mbppjsonl",
                local_filename="mbpp.jsonl",
                load_function="load_jsonl"
            ),

            "mostly_basic_python": DatasetInfo(
                name="Mostly Basic Python Problems Dataset",
                kaggle_link="https://www.kaggle.com/datasets/buntyshah/mostly-basic-python-problems-dataset",
                description="Entry-level Python problems with solutions",
                size="500 KB",
                format="JSON",
                structure_example="import json\ndata = json.load(open('mbpp.json'))['problems']\nprompts = [p['prompt'] for p in data]",
                kaggle_dataset="buntyshah/mostly-basic-python-problems-dataset",
                local_filename="mbpp.json",
                load_function="load_json"
            ),

            "codeparrot_1m": DatasetInfo(
                name="codeparrot_1M (GitHub Code Subset)",
                kaggle_link="https://www.kaggle.com/datasets/heyytanay/codeparrot-1m",
                description="1M tokenized Python code files from GitHub",
                size="100 MB",
                format="Lance/Parquet",
                structure_example="import lance\nds = lance.dataset('codeparrot.lance')\ndf = ds.to_table().to_pandas()\nprompts = [f\"Complete: {row['content'][:200]}\" for index, row in df.iterrows()]",
                kaggle_dataset="heyytanay/codeparrot-1m",
                local_filename="codeparrot.lance",
                load_function="load_lance"
            ),

            "python_code_instruction": DatasetInfo(
                name="Python Code Instruction Dataset",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/python-code-instruction-dataset",
                description="Instruction-based Python code generation tasks",
                size="50 MB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('instructions.csv')\nprompts = df['prompt'].tolist()",
                kaggle_dataset="thedevastator/python-code-instruction-dataset",
                local_filename="instructions.csv",
                load_function="load_csv"
            ),

            "python_code_instructions_18k": DatasetInfo(
                name="python_code_instructions_18k_alpaca",
                kaggle_link="https://www.kaggle.com/datasets/nikitakudriashov/python-code-instructions-18k-alpaca",
                description="18K Alpaca-style Python code instructions",
                size="100 MB",
                format="JSON",
                structure_example="import json\ndata = json.load(open('alpaca.json'))['data']\nprompts = [item['instruction'] for item in data]",
                kaggle_dataset="nikitakudriashov/python-code-instructions-18k-alpaca",
                local_filename="alpaca.json",
                load_function="load_json"
            ),

            "alpaca_language_training": DatasetInfo(
                name="Alpaca Language Instruction Training",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/alpaca-language-instruction-training",
                description="52K cleaned Alpaca instructions, filterable for code",
                size="200 MB",
                format="JSON",
                structure_example="import json\ndata = json.load(open('alpaca.json'))['items']\nprompts = [item['instruction'] for item in data]",
                kaggle_dataset="thedevastator/alpaca-language-instruction-training",
                local_filename="alpaca.json",
                load_function="load_json"
            ),

            "glaive_python_code_qa": DatasetInfo(
                name="Glaive Python Code QA Dataset",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/glaive-python-code-qa-dataset",
                description="140K Python code Q&A pairs",
                size="100 MB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('glaive.csv')\nprompts = df['question'].tolist()",
                kaggle_dataset="thedevastator/glaive-python-code-qa-dataset",
                local_filename="glaive.csv",
                load_function="load_csv"
            ),

            "code_contests": DatasetInfo(
                name="Code Contests Dataset",
                kaggle_link="https://www.kaggle.com/datasets/lallucycle/code-contests-dataset",
                description="Competition-style coding problems with samples",
                size="50 MB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('codecontests.csv')\nprompts = df['problem_statement'].tolist()",
                kaggle_dataset="lallucycle/code-contests-dataset",
                local_filename="codecontests.csv",
                load_function="load_csv"
            ),

            "livecodebench": DatasetInfo(
                name="LiveCodeBench",
                kaggle_link="https://www.kaggle.com/datasets/open-benchmarks/livecodebench",
                description="Live updating LeetCode-style benchmark",
                size="20 MB",
                format="JSON",
                structure_example="import json\ndata = json.load(open('livecodebench.json'))['problems']\nprompts = [p['problem'] for p in data]",
                kaggle_dataset="open-benchmarks/livecodebench",
                local_filename="livecodebench.json",
                load_function="load_json"
            ),

            "coding_questions_solutions": DatasetInfo(
                name="Coding Questions with Solutions",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/coding-questions-with-solutions",
                description="Multi-level coding questions for interviews/competitions",
                size="50 MB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('coding_questions.csv')\nprompts = df['question'].tolist()",
                kaggle_dataset="thedevastator/coding-questions-with-solutions",
                local_filename="coding_questions.csv",
                load_function="load_csv"
            ),

            "evol_instruct_code": DatasetInfo(
                name="Evol-Instruct-Code-80k-v1",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/evol-instruct-code-80k-v1-dataset",
                description="80K evolved code instructions for advanced gen",
                size="200 MB",
                format="JSON",
                structure_example="import json\ndata = json.load(open('evol_instruct.json'))['data']\nprompts = [item['instruction'] for item in data]",
                kaggle_dataset="thedevastator/evol-instruct-code-80k-v1-dataset",
                local_filename="evol_instruct.json",
                load_function="load_json"
            ),

            "python_programming_questions": DatasetInfo(
                name="Python Programming Questions Dataset",
                kaggle_link="https://www.kaggle.com/datasets/bhaveshmittal/python-programming-questions-dataset",
                description="150+ Python questions for ML training",
                size="100 MB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('python_questions.csv')\nprompts = df['question'].tolist()",
                kaggle_dataset="bhaveshmittal/python-programming-questions-dataset",
                local_filename="python_questions.csv",
                load_function="load_csv"
            ),

            "glaive_python_extended": DatasetInfo(
                name="Glaive Python Code QA Dataset (Extended)",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/glaive-python-code-qa-dataset",
                description="Extended 140k code QA with tool call support",
                size="100 MB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('glaive_extended.csv')\nprompts = df['question'].tolist()",
                kaggle_dataset="thedevastator/glaive-python-code-qa-dataset",
                local_filename="glaive_extended.csv",
                load_function="load_csv"
            ),

            "python_code_advanced": DatasetInfo(
                name="Python Code Instruction (Advanced)",
                kaggle_link="https://www.kaggle.com/datasets/thedevastator/python-code-instruction-dataset",
                description="Advanced instructions with web react/design patterns",
                size="50 MB",
                format="CSV",
                structure_example="import pandas as pd\ndf = pd.read_csv('advanced_instructions.csv')\nprompts = df['prompt'].tolist()",
                kaggle_dataset="thedevastator/python-code-instruction-dataset",
                local_filename="advanced_instructions.csv",
                load_function="load_csv"
            ),

            "python_source_150k": DatasetInfo(
                name="150k Python Source Code Dataset",
                kaggle_link="https://www.kaggle.com/datasets/veeralakrishna/150k-python-dataset",
                description="150K Python source code snippets for design/web patterns",
                size="500 MB",
                format="JSON",
                structure_example="import json\ndata = json.load(open('python_source.json'))\nprompts = [f\"Complete code: {item['snippet'][:200]}\" for item in data]",
                kaggle_dataset="veeralakrishna/150k-python-dataset",
                local_filename="python_source.json",
                load_function="load_json"
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

    def get_datasets_by_format(self, format_type: str) -> List[DatasetInfo]:
        """Get all datasets of a specific format"""
        return [info for info in self.datasets.values() if info.format.lower() == format_type.lower()]

    def get_total_size_info(self) -> Dict[str, Any]:
        """Get summary information about all datasets"""
        total_size = 0
        format_counts = {}

        for info in self.datasets.values():
            # Extract numeric size in MB (approximate)
            if "MB" in info.size:
                size_mb = float(info.size.replace(" MB", ""))
            elif "KB" in info.size:
                size_mb = float(info.size.replace(" KB", "")) / 1024
            elif "GB" in info.size:
                size_mb = float(info.size.replace(" GB", "")) * 1024
            else:
                continue

            total_size += size_mb
            format_counts[info.format] = format_counts.get(info.format, 0) + 1

        return {
            "total_datasets": len(self.datasets),
            "total_size_mb": round(total_size, 2),
            "total_size_gb": round(total_size / 1024, 2),
            "format_distribution": format_counts
        }