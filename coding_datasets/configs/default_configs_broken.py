"""
Default configuration templates
"""

def get_default_config() -> dict:
    """Get the complete default configuration"""
    return {
        "data": {
            "download_path": "./coding_datasets/data",
            "cache_enabled": True,
            "max_concurrent_downloads": 3,
            "auto_cleanup": False,
            "verify_downloads": True,
            "kaggle_api_path": None,
            "kaggle_username": None,
            "kaggle_key": None,
            "dataset_timeout": 300
        },
        "preprocessing": {
            "normalize_whitespace": True,
            "remove_duplicates": True,
            "filter_by_length": True,
            "min_length": 10,
            "max_length": 10000,
            "remove_code_blocks": False,
            "fix_encoding_issues": True,
            "normalize_punctuation": True,
            "remove_empty_lines": True,
            "strip_urls": True,
            "validate_code": True
        },
        "training": {
            "format": "instruction",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "random_seed": 42,
            "shuffle_data": True,
            "add_task_prefix": False,
            "include_metadata": True,
            "max_examples_per_dataset": None,
            "balance_datasets": False
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_logging": False,
            "log_file": "./coding_datasets/logs/datasets.log",
            "console_logging": True
        },
        "datasets": {
            "enabled": [
                "handcrafted_code_gen",
                "openai_human_eval",
                "mbpp_python_problems",
                "python_code_instruction",
                "glaive_python_code_qa",
                "evol_instruct_code"
            ],
            "priority": {
                "handcrafted_code_gen": 1,
                "openai_human_eval": 2,
                "mbpp_python_problems": 3,
                "python_code_instruction": 4,
                "glaive_python_code_qa": 5,
                "evol_instruct_code": 6
            },
            "custom_prompts": {
                "handcrafted_code_gen": "prompt",
                "openai_human_eval": "prompt",
                "mbpp_python_problems": "text",
                "python_code_instruction": "prompt",
                "glaive_python_code_qa": "question",
                "evol_instruct_code": "instruction"
            },
            "data_paths": {
                "handcrafted_code_gen": "humaneval.csv",
                "openai_human_eval": "humaneval.jsonl",
                "mbpp_python_problems": "mbpp.jsonl",
                "python_code_instruction": "instructions.csv",
                "glaive_python_code_qa": "glaive.csv",
                "evol_instruct_code": "evol_instruct.json"
            }
        },
        "validation": {
            "strict_mode": False,
            "check_duplicates": True,
            "check_encoding": True,
            "check_code_quality": True,
            "min_quality_score": 70,
            "validate_responses": True,
            "check_language_consistency": True
        },
        "export": {
            "format": "jsonl",
            "include_statistics": True,
            "compression": False,
            "split_files": True,
            "file_prefix": "coding_dataset"
        }
    }


def get_preprocessing_config() -> dict:
    """Get preprocessing-specific configuration"""
    return {
        "normalize_whitespace": {
            "enabled": True,
            "replace_tabs": True,
            "remove_multiple_spaces": True
        },
        "remove_duplicates": {
            "enabled": True,
            "method": "exact",  # "exact" or "fuzzy"
            "fuzzy_threshold": 0.9
        },
        "filter_by_length": {
            "enabled": True,
            "min_length": 10,
            "max_length": 10000,
            "measure": "characters"  # "characters" or "words"
        },
        "code_cleaning": {
            "enabled": True,
            "remove_markdown_fences": True,
            "fix_indentation": True,
            "normalize_quotes": True,
            "remove_comments": False
        },
        "content_filtering": {
            "enabled": True,
            "remove_empty": True,
            "remove_non_english": False,
            "strip_urls": True,
            "remove_special_patterns": True
        },
        "encoding_fixes": {
            "enabled": True,
            "common_issues": [
                "â€™", "â€œ", "â€", "â€\"", "â€¦", "â€""
            ],
            "normalize_unicode": True
        }
    }


def get_training_config() -> dict:
    """Get training-specific configuration"""
    return {
        "data_splitting": {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "stratify": False,
            "shuffle": True,
            "random_seed": 42
        },
        "formatting": {
            "style": "instruction",  # "instruction", "alpaca", "chat", "code_completion"
            "add_task_prefix": True,
            "include_examples": False,
            "max_examples": 3
        },
        "task_instructions": {
            "coding": "Write code to solve the following problem:",
            "debugging": "Debug and fix the following code:",
            "explanation": "Explain what the following code does:",
            "completion": "Complete the following code:",
            "optimization": "Optimize the following code:"
        },
        "quality_control": {
            "min_examples_per_dataset": 10,
            "max_examples_per_dataset": None,
            "balance_datasets": False,
            "remove_outliers": True,
            "outlier_threshold": 3.0
        },
        "metadata": {
            "include": True,
            "fields": [
                "dataset_id",
                "original_index",
                "length",
                "language",
                "complexity"
            ]
        }
    }


def get_dataset_specific_config(dataset_id: str) -> dict:
    """Get configuration specific to a dataset"""
    configs = {
        "handcrafted_code_gen": {
            "prompt_field": "prompt",
            "response_field": None,
            "preprocessing": {
                "remove_code_blocks": False,
                "min_length": 50
            },
            "validation": {
                "strict_mode": True,
                "check_code_quality": True
            }
        },
        "openai_human_eval": {
            "prompt_field": "prompt",
            "response_field": "canonical_solution",
            "preprocessing": {
                "remove_code_blocks": False,
                "preserve_formatting": True
            },
            "validation": {
                "strict_mode": True,
                "validate_code": True
            }
        },
        "mbpp_python_problems": {
            "prompt_field": "text",
            "response_field": "code",
            "preprocessing": {
                "remove_code_blocks": False,
                "min_length": 30
            },
            "validation": {
                "check_code_quality": True,
                "validate_code": True
            }
        },
        "python_code_instruction": {
            "prompt_field": "prompt",
            "response_field": None,
            "preprocessing": {
                "remove_code_blocks": False,
                "normalize_instructions": True
            },
            "validation": {
                "strict_mode": False,
                "check_code_quality": False
            }
        },
        "glaive_python_code_qa": {
            "prompt_field": "question",
            "response_field": "answer",
            "preprocessing": {
                "remove_code_blocks": False,
                "format_as_qa": True
            },
            "validation": {
                "strict_mode": False,
                "validate_responses": True
            }
        },
        "evol_instruct_code": {
            "prompt_field": "instruction",
            "response_field": None,
            "preprocessing": {
                "remove_code_blocks": False,
                "normalize_instructions": True,
                "min_length": 50
            },
            "validation": {
                "strict_mode": False,
                "check_code_quality": True
            }
        }
    }

    return configs.get(dataset_id, {})


def get_quality_config() -> dict:
    """Get data quality validation configuration"""
    return {
        "basic_checks": {
            "check_empty": True,
            "check_encoding": True,
            "check_length_bounds": True,
            "min_length": 10,
            "max_length": 10000
        },
        "content_quality": {
            "check_duplicates": True,
            "check_low_content": True,
            "check_suspicious_patterns": True,
            "suspicious_patterns": [
                "http[s]?://\\S+",
                "<script[^>]*>.*?</script>",
                "eval\\s*\\(",
                "exec\\s*\\("
            ]
        },
        "code_quality": {
            "validate_syntax": True,
            "check_security_issues": True,
            "validate_completeness": True,
            "min_code_complexity": "simple"
        },
        "language_consistency": {
            "check_primary_language": True,
            "allow_multiple_languages": True,
            "max_languages": 3,
            "min_code_ratio": 0.3
        },
        "response_validation": {
            "check_empty_responses": True,
            "check_response_quality": True,
            "check_response_length": True,
            "min_response_length": 5
        }
    }


def get_export_config() -> dict:
    """Get data export configuration"""
    return {
        "formats": {
            "primary": "jsonl",
            "secondary": ["json", "csv"],
            "compression": False
        },
        "file_organization": {
            "split_by_dataset": False,
            "split_by_split": True,  # train/val/test
            "include_timestamp": True,
            "file_prefix": "coding_dataset"
        },
        "content": {
            "include_metadata": True,
            "include_statistics": True,
            "include_validation_report": True,
            "pretty_print": False
        },
        "quality_filters": {
            "include_quality_scores": True,
            "min_quality_score": 70,
            "exclude_invalid": True
        }
    }