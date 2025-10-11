"""
Data validation utilities for coding datasets
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter


class DataValidator:
    """Utilities for validating coding dataset quality and integrity"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_dataset_quality(self, prompts: List[str], responses: List[str] = None,
                               strict_mode: bool = False) -> Dict[str, Any]:
        """
        Comprehensive dataset quality validation

        Args:
            prompts: List of prompt strings
            responses: List of response strings (optional)
            strict_mode: Enable strict validation rules

        Returns:
            Validation results with detailed metrics
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {},
            'quality_score': 0
        }

        # Basic checks
        basic_checks = self._validate_basic_properties(prompts)
        results['errors'].extend(basic_checks['errors'])
        results['warnings'].extend(basic_checks['warnings'])
        results['metrics'].update(basic_checks['metrics'])

        # Content quality checks
        content_checks = self._validate_content_quality(prompts, strict_mode)
        results['errors'].extend(content_checks['errors'])
        results['warnings'].extend(content_checks['warnings'])
        results['metrics'].update(content_checks['metrics'])

        # Duplicate detection
        duplicate_checks = self._check_duplicates(prompts)
        results['warnings'].extend(duplicate_checks['warnings'])
        results['metrics'].update(duplicate_checks['metrics'])

        # Language and format validation
        language_checks = self._validate_language_and_format(prompts)
        results['errors'].extend(language_checks['errors'])
        results['warnings'].extend(language_checks['warnings'])
        results['metrics'].update(language_checks['metrics'])

        # Response validation if provided
        if responses:
            response_checks = self._validate_responses(prompts, responses, strict_mode)
            results['errors'].extend(response_checks['errors'])
            results['warnings'].extend(response_checks['warnings'])
            results['metrics'].update(response_checks['metrics'])

        # Calculate overall quality score
        results['quality_score'] = self._calculate_quality_score(results)
        results['is_valid'] = len(results['errors']) == 0

        return results

    def _validate_basic_properties(self, prompts: List[str]) -> Dict[str, Any]:
        """Validate basic dataset properties"""
        errors = []
        warnings = []
        metrics = {}

        # Check if dataset is empty
        if not prompts:
            errors.append("Dataset is empty")
            return {'errors': errors, 'warnings': warnings, 'metrics': metrics}

        # Length statistics
        lengths = [len(p) for p in prompts]
        metrics['length_stats'] = {
            'min': min(lengths),
            'max': max(lengths),
            'mean': round(sum(lengths) / len(lengths), 2),
            'median': sorted(lengths)[len(lengths) // 2]
        }

        # Check for extremely short or long prompts
        very_short = sum(1 for l in lengths if l < 10)
        very_long = sum(1 for l in lengths if l > 10000)

        if very_short > 0:
            warnings.append(f"Found {very_short} very short prompts (< 10 chars)")
        if very_long > 0:
            warnings.append(f"Found {very_long} very long prompts (> 10000 chars)")

        # Check for empty prompts
        empty_count = sum(1 for p in prompts if not p.strip())
        if empty_count > 0:
            errors.append(f"Found {empty_count} empty prompts")

        metrics['empty_count'] = empty_count
        metrics['very_short_count'] = very_short
        metrics['very_long_count'] = very_long

        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}

    def _validate_content_quality(self, prompts: List[str], strict_mode: bool) -> Dict[str, Any]:
        """Validate content quality of prompts"""
        errors = []
        warnings = []
        metrics = {}

        encoding_issues = 0
        suspicious_patterns = 0
        low_content_count = 0

        for i, prompt in enumerate(prompts):
            # Check for encoding issues
            if self._has_encoding_issues(prompt):
                encoding_issues += 1

            # Check for suspicious patterns
            if self._has_suspicious_patterns(prompt):
                suspicious_patterns += 1

            # Check for low content prompts
            if self._is_low_content(prompt):
                low_content_count += 1

        if encoding_issues > 0:
            warnings.append(f"Found {encoding_issues} prompts with encoding issues")
        if suspicious_patterns > 0:
            warnings.append(f"Found {suspicious_patterns} prompts with suspicious patterns")
        if low_content_count > 0:
            warnings.append(f"Found {low_content_count} prompts with low content")

        if strict_mode and suspicious_patterns > len(prompts) * 0.1:
            errors.append("Too many prompts with suspicious patterns in strict mode")

        metrics['encoding_issues'] = encoding_issues
        metrics['suspicious_patterns'] = suspicious_patterns
        metrics['low_content_count'] = low_content_count

        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}

    def _check_duplicates(self, prompts: List[str]) -> Dict[str, Any]:
        """Check for duplicate prompts"""
        warnings = []
        metrics = {}

        # Exact duplicates
        unique_prompts = set(prompts)
        exact_duplicates = len(prompts) - len(unique_prompts)

        # Near duplicates (using simple similarity)
        near_duplicates = self._find_near_duplicates(prompts, threshold=0.9)

        if exact_duplicates > 0:
            warnings.append(f"Found {exact_duplicates} exact duplicate prompts")
        if near_duplicates > 0:
            warnings.append(f"Found {near_duplicates} near duplicate prompts")

        metrics['exact_duplicates'] = exact_duplicates
        metrics['near_duplicates'] = near_duplicates
        metrics['unique_count'] = len(unique_prompts)

        return {'errors': [], 'warnings': warnings, 'metrics': metrics}

    def _validate_language_and_format(self, prompts: List[str]) -> Dict[str, Any]:
        """Validate programming language and format consistency"""
        errors = []
        warnings = []
        metrics = {}

        language_distribution = Counter()
        format_distribution = Counter()
        code_snippet_count = 0

        for prompt in prompts:
            # Detect language
            language = self._detect_primary_language(prompt)
            language_distribution[language] += 1

            # Detect format
            format_type = self._detect_format_type(prompt)
            format_distribution[format_type] += 1

            # Count code snippets
            if self._contains_code_snippet(prompt):
                code_snippet_count += 1

        metrics['language_distribution'] = dict(language_distribution)
        metrics['format_distribution'] = dict(format_distribution)
        metrics['code_snippet_count'] = code_snippet_count
        metrics['code_ratio'] = round(code_snippet_count / len(prompts), 3)

        # Check for inconsistent language distribution
        if len(language_distribution) > 3:
            warnings.append("Dataset contains many different programming languages")

        # Check for low code ratio
        code_ratio = code_snippet_count / len(prompts)
        if code_ratio < 0.3:
            warnings.append(f"Low code ratio: {code_ratio:.1%} prompts contain code")

        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}

    def _validate_responses(self, prompts: List[str], responses: List[str],
                           strict_mode: bool) -> Dict[str, Any]:
        """Validate response data"""
        errors = []
        warnings = []
        metrics = {}

        # Check length mismatch
        if len(prompts) != len(responses):
            errors.append(f"Length mismatch: {len(prompts)} prompts vs {len(responses)} responses")
            return {'errors': errors, 'warnings': warnings, 'metrics': metrics}

        # Check for empty responses
        empty_responses = sum(1 for r in responses if not r.strip())
        if empty_responses > 0:
            warnings.append(f"Found {empty_responses} empty responses")

        # Check response quality
        low_quality_responses = 0
        for response in responses:
            if self._is_low_quality_response(response):
                low_quality_responses += 1

        if low_quality_responses > 0:
            warnings.append(f"Found {low_quality_responses} low quality responses")

        metrics['empty_responses'] = empty_responses
        metrics['low_quality_responses'] = low_quality_responses
        metrics['response_length_stats'] = self._calculate_length_stats(responses)

        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}

    def _has_encoding_issues(self, text: str) -> bool:
        """Check for common encoding issues"""
        problematic_chars = ['â€™', 'â€œ', 'â€', 'â€"', 'â€¦', 'â€"']
        return any(char in text for char in problematic_chars)

    def _has_suspicious_patterns(self, text: str) -> bool:
        """Check for suspicious or problematic patterns"""
        suspicious_patterns = [
            r'http[s]?://\S+',  # URLs
            r'<script[^>]*>.*?</script>',  # Script tags
            r'eval\s*\(',  # eval statements
            r'exec\s*\(',  # exec statements
            r'__import__\s*\(',  # dynamic imports
            r'subprocess\.',  # subprocess calls
            r'os\.system',  # system calls
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return True
        return False

    def _is_low_content(self, text: str) -> bool:
        """Check if text has low content value"""
        text = text.strip()
        if len(text) < 20:
            return True

        # Check if text is mostly repetitive
        words = text.split()
        if len(words) < 3:
            return True

        # Check word diversity
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.3:
            return True

        return False

    def _find_near_duplicates(self, prompts: List[str], threshold: float = 0.9) -> int:
        """Find near duplicates using simple similarity"""
        near_duplicates = 0
        seen = []

        for prompt in prompts:
            prompt_lower = prompt.lower()
            for seen_prompt in seen:
                similarity = self._calculate_similarity(prompt_lower, seen_prompt)
                if similarity > threshold:
                    near_duplicates += 1
                    break
            seen.append(prompt_lower)

        return near_duplicates

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 1.0

        return len(intersection) / len(union)

    def _detect_primary_language(self, text: str) -> str:
        """Detect primary programming language in text"""
        language_patterns = {
            'python': [r'\bdef\s+\w+\s*\(', r'\bclass\s+\w+\s*:', r'\bimport\s+\w+', r'\bfrom\s+\w+\s+import'],
            'javascript': [r'\bfunction\s+\w+\s*\(', r'\bconst\s+\w+\s*=', r'\blet\s+\w+\s*=', r'\bvar\s+\w+\s*='],
            'java': [r'\bpublic\s+class\s+\w+', r'\bprivate\s+\w+\s+\w+\s*\(', r'\bpublic\s+static\s+void\s+main'],
            'cpp': [r'#include\s*<', r'\bstd::', r'\bint\s+main\s*\(', r'\bcout\s*<<'],
            'html': [r'<html', r'<div', r'<p>', r'<script'],
            'css': [r'\.?\w+\s*\{', r'\bcolor\s*:', r'\bbackground\s*:'],
        }

        language_scores = {}
        for language, patterns in language_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
            language_scores[language] = score

        if not language_scores or max(language_scores.values()) == 0:
            return 'unknown'

        return max(language_scores, key=language_scores.get)

    def _detect_format_type(self, text: str) -> str:
        """Detect the format type of the prompt"""
        if re.search(r'```[a-zA-Z]*\n.*?\n```', text, re.DOTALL):
            return 'markdown_code_blocks'
        elif re.search(r'def\s+\w+\s*\(|class\s+\w+\s*:', text):
            return 'code_snippet'
        elif text.strip().endswith('?'):
            return 'question'
        elif any(word in text.lower() for word in ['write', 'create', 'implement', 'develop']):
            return 'instruction'
        else:
            return 'other'

    def _contains_code_snippet(self, text: str) -> bool:
        """Check if text contains code snippets"""
        code_indicators = [
            r'```[a-zA-Z]*\n',  # Markdown code blocks
            r'def\s+\w+\s*\(',  # Python function
            r'function\s+\w+\s*\(',  # JavaScript function
            r'class\s+\w+\s*:',  # Class definition
            r'#include\s*<',  # C++ include
            r'<html',  # HTML
        ]

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in code_indicators)

    def _is_low_quality_response(self, response: str) -> bool:
        """Check if response is low quality"""
        response = response.strip()

        if len(response) < 10:
            return True

        # Check for generic responses
        generic_responses = [
            "i don't know",
            "i cannot help",
            "i'm not sure",
            "this is not possible",
            "sorry, i can't",
            "error",
            "invalid",
        ]

        response_lower = response.lower()
        return any(generic in response_lower for generic in generic_responses)

    def _calculate_length_stats(self, texts: List[str]) -> Dict[str, float]:
        """Calculate length statistics for a list of texts"""
        if not texts:
            return {}

        lengths = [len(text) for text in texts]
        return {
            'min': min(lengths),
            'max': max(lengths),
            'mean': round(sum(lengths) / len(lengths), 2),
            'median': sorted(lengths)[len(lengths) // 2]
        }

    def _calculate_quality_score(self, results: Dict[str, Any]) -> int:
        """Calculate overall quality score (0-100)"""
        score = 100

        # Deduct points for errors
        score -= len(results['errors']) * 20

        # Deduct points for warnings
        score -= len(results['warnings']) * 5

        # Deduct points for quality issues
        metrics = results['metrics']

        if 'duplicate_count' in metrics:
            duplicate_ratio = metrics['duplicate_count'] / max(1, metrics.get('total_examples', 1))
            score -= int(duplicate_ratio * 20)

        if 'empty_count' in metrics and metrics['empty_count'] > 0:
            empty_ratio = metrics['empty_count'] / max(1, metrics.get('total_examples', 1))
            score -= int(empty_ratio * 30)

        return max(0, score)

    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a human-readable validation report"""
        report = []

        report.append("=== DATASET VALIDATION REPORT ===\n")

        # Overall status
        status = "VALID" if validation_results['is_valid'] else "INVALID"
        score = validation_results['quality_score']
        report.append(f"Overall Status: {status}")
        report.append(f"Quality Score: {score}/100")

        # Errors
        if validation_results['errors']:
            report.append(f"\nERRORS ({len(validation_results['errors'])}):")
            for error in validation_results['errors']:
                report.append(f"  ❌ {error}")

        # Warnings
        if validation_results['warnings']:
            report.append(f"\nWARNINGS ({len(validation_results['warnings'])}):")
            for warning in validation_results['warnings']:
                report.append(f"  ⚠️  {warning}")

        # Metrics
        metrics = validation_results['metrics']
        if metrics:
            report.append("\nMETRICS:")
            for key, value in metrics.items():
                if isinstance(value, dict):
                    report.append(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        report.append(f"    {sub_key}: {sub_value}")
                else:
                    report.append(f"  {key}: {value}")

        return "\n".join(report)