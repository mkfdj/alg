"""
Code cleaning utilities for programming datasets
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple


class CodeCleaner:
    """Utilities for cleaning and formatting code snippets"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def clean_code_snippet(self, code: str) -> str:
        """
        Clean a code snippet by removing common issues

        Args:
            code: Raw code string

        Returns:
            Cleaned code string
        """
        if not code:
            return code

        cleaned = code

        # Remove common artifacts
        cleaned = self._remove_markdown_fences(cleaned)
        cleaned = self._remove_explanation_text(cleaned)
        cleaned = self._fix_indentation(cleaned)
        cleaned = self._normalize_whitespace(cleaned)
        cleaned = self._fix_common_syntax_errors(cleaned)

        return cleaned.strip()

    def _remove_markdown_fences(self, code: str) -> str:
        """Remove markdown code fences"""
        # Remove ```python, ```javascript, etc.
        code = re.sub(r'```[a-zA-Z]*\n', '', code)
        # Remove closing ```
        code = re.sub(r'\n```', '', code)
        return code

    def _remove_explanation_text(self, code: str) -> str:
        """Remove common explanation text patterns"""
        # Remove lines that look like explanations
        lines = code.split('\n')
        code_lines = []

        for line in lines:
            line_stripped = line.strip()
            # Skip explanation-like lines
            if (line_stripped.startswith('#') and
                any(word in line_stripped.lower() for word in ['explanation', 'note', 'here', 'this code'])):
                continue
            if line_stripped.startswith('//') and any(word in line_stripped.lower() for word in ['explanation', 'note']):
                continue
            if any(phrase in line_stripped.lower() for phrase in
                   ['this function', 'this code', 'here we', 'the following', 'as you can see']):
                continue

            code_lines.append(line)

        return '\n'.join(code_lines)

    def _fix_indentation(self, code: str) -> str:
        """Fix indentation issues in code"""
        lines = code.split('\n')
        if not lines:
            return code

        # Find minimum indentation (excluding empty lines)
        indentations = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indentations.append(indent)

        if not indentations:
            return code

        min_indent = min(indentations)

        # Remove minimum indentation from all lines
        fixed_lines = []
        for line in lines:
            if line.strip():
                if len(line) >= min_indent:
                    fixed_lines.append(line[min_indent:])
                else:
                    fixed_lines.append(line.lstrip())
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _normalize_whitespace(self, code: str) -> str:
        """Normalize whitespace in code"""
        # Replace tabs with spaces (4 spaces per tab)
        code = code.replace('\t', '    ')
        # Remove trailing spaces
        lines = code.split('\n')
        lines = [line.rstrip() for line in lines]
        return '\n'.join(lines)

    def _fix_common_syntax_errors(self, code: str) -> str:
        """Fix common syntax errors"""
        # Fix multiple consecutive spaces in indentation
        code = re.sub(r'\n(\s{8,})', lambda m: '\n' + ('    ' * (len(m.group(1)) // 4)), code)
        # Fix missing colons after function/class definitions
        code = re.sub(r'(def |class |if |for |while |try |with |elif |else:\s*)(\n[^\s:])',
                     r'\1:\2', code)
        return code

    def extract_code_from_text(self, text: str) -> List[str]:
        """
        Extract code snippets from mixed text

        Args:
            text: Text containing code snippets

        Returns:
            List of extracted code snippets
        """
        code_snippets = []

        # Pattern for markdown code blocks
        markdown_pattern = r'```[a-zA-Z]*\n(.*?)\n```'
        matches = re.findall(markdown_pattern, text, re.DOTALL)
        code_snippets.extend(matches)

        # Pattern for inline code
        inline_pattern = r'`([^`\n]+)`'
        inline_matches = re.findall(inline_pattern, text)
        # Only keep inline code that looks like actual code
        for inline in inline_matches:
            if any(char in inline for char in ['(', ')', '[', ']', '{', '}', '=', '+', '-', '*', '/']):
                code_snippets.append(inline)

        # Pattern for Python-like code (lines with common keywords)
        python_pattern = r'(?:^|\n)(def |class |import |from |if |for |while |try |with |return |print\()'
        python_matches = re.findall(python_pattern, text, re.MULTILINE)
        if python_matches:
            # Extract full code blocks
            lines = text.split('\n')
            current_block = []
            in_code_block = False

            for line in lines:
                if re.match(python_pattern, line):
                    in_code_block = True
                    current_block.append(line)
                elif in_code_block:
                    if line.strip() == '':
                        current_block.append(line)
                    elif line.startswith('    ') or line.startswith('\t'):
                        current_block.append(line)
                    else:
                        # End of code block
                        if current_block:
                            code_snippets.append('\n'.join(current_block))
                        current_block = []
                        in_code_block = False

            if current_block:
                code_snippets.append('\n'.join(current_block))

        return [self.clean_code_snippet(snippet) for snippet in code_snippets if snippet.strip()]

    def validate_python_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax

        Args:
            code: Python code string

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def format_code_for_training(self, code: str, language: str = 'python',
                                style: str = 'clean') -> str:
        """
        Format code for training purposes

        Args:
            code: Code string
            language: Programming language
            style: Formatting style ('clean', 'commented', 'annotated')

        Returns:
            Formatted code string
        """
        cleaned_code = self.clean_code_snippet(code)

        if style == 'clean':
            return cleaned_code
        elif style == 'commented':
            return self._add_training_comments(cleaned_code, language)
        elif style == 'annotated':
            return self._add_annotations(cleaned_code, language)
        else:
            return cleaned_code

    def _add_training_comments(self, code: str, language: str) -> str:
        """Add helpful comments for training"""
        lines = code.split('\n')
        commented_lines = []

        for i, line in enumerate(lines):
            if line.strip():
                if language == 'python':
                    if line.strip().startswith('def '):
                        commented_lines.append(f"# Function definition at line {i+1}")
                        commented_lines.append(line)
                    elif line.strip().startswith('class '):
                        commented_lines.append(f"# Class definition at line {i+1}")
                        commented_lines.append(line)
                    elif line.strip().startswith('import ') or line.strip().startswith('from '):
                        commented_lines.append(f"# Import statement at line {i+1}")
                        commented_lines.append(line)
                    else:
                        commented_lines.append(line)
                else:
                    commented_lines.append(line)
            else:
                commented_lines.append(line)

        return '\n'.join(commented_lines)

    def _add_annotations(self, code: str, language: str) -> str:
        """Add annotations explaining code structure"""
        # This is a simplified version - could be enhanced with AST analysis
        annotated = f"# Code in {language}\n"
        annotated += f"# Length: {len(code)} characters\n\n"
        annotated += code
        return annotated

    def get_code_metrics(self, code: str) -> Dict[str, Any]:
        """
        Get various metrics about code

        Args:
            code: Code string

        Returns:
            Dictionary with code metrics
        """
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        metrics = {
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'empty_lines': len(lines) - len(non_empty_lines),
            'character_count': len(code),
            'estimated_complexity': self._estimate_complexity(code),
            'has_functions': 'def ' in code,
            'has_classes': 'class ' in code,
            'has_loops': any(keyword in code for keyword in ['for ', 'while ']),
            'has_conditionals': any(keyword in code for keyword in ['if ', 'elif ', 'else:']),
            'language': self._detect_language(code)
        }

        return metrics

    def _estimate_complexity(self, code: str) -> str:
        """Estimate code complexity based on various factors"""
        complexity_score = 0

        # Count various complexity indicators
        complexity_indicators = [
            ('def ', 2),      # Functions
            ('class ', 3),    # Classes
            ('for ', 2),      # For loops
            ('while ', 2),    # While loops
            ('if ', 1),       # Conditionals
            ('elif ', 1),     # Else if
            ('try:', 2),      # Try blocks
            ('except', 2),    # Exception handling
            ('with ', 1),     # Context managers
        ]

        for indicator, weight in complexity_indicators:
            complexity_score += code.count(indicator) * weight

        if complexity_score < 5:
            return 'simple'
        elif complexity_score < 15:
            return 'moderate'
        elif complexity_score < 30:
            return 'complex'
        else:
            return 'very_complex'

    def _detect_language(self, code: str) -> str:
        """Detect programming language from code snippet"""
        language_patterns = {
            'python': [r'def\s+\w+\s*\(', r'import\s+\w+', r'from\s+\w+\s+import', r'print\s*\('],
            'javascript': [r'function\s+\w+\s*\(', r'const\s+\w+\s*=', r'let\s+\w+\s*=', r'console\.log'],
            'java': [r'public\s+class', r'private\s+\w+\s+\w+\s*\(', r'public\s+static\s+void\s+main'],
            'cpp': [r'#include\s*<', r'std::', r'int\s+main\s*\(', r'cout\s*<<'],
            'html': [r'<html', r'<div', r'<p>', r'<script'],
            'css': [r'\.?\w+\s*\{', r'color\s*:', r'background\s*:'],
        }

        for language, patterns in language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return language

        return 'unknown'