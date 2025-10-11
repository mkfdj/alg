"""
Prompt formatting utilities for different training formats
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum


class PromptFormat(Enum):
    """Different prompt formatting styles"""
    ALPACA = "alpaca"
    CHAT = "chat"
    INSTRUCTION = "instruction"
    CODE_COMPLETION = "code_completion"
    QA = "qa"


class PromptFormatter:
    """Utilities for formatting prompts for different training formats"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def format_prompts(self, prompts: List[str], responses: List[str] = None,
                      format_type: PromptFormat = PromptFormat.INSTRUCTION,
                      config: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Format prompts according to specified format

        Args:
            prompts: List of prompt strings
            responses: List of response strings (optional)
            format_type: Type of formatting to apply
            config: Additional configuration

        Returns:
            List of formatted examples
        """
        if config is None:
            config = {}

        if format_type == PromptFormat.ALPACA:
            return self._format_alpaca(prompts, responses, config)
        elif format_type == PromptFormat.CHAT:
            return self._format_chat(prompts, responses, config)
        elif format_type == PromptFormat.INSTRUCTION:
            return self._format_instruction(prompts, responses, config)
        elif format_type == PromptFormat.CODE_COMPLETION:
            return self._format_code_completion(prompts, responses, config)
        elif format_type == PromptFormat.QA:
            return self._format_qa(prompts, responses, config)
        else:
            return self._format_basic(prompts, responses, config)

    def _format_alpaca(self, prompts: List[str], responses: List[str] = None,
                      config: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """Format prompts in Alpaca style"""
        if config is None:
            config = {}

        examples = []
        add_input_field = config.get('add_input_field', True)
        default_input = config.get('default_input', '')

        for i, prompt in enumerate(prompts):
            example = {
                'instruction': prompt,
                'input': default_input,
                'output': responses[i] if responses and i < len(responses) else ''
            }

            if not add_input_field:
                del example['input']

            examples.append(example)

        return examples

    def _format_chat(self, prompts: List[str], responses: List[str] = None,
                    config: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """Format prompts for chat training"""
        if config is None:
            config = {}

        system_message = config.get('system_message', 'You are a helpful coding assistant.')
        examples = []

        for i, prompt in enumerate(prompts):
            messages = [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}
            ]

            if responses and i < len(responses):
                messages.append({'role': 'assistant', 'content': responses[i]})

            examples.append({
                'messages': messages,
                'prompt': prompt,
                'response': responses[i] if responses and i < len(responses) else ''
            })

        return examples

    def _format_instruction(self, prompts: List[str], responses: List[str] = None,
                           config: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """Format prompts as instruction-response pairs"""
        examples = []

        for i, prompt in enumerate(prompts):
            example = {
                'instruction': prompt,
                'response': responses[i] if responses and i < len(responses) else ''
            }
            examples.append(example)

        return examples

    def _format_code_completion(self, prompts: List[str], responses: List[str] = None,
                               config: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """Format prompts for code completion tasks"""
        if config is None:
            config = {}

        add_prefix = config.get('add_prefix', True)
        prefix_text = config.get('prefix_text', 'Complete the following code:\n\n')

        examples = []

        for i, prompt in enumerate(prompts):
            formatted_prompt = prompt
            if add_prefix and not prompt.startswith(prefix_text):
                formatted_prompt = prefix_text + prompt

            example = {
                'prompt': formatted_prompt,
                'completion': responses[i] if responses and i < len(responses) else '',
                'original_prompt': prompt
            }
            examples.append(example)

        return examples

    def _format_qa(self, prompts: List[str], responses: List[str] = None,
                  config: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """Format prompts as question-answer pairs"""
        examples = []

        for i, prompt in enumerate(prompts):
            # Try to detect if prompt is already a question
            question = prompt
            if not prompt.strip().endswith('?'):
                question = prompt + '?'

            example = {
                'question': question,
                'answer': responses[i] if responses and i < len(responses) else '',
                'context': ''
            }
            examples.append(example)

        return examples

    def _format_basic(self, prompts: List[str], responses: List[str] = None,
                     config: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """Basic prompt formatting"""
        examples = []

        for i, prompt in enumerate(prompts):
            example = {
                'prompt': prompt,
                'response': responses[i] if responses and i < len(responses) else ''
            }
            examples.append(example)

        return examples

    def add_task_specific_instructions(self, prompts: List[str],
                                     task_type: str = 'coding') -> List[str]:
        """
        Add task-specific instructions to prompts

        Args:
            prompts: List of prompt strings
            task_type: Type of task (coding, debugging, explanation, etc.)

        Returns:
            List of prompts with added instructions
        """
        instructions = {
            'coding': "Write code to solve the following problem:",
            'debugging': "Debug and fix the following code:",
            'explanation': "Explain what the following code does:",
            'optimization': "Optimize the following code:",
            'completion': "Complete the following code:",
            'conversion': "Convert the following code to a different language:",
            'testing': "Write tests for the following code:",
            'documentation': "Write documentation for the following code:"
        }

        instruction = instructions.get(task_type, instructions['coding'])

        enhanced_prompts = []
        for prompt in prompts:
            if not prompt.startswith(instruction):
                enhanced_prompt = f"{instruction}\n\n{prompt}"
            else:
                enhanced_prompt = prompt
            enhanced_prompts.append(enhanced_prompt)

        return enhanced_prompts

    def create_few_shot_examples(self, prompts: List[str], responses: List[str] = None,
                               num_examples: int = 3) -> List[Dict[str, str]]:
        """
        Create few-shot learning examples

        Args:
            prompts: List of prompts
            responses: List of responses
            num_examples: Number of examples to include

        Returns:
            List of few-shot examples
        """
        if not prompts:
            return []

        examples = []
        n = min(num_examples, len(prompts))

        for i in range(n):
            example = {
                'demonstration_prompt': prompts[i],
                'demonstration_response': responses[i] if responses and i < len(responses) else '',
                'target_prompt': prompts[i + n] if i + n < len(prompts) else prompts[-1],
                'target_response': responses[i + n] if responses and i + n < len(responses) else ''
            }
            examples.append(example)

        return examples

    def format_for_evaluation(self, prompts: List[str], format_type: str = 'openai') -> List[Dict[str, Any]]:
        """
        Format prompts for evaluation/testing

        Args:
            prompts: List of prompts
            format_type: Format type ('openai', 'huggingface', 'custom')

        Returns:
            Formatted evaluation prompts
        """
        if format_type == 'openai':
            return [{'prompt': prompt, 'max_tokens': 1000} for prompt in prompts]
        elif format_type == 'huggingface':
            return [{'text': prompt} for prompt in prompts]
        elif format_type == 'custom':
            return [{'input': prompt, 'expected_output': ''} for prompt in prompts]
        else:
            return [{'prompt': prompt} for prompt in prompts]

    def create_code_instruction_variations(self, prompts: List[str]) -> List[str]:
        """
        Create variations of coding instruction prompts

        Args:
            prompts: Original prompts

        Returns:
            List of prompt variations
        """
        variations = []
        prefixes = [
            "Write a function that",
            "Create a Python function to",
            "Implement code for",
            "Develop a solution for",
            "Write code to"
        ]

        suffixes = [
            "",
            "\n\nProvide only the code, no explanation.",
            "\n\nInclude comments explaining the logic.",
            "\n\nMake sure to handle edge cases.",
            "\n\nThe function should be efficient and well-documented."
        ]

        for prompt in prompts:
            # Original prompt
            variations.append(prompt)

            # Add variations with different prefixes
            for prefix in prefixes:
                if not prompt.startswith(prefix):
                    variations.append(f"{prefix} {prompt.lstrip()}")

            # Add variations with different suffixes
            for suffix in suffixes:
                if suffix and not prompt.endswith(suffix):
                    variations.append(f"{prompt}{suffix}")

        return variations

    def generate_training_statistics(self, examples: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generate statistics about formatted examples

        Args:
            examples: List of formatted examples

        Returns:
            Statistics dictionary
        """
        if not examples:
            return {}

        stats = {
            'total_examples': len(examples),
            'has_responses': all('response' in ex and ex['response'] for ex in examples),
            'avg_prompt_length': 0,
            'avg_response_length': 0,
            'fields': list(examples[0].keys()) if examples else []
        }

        prompt_lengths = []
        response_lengths = []

        for ex in examples:
            if 'prompt' in ex:
                prompt_lengths.append(len(ex['prompt']))
            if 'instruction' in ex:
                prompt_lengths.append(len(ex['instruction']))
            if 'response' in ex and ex['response']:
                response_lengths.append(len(ex['response']))
            if 'output' in ex and ex['output']:
                response_lengths.append(len(ex['output']))

        if prompt_lengths:
            stats['avg_prompt_length'] = round(sum(prompt_lengths) / len(prompt_lengths), 2)
            stats['min_prompt_length'] = min(prompt_lengths)
            stats['max_prompt_length'] = max(prompt_lengths)

        if response_lengths:
            stats['avg_response_length'] = round(sum(response_lengths) / len(response_lengths), 2)
            stats['min_response_length'] = min(response_lengths)
            stats['max_response_length'] = max(response_lengths)

        return stats