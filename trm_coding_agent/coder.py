"""
Coder environment for TRM Coding Agent.

This module implements the coding environment with tool simulation,
state management, and reward computation for recursive code generation
and refinement.
"""

import os
import sys
import json
import ast
import re
import signal
import importlib
import subprocess
import timeout_decorator
from io import StringIO
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from .config import EnvironmentConfig, ToolConfig, Config
from .utils import simulate_code_execution, validate_syntax, binarize
from .data_handler import DatasetSample

warnings.filterwarnings("ignore", category=FutureWarning)


class ActionType(Enum):
    """Types of actions the agent can take."""
    GENERATE_CODE = "generate_code"
    REFINE_CODE = "refine_code"
    EXECUTE_CODE = "execute_code"
    RUN_TESTS = "run_tests"
    DEBUG_CODE = "debug_code"
    SEARCH_DOCS = "search_docs"
    CHECK_SYNTAX = "check_syntax"
    ANALYZE_ERROR = "analyze_error"
    IMPORT_RESOLVE = "import_resolve"


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoderState:
    """State representation for the coding environment."""
    # Core state
    prompt: str
    current_code: str
    previous_code: str
    execution_history: List[str]
    error_history: List[str]
    tool_usage_history: List[ToolResult]

    # Binary thinking state
    binary_decisions: List[int]
    refinement_needed: bool
    recursion_depth: int

    # Metadata
    step_count: int
    total_reward: float
    episode_done: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            'prompt': self.prompt,
            'current_code': self.current_code,
            'previous_code': self.previous_code,
            'execution_history': self.execution_history,
            'error_history': self.error_history,
            'tool_usage_history': [
                {
                    'tool_name': tr.tool_name,
                    'success': tr.success,
                    'result': str(tr.result)[:200],  # Truncate for serialization
                    'error': tr.error,
                    'execution_time': tr.execution_time,
                    'metadata': tr.metadata
                } for tr in self.tool_usage_history
            ],
            'binary_decisions': self.binary_decisions,
            'refinement_needed': self.refinement_needed,
            'recursion_depth': self.recursion_depth,
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'episode_done': self.episode_done
        }


class ToolSimulator:
    """Simulates various coding tools and utilities."""

    def __init__(self, config: ToolConfig):
        self.config = config
        self.documentation_cache = {}
        self.error_patterns = self._load_error_patterns()

    def _load_error_patterns(self) -> Dict[str, str]:
        """Load common error patterns and solutions."""
        return {
            'SyntaxError': 'Check for missing colons, incorrect indentation, or mismatched parentheses',
            'NameError': 'Variable or function is not defined. Check spelling and scope',
            'TypeError': 'Incorrect type used in operation. Check variable types',
            'ImportError': 'Module not found. Check import statements and module installation',
            'AttributeError': 'Object does not have this attribute. Check object type and available methods',
            'KeyError': 'Key not found in dictionary. Check if key exists',
            'IndexError': 'Index out of range. Check list/dict bounds',
            'ValueError': 'Incorrect value for operation. Check value constraints',
            'ZeroDivisionError': 'Division by zero. Check divisor before division',
            'AssertionError': 'Assertion failed. Check assert conditions',
        }

    def execute_tool(
        self,
        tool_name: str,
        code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Execute a specific tool.

        Args:
            tool_name: Name of the tool to execute
            code: Current code context
            context: Additional context for tool execution

        Returns:
            Tool execution result
        """
        if tool_name == "code_execution":
            return self._execute_code(code)
        elif tool_name == "syntax_checker":
            return self._check_syntax(code)
        elif tool_name == "debugger":
            return self._debug_code(code, context)
        elif tool_name == "test_runner":
            return self._run_tests(code, context)
        elif tool_name == "documentation_search":
            return self._search_documentation(code, context)
        elif tool_name == "import_resolver":
            return self._resolve_imports(code)
        elif tool_name == "error_analyzer":
            return self._analyze_errors(code, context)
        else:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_name}"
            )

    def _execute_code(self, code: str) -> ToolResult:
        """Execute code in safe environment."""
        try:
            import time
            start_time = time.time()

            result = simulate_code_execution(
                code,
                timeout=self.config.execution_timeout
            )

            execution_time = time.time() - start_time

            return ToolResult(
                tool_name="code_execution",
                success=result['success'],
                result=result,
                execution_time=execution_time
            )

        except Exception as e:
            return ToolResult(
                tool_name="code_execution",
                success=False,
                result=None,
                error=str(e)
            )

    def _check_syntax(self, code: str) -> ToolResult:
        """Check code syntax."""
        try:
            import time
            start_time = time.time()

            syntax_result = validate_syntax(code)

            execution_time = time.time() - start_time

            return ToolResult(
                tool_name="syntax_checker",
                success=syntax_result['valid'],
                result=syntax_result,
                execution_time=execution_time
            )

        except Exception as e:
            return ToolResult(
                tool_name="syntax_checker",
                success=False,
                result=None,
                error=str(e)
            )

    def _debug_code(self, code: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Debug code and identify issues."""
        try:
            import time
            start_time = time.time()

            debug_info = {
                'syntax_valid': False,
                'potential_issues': [],
                'suggestions': []
            }

            # Check syntax
            syntax_result = validate_syntax(code)
            debug_info['syntax_valid'] = syntax_result['valid']

            if not syntax_result['valid']:
                debug_info['potential_issues'].append({
                    'type': 'syntax_error',
                    'message': syntax_result['error'],
                    'line_number': syntax_result.get('line_number')
                })

            # Analyze code structure
            try:
                tree = ast.parse(code)

                # Check for common issues
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if len(node.body) == 0:
                            debug_info['potential_issues'].append({
                                'type': 'empty_function',
                                'message': f"Function '{node.name}' is empty",
                                'line_number': node.lineno
                            })
                        elif not any(isinstance(n, ast.Return) for n in ast.walk(node)):
                            debug_info['suggestions'].append({
                                'type': 'missing_return',
                                'message': f"Function '{node.name}' might need a return statement",
                                'line_number': node.lineno
                            })

            except Exception:
                debug_info['potential_issues'].append({
                    'type': 'parse_error',
                    'message': 'Could not parse AST',
                    'line_number': None
                })

            execution_time = time.time() - start_time

            return ToolResult(
                tool_name="debugger",
                success=True,
                result=debug_info,
                execution_time=execution_time
            )

        except Exception as e:
            return ToolResult(
                tool_name="debugger",
                success=False,
                result=None,
                error=str(e)
            )

    def _run_tests(self, code: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Run tests for the code."""
        try:
            import time
            start_time = time.time()

            test_results = {
                'tests_run': 0,
                'tests_passed': 0,
                'test_details': [],
                'coverage_estimate': 0.0
            }

            # Extract test cases from context if available
            test_cases = []
            if context and 'validation' in context:
                test_cases = self._parse_test_cases(context['validation'])

            # If no test cases provided, create basic tests
            if not test_cases:
                test_cases = self._generate_basic_tests(code)

            # Run tests
            for i, test_case in enumerate(test_cases):
                test_results['tests_run'] += 1

                try:
                    # Combine code with test
                    full_code = code + "\n" + test_case['code']

                    result = simulate_code_execution(full_code, timeout=5)

                    if result['success']:
                        test_results['tests_passed'] += 1
                        test_results['test_details'].append({
                            'test_id': i,
                            'status': 'passed',
                            'output': result.get('output', '')
                        })
                    else:
                        test_results['test_details'].append({
                            'test_id': i,
                            'status': 'failed',
                            'error': result.get('error', '')
                        })

                except Exception as e:
                    test_results['test_details'].append({
                        'test_id': i,
                        'status': 'error',
                        'error': str(e)
                    })

            # Calculate coverage estimate
            if test_results['tests_run'] > 0:
                test_results['coverage_estimate'] = test_results['tests_passed'] / test_results['tests_run']

            execution_time = time.time() - start_time

            return ToolResult(
                tool_name="test_runner",
                success=test_results['tests_passed'] > 0,
                result=test_results,
                execution_time=execution_time
            )

        except Exception as e:
            return ToolResult(
                tool_name="test_runner",
                success=False,
                result=None,
                error=str(e)
            )

    def _search_documentation(self, code: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Search documentation for code elements."""
        try:
            import time
            start_time = time.time()

            # Extract function names and imports from code
            doc_results = {
                'functions_found': [],
                'imports_found': [],
                'documentation': {}
            }

            try:
                tree = ast.parse(code)

                # Find functions
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        doc_results['functions_found'].append({
                            'name': node.name,
                            'line': node.lineno,
                            'docstring': ast.get_docstring(node) or "No docstring available"
                        })

                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            doc_results['imports_found'].append({
                                'module': alias.name,
                                'alias': alias.asname,
                                'line': node.lineno
                            })

                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        for alias in node.names:
                            doc_results['imports_found'].append({
                                'module': f"{module}.{alias.name}",
                                'alias': alias.asname,
                                'line': node.lineno
                            })

            except Exception as e:
                doc_results['parse_error'] = str(e)

            # Add mock documentation (in real implementation, would search actual docs)
            for func_info in doc_results['functions_found']:
                func_name = func_info['name']
                if func_name not in self.documentation_cache:
                    self.documentation_cache[func_name] = f"Documentation for {func_name}: This function..."

                doc_results['documentation'][func_name] = self.documentation_cache[func_name]

            execution_time = time.time() - start_time

            return ToolResult(
                tool_name="documentation_search",
                success=True,
                result=doc_results,
                execution_time=execution_time
            )

        except Exception as e:
            return ToolResult(
                tool_name="documentation_search",
                success=False,
                result=None,
                error=str(e)
            )

    def _resolve_imports(self, code: str) -> ToolResult:
        """Resolve and suggest imports for the code."""
        try:
            import time
            start_time = time.time()

            import_results = {
                'missing_imports': [],
                'available_modules': [],
                'suggestions': []
            }

            # Find undefined names in code
            try:
                tree = ast.parse(code)

                # Collect all names
                all_names = set()
                defined_names = set()
                imported_names = set()

                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        all_names.add(node.id)
                    elif isinstance(node, ast.FunctionDef):
                        defined_names.add(node.name)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                defined_names.add(target.id)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            imported_names.add(alias.asname or alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            imported_names.add(alias.asname or alias.name)

                # Find potentially missing imports
                missing_names = all_names - defined_names - imported_names - set(dir(__builtins__))

                # Common standard library modules
                stdlib_modules = [
                    'os', 'sys', 'json', 'math', 'random', 'datetime', 'itertools',
                    'collections', 'functools', 'operator', 're', 'string', 'typing'
                ]

                import_results['available_modules'] = stdlib_modules

                for name in missing_names:
                    if name in stdlib_modules:
                        import_results['missing_imports'].append({
                            'name': name,
                            'suggested_import': f"import {name}",
                            'type': 'stdlib'
                        })
                    elif name.lower() in [m.lower() for m in stdlib_modules]:
                        # Case-insensitive match
                        suggested = next(m for m in stdlib_modules if m.lower() == name.lower())
                        import_results['missing_imports'].append({
                            'name': name,
                            'suggested_import': f"import {suggested}",
                            'type': 'stdlib_case_mismatch'
                        })

            except Exception as e:
                import_results['parse_error'] = str(e)

            execution_time = time.time() - start_time

            return ToolResult(
                tool_name="import_resolver",
                success=True,
                result=import_results,
                execution_time=execution_time
            )

        except Exception as e:
            return ToolResult(
                tool_name="import_resolver",
                success=False,
                result=None,
                error=str(e)
            )

    def _analyze_errors(self, code: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Analyze errors in code execution."""
        try:
            import time
            start_time = time.time()

            error_analysis = {
                'errors_found': [],
                'suggestions': [],
                'patterns_matched': []
            }

            # Get error history from context
            error_history = []
            if context and 'error_history' in context:
                error_history = context['error_history']

            # Analyze each error
            for error_msg in error_history:
                error_info = {
                    'message': error_msg,
                    'type': 'unknown',
                    'suggestions': []
                }

                # Match against known error patterns
                for error_type, suggestion in self.error_patterns.items():
                    if error_type in error_msg:
                        error_info['type'] = error_type
                        error_info['suggestions'].append(suggestion)
                        error_analysis['patterns_matched'].append(error_type)
                        break

                # Specific error analysis
                if 'undefined' in error_msg.lower() or 'not defined' in error_msg.lower():
                    # Extract undefined variable name
                    match = re.search(r"'(\w+)'", error_msg)
                    if match:
                        var_name = match.group(1)
                        error_info['suggestions'].append(f"Check if variable '{var_name}' is defined or imported")

                elif 'index out of range' in error_msg.lower():
                    error_info['suggestions'].append("Check array/list bounds before accessing elements")

                elif 'division by zero' in error_msg.lower():
                    error_info['suggestions'].append("Add zero check before division operations")

                error_analysis['errors_found'].append(error_info)

            execution_time = time.time() - start_time

            return ToolResult(
                tool_name="error_analyzer",
                success=True,
                result=error_analysis,
                execution_time=execution_time
            )

        except Exception as e:
            return ToolResult(
                tool_name="error_analyzer",
                success=False,
                result=None,
                error=str(e)
            )

    def _parse_test_cases(self, validation: str) -> List[Dict[str, Any]]:
        """Parse test cases from validation string."""
        test_cases = []

        lines = validation.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('assert'):
                test_cases.append({
                    'type': 'assert',
                    'code': line
                })

        return test_cases

    def _generate_basic_tests(self, code: str) -> List[Dict[str, Any]]:
        """Generate basic tests for the code."""
        test_cases = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Generate a basic test that tries to call the function
                    func_name = node.name
                    args = [arg.arg for arg in node.args.args]

                    if args:
                        # Create test with simple arguments
                        test_args = ", ".join(["1"] * len(args))
                        test_code = f"print({func_name}({test_args}))"
                    else:
                        test_code = f"print({func_name}())"

                    test_cases.append({
                        'type': 'basic_call',
                        'code': test_code,
                        'function': func_name
                    })

        except Exception:
            pass

        return test_cases


class CoderEnvironment:
    """Main coding environment for TRM agent."""

    def __init__(self, config: Config):
        self.config = config
        self.env_config = config.environment
        self.tool_config = config.tool
        self.tool_simulator = ToolSimulator(self.tool_config)

        # Environment state
        self.current_state: Optional[CoderState] = None
        self.episode_history: List[CoderState] = []

        # Metrics
        self.total_episodes = 0
        self.total_steps = 0
        self.successful_episodes = 0

    def reset(self, sample: DatasetSample) -> CoderState:
        """
        Reset environment with new sample.

        Args:
            sample: Dataset sample to start with

        Returns:
            Initial coder state
        """
        self.current_state = CoderState(
            prompt=sample.prompt,
            current_code="",
            previous_code="",
            execution_history=[],
            error_history=[],
            tool_usage_history=[],
            binary_decisions=[sample.binary_decision],
            refinement_needed=sample.binary_decision == 1,
            recursion_depth=0,
            step_count=0,
            total_reward=0.0,
            episode_done=False
        )

        return self.current_state

    def step(
        self,
        action: ActionType,
        action_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[CoderState, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take
            action_params: Parameters for the action

        Returns:
            Tuple of (new_state, reward, done, info)
        """
        if self.current_state is None:
            raise ValueError("Environment not reset. Call reset() first.")

        if action_params is None:
            action_params = {}

        state = self.current_state
        info = {}

        # Initialize reward
        reward = 0.0

        # Execute action
        if action == ActionType.GENERATE_CODE:
            new_code = action_params.get('code', '')
            state.current_code = new_code
            state.step_count += 1

            # Check syntax
            tool_result = self.tool_simulator.execute_tool(
                "syntax_checker", new_code
            )
            state.tool_usage_history.append(tool_result)

            if tool_result.success:
                reward += self.env_config.syntax_correct_reward
            else:
                reward += self.env_config.syntax_error_penalty
                state.error_history.append(tool_result.result.get('error', 'Syntax error'))

        elif action == ActionType.EXECUTE_CODE:
            if not state.current_code:
                reward += self.env_config.timeout_penalty
            else:
                tool_result = self.tool_simulator.execute_tool(
                    "code_execution", state.current_code
                )
                state.tool_usage_history.append(tool_result)
                state.execution_history.append(str(tool_result.result))

                if tool_result.success:
                    reward += self.env_config.test_pass_reward
                else:
                    reward += self.env_config.runtime_error_penalty
                    if tool_result.error:
                        state.error_history.append(tool_result.error)

        elif action == ActionType.RUN_TESTS:
            if not state.current_code:
                reward += self.env_config.timeout_penalty
            else:
                context = {'validation': action_params.get('validation', '')}
                tool_result = self.tool_simulator.execute_tool(
                    "test_runner", state.current_code, context
                )
                state.tool_usage_history.append(tool_result)

                if tool_result.success:
                    test_results = tool_result.result
                    passed_ratio = test_results['tests_passed'] / max(test_results['tests_run'], 1)
                    reward += self.env_config.test_pass_reward * passed_ratio
                else:
                    reward += self.env_config.runtime_error_penalty

        elif action == ActionType.DEBUG_CODE:
            if state.current_code:
                tool_result = self.tool_simulator.execute_tool(
                    "debugger", state.current_code
                )
                state.tool_usage_history.append(tool_result)

                if tool_result.success:
                    debug_info = tool_result.result
                    issues_found = len(debug_info['potential_issues'])
                    if issues_found == 0:
                        reward += 0.5  # Small reward for clean code
                    else:
                        reward -= 0.1 * issues_found  # Penalty for issues

        elif action == ActionType.SEARCH_DOCS:
            if state.current_code:
                tool_result = self.tool_simulator.execute_tool(
                    "documentation_search", state.current_code
                )
                state.tool_usage_history.append(tool_result)

                if tool_result.success:
                    doc_results = tool_result.result
                    functions_found = len(doc_results['functions_found'])
                    reward += 0.1 * functions_found  # Small reward for documentation

        elif action == ActionType.ANALYZE_ERROR:
            if state.error_history:
                context = {'error_history': state.error_history}
                tool_result = self.tool_simulator.execute_tool(
                    "error_analyzer", state.current_code, context
                )
                state.tool_usage_history.append(tool_result)

                if tool_result.success:
                    error_analysis = tool_result.result
                    patterns_matched = len(error_analysis['patterns_matched'])
                    reward += 0.2 * patterns_matched  # Reward for identifying error patterns

        # Update recursion depth for TRM
        if action in [ActionType.GENERATE_CODE, ActionType.REFINE_CODE]:
            state.recursion_depth += 1

        # Binary decision logic
        if state.recursion_depth < self.config.trm.recursion_depth:
            # Determine if further refinement is needed
            binary_decision = self._make_binary_decision(state)
            state.binary_decisions.append(binary_decision)
            state.refinement_needed = binary_decision == 1

        # Check episode termination
        done = self._check_episode_done(state)

        if done:
            state.episode_done = True
            self.total_episodes += 1
            if state.total_reward > 0:
                self.successful_episodes += 1

        state.total_reward += reward
        self.total_steps += 1

        info.update({
            'step_count': state.step_count,
            'recursion_depth': state.recursion_depth,
            'binary_decision': state.binary_decisions[-1] if state.binary_decisions else 0,
            'total_episodes': self.total_episodes,
            'success_rate': self.successful_episodes / max(self.total_episodes, 1)
        })

        return state, reward, done, info

    def _make_binary_decision(self, state: CoderState) -> int:
        """Make binary decision for refinement."""
        # Decision factors
        factors = []

        # Recent errors
        if state.error_history:
            factors.append(1)  # Recent errors suggest refinement needed

        # Low reward
        if state.total_reward < 0:
            factors.append(1)

        # Syntax issues
        if state.tool_usage_history:
            last_tool = state.tool_usage_history[-1]
            if last_tool.tool_name == "syntax_checker" and not last_tool.success:
                factors.append(1)

        # Recursion depth limit
        if state.recursion_depth >= self.config.trm.recursion_depth - 2:
            factors.append(0)  # Force stop near limit

        # Binary logic: if any factor suggests refinement, return 1
        return 1 if any(factors) else 0

    def _check_episode_done(self, state: CoderState) -> bool:
        """Check if episode should terminate."""
        # Termination conditions
        if state.step_count >= self.env_config.max_episode_length:
            return True

        if state.recursion_depth >= self.config.trm.recursion_depth:
            return True

        if not state.refinement_needed and state.recursion_depth > 2:
            return True

        if state.total_reward > self.env_config.test_pass_reward:
            return True

        return False

    def get_state_representation(self, state: CoderState) -> jnp.ndarray:
        """
        Convert state to numerical representation for model input.

        Args:
            state: Current coder state

        Returns:
            Numerical state representation
        """
        # This is a simplified representation
        # In practice, would use more sophisticated encoding
        state_features = [
            len(state.prompt) / 1000.0,  # Normalized prompt length
            len(state.current_code) / 1000.0,  # Normalized code length
            len(state.execution_history) / 10.0,  # Normalized execution count
            len(state.error_history) / 10.0,  # Normalized error count
            state.recursion_depth / 16.0,  # Normalized recursion depth
            state.step_count / 100.0,  # Normalized step count
            float(state.binary_decisions[-1] if state.binary_decisions else 0),
            float(state.refinement_needed),
            state.total_reward / 100.0,  # Normalized reward
            len(state.tool_usage_history) / 20.0,  # Normalized tool usage
        ]

        return jnp.array(state_features, dtype=jnp.float32)

    def render(self, state: Optional[CoderState] = None) -> str:
        """
        Render current state for visualization.

        Args:
            state: State to render (uses current if None)

        Returns:
            String representation of state
        """
        if state is None:
            state = self.current_state

        if state is None:
            return "No active state"

        output = []
        output.append(f"=== Coding Environment State ===")
        output.append(f"Step: {state.step_count}")
        output.append(f"Recursion Depth: {state.recursion_depth}")
        output.append(f"Total Reward: {state.total_reward:.2f}")
        output.append(f"Binary Decisions: {state.binary_decisions}")
        output.append(f"Refinement Needed: {state.refinement_needed}")
        output.append("")
        output.append("Prompt:")
        output.append(state.prompt[:200] + "..." if len(state.prompt) > 200 else state.prompt)
        output.append("")
        output.append("Current Code:")
        output.append(state.current_code[:300] + "..." if len(state.current_code) > 300 else state.current_code)
        output.append("")
        output.append("Recent Errors:")
        for error in state.error_history[-3:]:
            output.append(f"  - {error[:100]}...")
        output.append("")
        output.append("Recent Tools:")
        for tool in state.tool_usage_history[-3:]:
            output.append(f"  - {tool.tool_name}: {'Success' if tool.success else 'Failed'}")

        return "\n".join(output)

    def get_metrics(self) -> Dict[str, Any]:
        """Get environment metrics."""
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'successful_episodes': self.successful_episodes,
            'success_rate': self.successful_episodes / max(self.total_episodes, 1),
            'average_steps_per_episode': self.total_steps / max(self.total_episodes, 1),
        }


if __name__ == "__main__":
    # Test coder environment
    from .config import get_config
    from .data_handler import DatasetSample

    config = get_config("debug")
    env = CoderEnvironment(config)

    print("Testing coder environment...")

    # Create test sample
    test_sample = DatasetSample(
        prompt="Write a function that adds two numbers",
        solution="def add(a, b): return a + b",
        validation="assert add(2, 3) == 5",
        binary_decision=1,
        metadata={'dataset': 'test'}
    )

    # Reset environment
    state = env.reset(test_sample)
    print(f"Initial state: {state.prompt[:50]}...")

    # Test actions
    state, reward, done, info = env.step(
        ActionType.GENERATE_CODE,
        {'code': 'def add(a, b): return a + b'}
    )
    print(f"After generation: reward={reward}, done={done}")

    state, reward, done, info = env.step(
        ActionType.RUN_TESTS,
        {'validation': 'assert add(2, 3) == 5'}
    )
    print(f"After testing: reward={reward}, done={done}")

    print(f"Final metrics: {env.get_metrics()}")
    print("Coder environment tests completed!")