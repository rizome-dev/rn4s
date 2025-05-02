"""
Code execution system for the agent.
Inspired by smolagents' implementation of Python code execution.
"""

import sys
import re
import traceback
import io
import inspect
import builtins
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
import asyncio
import anyio
from pydantic import BaseModel, Field

class CodeExecutionResult(BaseModel):
    """Result of a code execution."""
    stdout: str = Field("", description="Standard output from code execution")
    stderr: str = Field("", description="Standard error from code execution")
    result: Any = Field(None, description="Return value of the executed code")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    exception_type: Optional[str] = Field(None, description="Type of exception if execution failed")
    success: bool = Field(True, description="Whether the execution was successful")
    execution_time: float = Field(0.0, description="Time taken for execution in seconds")

class CodeExecutor:
    """
    Safe code executor for running Python code with state persistence.
    Inspired by smolagents' implementation for improved safety and UX.
    """
    
    def __init__(
        self,
        timeout: float = 10.0,
        max_output_size: int = 10000,
        allowed_imports: Optional[List[str]] = None,
        restricted_builtins: Optional[Set[str]] = None,
        additional_globals: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        strict_imports: bool = True,
    ):
        """
        Initialize the code executor.
        
        Args:
            timeout: Maximum execution time in seconds
            max_output_size: Maximum output size in characters
            allowed_imports: List of allowed import modules (None means standard restrictions)
            restricted_builtins: Set of restricted builtin functions
            additional_globals: Additional globals to add to the execution environment
            debug: Whether to enable debug mode (prints more information)
            strict_imports: Whether to strictly enforce allowed imports
        """
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.debug = debug
        self.strict_imports = strict_imports
        self.session_id = str(uuid.uuid4())[:8]  # For tracking execution sessions
        
        # Default allowed imports (safe standard library modules)
        self.allowed_imports = allowed_imports or [
            'math', 'random', 'datetime', 'time', 'json', 're', 'collections',
            'itertools', 'functools', 'string', 'textwrap', 'uuid', 'hashlib',
            'copy', 'typing', 'dataclasses', 'enum', 'statistics', 'csv', 
            'base64', 'urllib.parse', 'zlib', 'gzip', 'decimal',
        ]
        
        # Default restricted builtins (potentially dangerous operations)
        self.restricted_builtins = restricted_builtins or {
            'exec', 'eval', 'compile', '__import__', 'open', 'input',
            'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr',
            'breakpoint', 'help', 'memoryview', 'object', 'staticmethod', 
            'classmethod', 'property', 'super', 'type', '__build_class__'
        }
        
        # Store execution history for debugging purposes
        self.execution_history = []
        
        # Internal state for execution
        self.globals = self._create_safe_globals()
        
        # Add additional globals
        if additional_globals:
            for name, value in additional_globals.items():
                self.globals[name] = value
        
        # Add helper functions commonly used in agents
        self._add_helper_functions()
        
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a safe globals dictionary for execution."""
        # Start with a clean environment
        safe_globals = {'__builtins__': {}}
        
        # Add safe built-ins
        for name in dir(builtins):
            if name not in self.restricted_builtins:
                safe_globals['__builtins__'][name] = getattr(builtins, name)
        
        # Add print function that respects max_output_size
        safe_globals['__builtins__']['print'] = self._create_safe_print()
        
        # Add a safe version of import
        safe_globals['__builtins__']['__import__'] = self._create_safe_import()
        
        return safe_globals
    
    def _create_safe_print(self) -> Callable:
        """
        Create a safe print function that respects max_output_size.
        
        Returns:
            A safe print function
        """
        orig_print = builtins.print
        max_size = self.max_output_size
        output_counter = {'size': 0}
        
        def safe_print(*args, **kwargs):
            """A print function that respects max_output_size."""
            if output_counter['size'] >= max_size:
                return
                
            try:
                # Use StringIO to capture the output
                buf = io.StringIO()
                kwargs['file'] = buf
                orig_print(*args, **kwargs)
                output = buf.getvalue()
                
                # Check if adding this would exceed the limit
                remaining = max_size - output_counter['size']
                if len(output) > remaining:
                    output = output[:remaining] + "\n... [output truncated]"
                    output_counter['size'] = max_size
                else:
                    output_counter['size'] += len(output)
                
                # Print to the actual stdout
                kwargs['file'] = sys.stdout
                orig_print(output, end='')
            except Exception as e:
                orig_print(f"Error in safe_print: {e}", file=sys.stderr)
                
        return safe_print
    
    def _create_safe_import(self) -> Callable:
        """
        Create a safe import function that only allows specified modules.
        
        Returns:
            A safe import function
        """
        orig_import = builtins.__import__
        allowed_imports = self.allowed_imports
        strict = self.strict_imports
        debug = self.debug
        
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            """A safe import function that only allows specified modules."""
            base_module = name.split('.')[0]
            
            if base_module in allowed_imports:
                try:
                    return orig_import(name, globals, locals, fromlist, level)
                except ImportError as e:
                    if debug:
                        print(f"Import error for {name}: {str(e)}", file=sys.stderr)
                    raise
            elif strict:
                raise ImportError(f"Module '{base_module}' is not in the allowed imports list")
            else:
                print(f"Warning: Import of '{base_module}' was blocked for security reasons", file=sys.stderr)
                # Return a dummy module that raises AttributeError for any access
                class DummyModule:
                    def __getattr__(self, attr):
                        raise AttributeError(f"Module '{name}' is not available")
                return DummyModule()
                
        return safe_import
    
    def _add_helper_functions(self):
        """Add helper functions to the globals."""
        # Add a final_answer function that agents can use to indicate completion
        def final_answer(answer):
            """Indicate a final answer for the task."""
            return f"FINAL_ANSWER: {answer}"
        
        self.globals['final_answer'] = final_answer
    
    def add_tool(self, name: str, tool_function: Callable):
        """
        Add a tool to the globals for execution.
        
        Args:
            name: Name of the tool to add
            tool_function: Function to add as a tool
        """
        self.globals[name] = tool_function
    
    async def execute_code(self, code: str) -> CodeExecutionResult:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            
        Returns:
            CodeExecutionResult with execution details
        """
        # Process the code to handle potential imports
        processed_code = self._process_code(code)
        
        # Set up capture of stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Set up the execution environment - no need to copy globals to persist state
        exec_globals = self.globals
        
        # Track execution time
        start_time = time.time()
        success = True
        error_msg = None
        exception_type = None
        result = None
        
        try:
            # Run in separate thread to allow for timeout
            with anyio.move_on_after(self.timeout) as scope:
                def execute():
                    nonlocal result
                    try:
                        # Redirect stdout and stderr
                        sys_stdout, sys_stderr = sys.stdout, sys.stderr
                        sys.stdout, sys.stderr = stdout_capture, stderr_capture
                        
                        try:
                            # Execute the code
                            if self._is_single_expression(processed_code):
                                # For simple expressions, capture the result
                                result = eval(processed_code, exec_globals)
                            else:
                                # For code blocks, execute and look for result variable
                                exec(processed_code, exec_globals)
                                # Check if there's a result variable after execution
                                if 'result' in exec_globals:
                                    result = exec_globals['result']
                                # Check if the code called final_answer
                                elif '_final_answer' in exec_globals:
                                    result = exec_globals['_final_answer']
                        finally:
                            # Restore stdout and stderr
                            sys.stdout, sys.stderr = sys_stdout, sys_stderr
                    except Exception as e:
                        # Re-raise to be caught outside
                        raise e
                
                await anyio.to_thread.run_sync(execute)
            
            # Check if we timed out
            if scope.cancel_called:
                success = False
                error_msg = f"Execution timed out after {self.timeout} seconds"
                exception_type = "TimeoutError"
        
        except Exception as e:
            success = False
            error_msg = str(e)
            exception_type = e.__class__.__name__
            
            # Get traceback for detailed error info
            tb = traceback.format_exc()
            print(f"Code execution error: {tb}", file=stderr_capture)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Get captured output, truncating if necessary
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        
        if len(stdout) > self.max_output_size:
            stdout = stdout[:self.max_output_size] + "... [output truncated]"
        if len(stderr) > self.max_output_size:
            stderr = stderr[:self.max_output_size] + "... [output truncated]"
        
        # Create the result
        execution_result = CodeExecutionResult(
            stdout=stdout,
            stderr=stderr,
            result=result,
            error=error_msg,
            exception_type=exception_type,
            success=success,
            execution_time=execution_time
        )
        
        # Add to execution history
        self.execution_history.append({
            "code": code,
            "success": success,
            "execution_time": execution_time,
            "timestamp": time.time()
        })
        
        return execution_result
    
    def _process_code(self, code: str) -> str:
        """
        Process code to handle imports safely.
        
        Args:
            code: Original code
            
        Returns:
            Processed code with import validation
        """
        # Check for import statements
        import_pattern = r"^\s*(import|from)\s+([^\s]+)"
        imports = re.findall(import_pattern, code, re.MULTILINE)
        
        processed_lines = []
        for line in code.split('\n'):
            # Check if this line has an import
            match = re.match(import_pattern, line.strip())
            if match:
                import_type, module_name = match.groups()
                base_module = module_name.split('.')[0]
                
                if base_module not in self.allowed_imports:
                    # If strict imports, add a line that will raise an ImportError
                    if self.strict_imports:
                        processed_lines.append(
                            f"# BLOCKED IMPORT: {line.strip()}\n"
                            f"raise ImportError(\"Module '{base_module}' is not in the allowed imports list\")"
                        )
                    else:
                        # Just add a comment warning
                        processed_lines.append(f"# BLOCKED IMPORT: {line.strip()}")
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _is_single_expression(self, code: str) -> bool:
        """
        Check if code is a single expression (for eval).
        
        Args:
            code: Code to check
            
        Returns:
            True if code is a single expression
        """
        code = code.strip()
        if not code:
            return False
            
        # Check for assignment, imports, function definitions, etc.
        if any(keyword in code for keyword in ['=', 'def ', 'class ', 'import ', 'from ']):
            return False
            
        # Check if it spans multiple lines (ignoring comments and whitespace)
        meaningful_lines = [line for line in code.split('\n') 
                           if line.strip() and not line.strip().startswith('#')]
        if len(meaningful_lines) > 1:
            return False
        
        # Try to compile as an expression
        try:
            compile(code, '<string>', 'eval')
            return True
        except SyntaxError:
            return False
    
    def reset(self, keep_tools: bool = True):
        """
        Reset the execution environment.
        
        Args:
            keep_tools: Whether to keep added tools
        """
        # Store tools if needed
        tools = {}
        if keep_tools:
            for name, value in self.globals.items():
                if callable(value) and name != 'final_answer' and not name.startswith('__'):
                    tools[name] = value
        
        # Reset globals
        self.globals = self._create_safe_globals()
        
        # Re-add tools if keeping them
        if keep_tools:
            for name, func in tools.items():
                self.globals[name] = func
        
        # Re-add helper functions
        self._add_helper_functions()
    
    def get_variables(self) -> Dict[str, Any]:
        """
        Get all user-defined variables in the execution environment.
        
        Returns:
            Dictionary of user-defined variables
        """
        variables = {}
        
        for name, value in self.globals.items():
            # Skip private and predefined variables
            if (not name.startswith('__') and 
                name not in dir(builtins) and 
                name not in ['final_answer']):
                # Skip functions and modules
                if not inspect.ismodule(value) and not inspect.isfunction(value):
                    variables[name] = value
        
        return variables
    
    def get_defined_functions(self) -> Dict[str, Callable]:
        """
        Get all user-defined functions in the execution environment.
        
        Returns:
            Dictionary of user-defined functions
        """
        functions = {}
        
        for name, value in self.globals.items():
            # Check for user-defined functions
            if (inspect.isfunction(value) and 
                not name.startswith('__') and 
                name not in ['final_answer']):
                functions[name] = value
        
        return functions 