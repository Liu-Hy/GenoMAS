import sys
import contextlib
from typing import Optional, Any, Callable
import traceback


class SafeExecutionError(Exception):
    """Custom exception for safe execution failures"""
    def __init__(self, error_type: str, original_error: Exception):
        self.error_type = error_type
        self.original_error = original_error
        super().__init__(f"{error_type}: {str(original_error)}")


@contextlib.contextmanager
def safe_execution_context():
    """
    A context manager that prevents exit() calls from terminating the program.
    Instead, it raises a SafeExecutionError that can be caught and handled.
    """
    # Store the original exit function
    original_exit = sys.exit
    
    def custom_exit(code=0):
        raise SafeExecutionError("SystemExit", Exception(f"Exit called with code {code}"))
    
    try:
        # Replace system exit with our custom exit
        sys.exit = custom_exit
        yield
    finally:
        # Restore the original exit function
        sys.exit = original_exit


def safe_execute(func: Callable, *args, **kwargs) -> tuple[bool, Optional[Any], Optional[Exception]]:
    """
    Safely executes a function that might contain exit() calls or raise exceptions.
    
    Args:
        func: The function to execute
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        tuple containing:
        - success (bool): Whether the execution was successful
        - result (Any): The result of the function if successful, None otherwise
        - error (Exception): The exception that occurred if unsuccessful, None otherwise
    """
    try:
        with safe_execution_context():
            result = func(*args, **kwargs)
            return True, result, None
    except SafeExecutionError as e:
        return False, None, e
    except Exception as e:
        return False, None, e
