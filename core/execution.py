import asyncio
import io
import sys
import os
import builtins
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Dict, Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.config import GLOBAL_MAX_TIME, MAX_STDOUT_LENGTH

# Global exit-function patch state to avoid conflicts across multiple CodeExecutor instances
_EXIT_PATCHED: bool = False
_EXIT_PATCH_COUNT: int = 0
_ORIGINAL_EXITS: Dict[str, object] = {}


@dataclass
class ExecutionResult:
    stdout: str
    error_trace: Optional[str] = None
    error: Optional[Exception] = None
    is_timeout: bool = False


class CodeExecutor:
    @staticmethod
    def custom_exit(code=0):
        raise RuntimeError(f"Exit requested with code {code}")

    def __init__(self, max_stdout_length: Optional[int] = None):
        self.namespace: Dict = {}
        self.setup_code: Optional[str] = None
        self.max_stdout_length: int = max_stdout_length if max_stdout_length is not None else MAX_STDOUT_LENGTH

        # Patch exit functions in a process-wide, reference-counted manner to avoid conflicts
        global _EXIT_PATCHED, _EXIT_PATCH_COUNT, _ORIGINAL_EXITS
        if not _EXIT_PATCHED:
            _ORIGINAL_EXITS = {
                'builtin_exit': builtins.exit,
                'sys_exit': sys.exit,
                'os_exit': os._exit,
            }
            # Use class-level custom_exit to avoid binding to a particular instance
            builtins.exit = CodeExecutor.custom_exit
            sys.exit = CodeExecutor.custom_exit
            os._exit = CodeExecutor.custom_exit
            _EXIT_PATCHED = True
            _EXIT_PATCH_COUNT = 1
        else:
            _EXIT_PATCH_COUNT += 1

    def __del__(self):
        # Restore original functions when the last executor is garbage-collected
        global _EXIT_PATCHED, _EXIT_PATCH_COUNT, _ORIGINAL_EXITS
        try:
            if _EXIT_PATCHED:
                _EXIT_PATCH_COUNT -= 1
                if _EXIT_PATCH_COUNT <= 0:
                    builtins.exit = _ORIGINAL_EXITS.get('builtin_exit', builtins.exit)
                    sys.exit = _ORIGINAL_EXITS.get('sys_exit', sys.exit)
                    os._exit = _ORIGINAL_EXITS.get('os_exit', os._exit)
                    _EXIT_PATCHED = False
                    _EXIT_PATCH_COUNT = 0
                    _ORIGINAL_EXITS = {}
        except Exception:
            # Be conservative: never raise from __del__
            pass

    def set_setup_code(self, setup_code: str):
        self.setup_code = setup_code

    def clear_namespace(self):
        self.namespace.clear()

    async def execute(self, code: str, timeout: float = GLOBAL_MAX_TIME) -> ExecutionResult:
        # Initialize namespace with environment setup if empty
        if not self.namespace:
            if self.setup_code:
                exec(self.setup_code, self.namespace)
            # Provide a local 'exit' that raises instead of terminating the process
            self.namespace['exit'] = CodeExecutor.custom_exit

        stdout = io.StringIO()

        # Define a synchronous function to run in a background thread
        def _exec_sync() -> None:
            with redirect_stdout(stdout):
                exec(code, self.namespace)

        try:
            loop = asyncio.get_running_loop()
            # Run the blocking execution in a background thread and enforce timeout
            await asyncio.wait_for(loop.run_in_executor(None, _exec_sync), timeout)

            # Truncate stdout if too long
            stdout_content = stdout.getvalue()
            if len(stdout_content) > self.max_stdout_length:
                truncation_msg = f"\n[Max character {self.max_stdout_length} reached, truncated]"
                stdout_content = stdout_content[:self.max_stdout_length] + truncation_msg

            return ExecutionResult(
                stdout=stdout_content
            )
        except Exception as e:
            error_trace_str = traceback.format_exc()
            
            # Truncate stdout if too long
            stdout_content = stdout.getvalue()
            if len(stdout_content) > self.max_stdout_length:
                truncation_msg = f"\n[Max character {self.max_stdout_length} reached, truncated]"
                stdout_content = stdout_content[:self.max_stdout_length] + truncation_msg
            
            return ExecutionResult(
                stdout=stdout_content,
                error_trace=error_trace_str,
                error=e,
                is_timeout=isinstance(e, asyncio.TimeoutError)
            )

    async def reset_and_execute_to_step(
            self,
            task_context: 'TaskContext',
            step: int,
            time_out: float
    ) -> Optional[ExecutionResult]:
        """Clear namespace and execute all code snippets up to given step"""
        self.clear_namespace()
        code = task_context.concatenate_snippets(end_step=step)
        return await self.execute(code, timeout=time_out)


# Example usage and tests
async def main():
    executor = CodeExecutor()

    # Test 1: Normal execution
    result = await executor.execute('print("hello world")')
    print("Test 1 - Normal execution:")
    print(f"stdout: {result.stdout}")
    print(f"error: {result.error}\n")

    # Test 2: Exit attempt with exit()
    result = await executor.execute('exit(1)')
    print("Test 2 - exit() attempt:")
    print(f"stdout: {result.stdout}")
    print(f"error: {result.error}\n")

    # Test 3: Exit attempt with sys.exit()
    result = await executor.execute('import sys; sys.exit(2)')
    print("Test 3 - sys.exit() attempt:")
    print(f"stdout: {result.stdout}")
    print(f"error: {result.error}\n")

    # Test 4: Exit attempt with os._exit()
    result = await executor.execute('import os; os._exit(3)')
    print("Test 4 - os._exit() attempt:")
    print(f"stdout: {result.stdout}")
    print(f"error: {result.error}\n")

    # Test 5: Timeout
    result = await executor.execute('while True: pass', timeout=1.0)
    print("Test 5 - Timeout test:")
    print(f"stdout: {result.stdout}")
    print(f"error: {result.error}")
    print(f"is_timeout: {result.is_timeout}\n")


if __name__ == "__main__":
    asyncio.run(main())