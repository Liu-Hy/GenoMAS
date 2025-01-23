import asyncio
import io
import sys
import os
import builtins
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Dict, Optional

from utils.config import GLOBAL_MAX_TIME


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

    def __init__(self):
        self.namespace: Dict = {}
        self.setup_code: Optional[str] = None

        # Store original functions for cleanup
        self._original_exits = {
            'builtin_exit': builtins.exit,
            'sys_exit': sys.exit,
            'os_exit': os._exit
        }

        # Replace all exit functions
        builtins.exit = self.custom_exit
        sys.exit = self.custom_exit
        os._exit = os._exit = self.custom_exit  # lambda code=0: self.custom_exit(code)

    def __del__(self):
        # Restore original functions on cleanup
        builtins.exit = self._original_exits['builtin_exit']
        sys.exit = self._original_exits['sys_exit']
        os._exit = self._original_exits['os_exit']

    def set_setup_code(self, setup_code: str):
        self.setup_code = setup_code

    def clear_namespace(self):
        self.namespace.clear()

    async def execute(self, code: str, timeout: float = GLOBAL_MAX_TIME) -> ExecutionResult:
        # Initialize namespace with environment setup if empty
        if not self.namespace:
            if self.setup_code:
                exec(self.setup_code, self.namespace)
            self.namespace['exit'] = self.custom_exit  # Preserve exit handling

        stdout = io.StringIO()

        try:
            async def exec_with_timeout():
                with redirect_stdout(stdout):
                    exec(code, self.namespace)

            await asyncio.wait_for(exec_with_timeout(), timeout)

            return ExecutionResult(
                stdout=stdout.getvalue()
            )
        except Exception as e:
            error_trace_str = traceback.format_exc()
            return ExecutionResult(
                stdout=stdout.getvalue(),
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