import asyncio
import io
import sys
import os
import traceback
import builtins
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
    is_exit: bool = False
    exit_code: Optional[int] = None


class CodeExecutor:
    # Define these as class-level functions
    @staticmethod
    def global_custom_exit(code=0):
        raise SystemExit(code)

    @staticmethod
    def blocked_os_exit(code=0):
        raise RuntimeError("Attempted to call os._exit; call blocked.")

    def __init__(self):
        self.namespace: Dict = {}
        self.setup_code: Optional[str] = None

        # Store original functions for potential cleanup
        self._original_exits = {
            'builtin_exit': builtins.exit,
            'sys_exit': sys.exit,
            'os_exit': os._exit
        }

        # Monkey-patch exit functions
        builtins.exit = self.global_custom_exit
        sys.exit = self.global_custom_exit
        os._exit = self.blocked_os_exit

    def __del__(self):
        # Restore original functions on cleanup
        builtins.exit = self._original_exits['builtin_exit']
        sys.exit = self._original_exits['sys_exit']
        os._exit = self._original_exits['os_exit']

    def custom_exit(self, code=0):
        """Used within the exec namespace to intercept calls like exit()."""
        raise SystemExit(code)

    def set_setup_code(self, setup_code: str):
        self.setup_code = setup_code

    def clear_namespace(self):
        self.namespace.clear()

    async def execute(self, code: str, timeout: float = GLOBAL_MAX_TIME) -> ExecutionResult:
        # Initialize namespace with environment setup if empty
        if (not self.namespace) and self.setup_code:
            exec(self.setup_code, self.namespace)

        stdout = io.StringIO()

        # Add our custom exit to the namespace
        self.namespace['exit'] = self.custom_exit

        try:
            async def exec_with_timeout():
                with redirect_stdout(stdout):
                    exec(code, self.namespace)

            await asyncio.wait_for(exec_with_timeout(), timeout)

            return ExecutionResult(
                stdout=stdout.getvalue()
            )
        except SystemExit as e:
            return ExecutionResult(
                stdout=stdout.getvalue(),
                is_exit=True,
                exit_code=e.code
            )
        except Exception as e:
            error_trace_str = traceback.format_exc()
            return ExecutionResult(
                stdout=stdout.getvalue(),
                error_trace=error_trace_str,
                error=e,
                is_timeout=isinstance(e, asyncio.TimeoutError)
            )

    async def reset_and_execute_to_step(self, task_context: 'TaskContext', step: int, time_out: float) -> Optional[
        ExecutionResult]:
        """Clear namespace and execute all code snippets up to given step"""
        self.clear_namespace()
        code = task_context.concatenate_snippets(end_step=step)

        return await self.execute(code, timeout=time_out)
