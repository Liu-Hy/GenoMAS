import argparse
import re
import time
from typing import Dict, List, Optional

from core.context import ActionUnit, TaskContext
from core.messaging import Role, Message, MessageType
from utils.config import GLOBAL_MAX_TIME
from utils.llm import LLMClient
from utils.logger import Logger


class BaseAgent:
    def __init__(self, role: Role, client: LLMClient, logger: Logger, role_prompt: Optional[str] = None,
                 guidelines: Optional[str] = None, tools: Optional[Dict[str, str]] = None, setups: Optional[str] = None,
                 action_units: Optional[List[ActionUnit]] = None, args: Optional[argparse.Namespace] = None):
        self.role = role
        self.client = client
        self.logger = logger
        if not role_prompt:
            role_prompt = f"You are a {role.value} in this biomedical research project."
        self.role_prompt = role_prompt

        self.guidelines = guidelines
        self.tools = tools
        self.setups = setups
        self.action_units = {unit.name: unit for unit in action_units} if action_units else None
        self.args = args
        if args:
            self.max_time = args.max_time if hasattr(args, 'max_time') else GLOBAL_MAX_TIME
        else:
            self.max_time = GLOBAL_MAX_TIME
        self.start_time = time.time()
        self.memory: List[Message] = []

    async def observe(self, message: Message) -> None:
        self.memory.append(message)

    async def act(self, message: Message) -> Optional[Message]:
        """Execute action based on received message"""
        raise NotImplementedError

    async def ask_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if system_prompt is None:
            system_prompt = self.role_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        start_time = time.time()
        try:
            response = await self.client.generate_completion(messages)
            if self.logger:
                input_tokens = response["usage"].get("input_tokens", 0)
                output_tokens = response["usage"].get("output_tokens", 0)
                cost = response["usage"].get("cost", 0.0)
                self.logger.log_api_call(
                    time.time() - start_time,
                    input_tokens,
                    output_tokens,
                    cost
                )
            return response["content"]
        except Exception as e:
            error_msg = f"Non-transient LLM API Error - {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.logger.log_message(error_msg)
            return ""

    def create_timeout_message(self, context: str) -> Message:
        if hasattr(self, 'task_completed'):
            self.task_completed = True
        return Message(
            role=self.role,
            type=MessageType.TIMEOUT,
            content=f"Execution timeout occurred during {context}.",
            target_roles={Role.PI}
        )

    def get_remaining_time(self) -> float:
        elapsed_time = time.time() - self.start_time
        if not self.max_time:
            remaining_time = GLOBAL_MAX_TIME
        else:
            remaining_time = max(1, self.max_time - elapsed_time)  # Ensure at least 1 second
        return remaining_time

    def clear_states(self):
        self.memory.clear()
        self.start_time = time.time()
        if hasattr(self, 'task_context'):
            self.task_context = TaskContext()
        if hasattr(self, 'executor'):
            self.executor.clear_namespace()
        if hasattr(self, 'task_completed'):
            self.task_completed = False
        if hasattr(self, 'current_action'):
            self.current_action = None
        if hasattr(self, 'review_round'):
            self.review_round = 0

    @staticmethod
    def parse_code(rsp):
        pattern = r"```python(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text.strip()
