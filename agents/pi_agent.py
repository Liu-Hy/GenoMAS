import argparse
from typing import Optional

from core.messaging import Message, MessageType, Role
from utils.llm import LLMClient
from utils.logger import Logger
from .base_agent import BaseAgent


class PIAgent(BaseAgent):
    def __init__(self, client: LLMClient, logger: Logger, role_prompt: Optional[str] = None,
                 guidelines: Optional[str] = None,
                 args: Optional[argparse.Namespace] = None):
        super().__init__(Role.PI, client, logger=logger, role_prompt=role_prompt, guidelines=guidelines, args=args)

    async def act(self, message: Optional[Message] = None) -> Optional[Message]:
        return None

    async def assign_task(self, role: Role) -> Message:
        return Message(
            role=self.role,
            type=MessageType.TASK_REQUEST,
            content="Please start doing the data analysis task.",
            target_roles={role}
        )
