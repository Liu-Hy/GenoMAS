import argparse
from typing import Optional

from core.messaging import Message, MessageType, Role
from utils.llm import LLMClient
from utils.logger import Logger
from .base_agent import BaseAgent


class CodeReviewerAgent(BaseAgent):
    def __init__(self, client: LLMClient, logger: Logger, role_prompt: Optional[str] = None,
                 guidelines: Optional[str] = None,
                 args: Optional[argparse.Namespace] = None):
        super().__init__(Role.CODE_REVIEWER, client, logger, role_prompt=role_prompt, guidelines=guidelines, args=args)

    async def act(self, message: Message) -> Optional[Message]:
        """Handle review requests from a programming agent"""
        if message.type == MessageType.CODE_REVIEW_REQUEST:
            # Review request contains complete prompt
            feedback = await self.ask_llm(message.content)

            return Message(
                role=self.role,
                type=MessageType.CODE_REVIEW_RESPONSE,
                content=feedback,
                target_roles={message.role}
            )
        return None
