import argparse
from typing import Optional

from core.messaging import Message, MessageType, Role
from utils.llm import LLMClient
from utils.logger import Logger
from .base_agent import BaseAgent


class DomainExpertAgent(BaseAgent):
    def __init__(self, client: LLMClient, logger: Logger, role_prompt: Optional[str] = None,
                 guidelines: Optional[str] = None,
                 args: Optional[argparse.Namespace] = None):
        super().__init__(Role.DOMAIN_EXPERT, client, logger, role_prompt=role_prompt, guidelines=guidelines, args=args)

    async def act(self, message: Message) -> Optional[Message]:
        """Handle code writing, review, or revision requests from a programming agent"""
        if MessageType.is_request(message) and message.role in {Role.GEO_AGENT, Role.TCGA_AGENT}:
            response = await self.ask_llm(message.content)

            return Message(
                role=self.role,
                type=MessageType.get_response_type(message),
                content=response,
                target_roles={message.role}
            )
        return None
