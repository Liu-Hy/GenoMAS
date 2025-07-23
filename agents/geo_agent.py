import argparse
from typing import Dict, List, Optional

from core.context import ActionUnit
from core.messaging import Role
from utils.llm import LLMClient
from utils.logger import Logger
from .multi_step_agent import MultiStepProgrammingAgent


class GEOAgent(MultiStepProgrammingAgent):
    def __init__(self, client: LLMClient, logger: Logger, role_prompt: str, guidelines: str, tools: Dict[str, str],
                 setups: str,
                 action_units: List[ActionUnit], args: argparse.Namespace,
                 planning_client: Optional[LLMClient] = None):
        super().__init__(Role.GEO_AGENT, client, logger, role_prompt=role_prompt, guidelines=guidelines, tools=tools,
                         setups=setups, action_units=action_units, args=args, planning_client=planning_client)
