from typing import List, Dict, Optional, Set
import asyncio
from dataclasses import dataclass
from enum import Enum
import json


class Role(Enum):
    PRODUCT_MANAGER = "product_manager"
    ARCHITECT = "architect"
    PROJECT_MANAGER = "project_manager"
    ENGINEER = "engineer"
    REVIEWER = "reviewer"


@dataclass
class Message:
    role: Role
    content: str
    cause_by: Optional[str] = None
    target_roles: Optional[Set[Role]] = None  # Specific roles that should receive this message


class TeamTopology:
    def __init__(self):
        self.communication_paths: Dict[Role, Set[Role]] = {role: set() for role in Role}

    def add_communication_path(self, from_role: Role, to_role: Role, bidirectional: bool = True):
        """Define who can communicate with whom"""
        self.communication_paths[from_role].add(to_role)
        if bidirectional:
            self.communication_paths[to_role].add(from_role)

    def can_communicate(self, from_role: Role, to_role: Role) -> bool:
        """Check if one role can send messages to another"""
        return to_role in self.communication_paths[from_role]

    def get_receivers(self, sender: Role) -> Set[Role]:
        """Get all roles that can receive messages from the sender"""
        return self.communication_paths[sender]


class Action:
    def __init__(self, name: str, instruction: str, target_roles: Optional[Set[Role]] = None):
        self.name = name
        self.instruction = instruction
        self.target_roles = target_roles

    async def run(self, context: Dict) -> str:
        return f"Simulated response for {self.name}: {self.instruction}"


class Agent:
    def __init__(self, role: Role, actions: List[Action]):
        self.role = role
        self.actions = {action.name: action for action in actions}
        self.memory: List[Message] = []

    async def observe(self, message: Message) -> None:
        self.memory.append(message)

    async def think(self) -> Optional[Action]:
        if not self.memory:
            return None
        latest_message = self.memory[-1]

        # Role-specific logic
        if self.role == Role.PRODUCT_MANAGER:
            return self.actions.get("write_prd")
        elif self.role == Role.ARCHITECT:
            if latest_message.role == Role.PRODUCT_MANAGER:
                return self.actions.get("design_system")
        elif self.role == Role.ENGINEER:
            if latest_message.role == Role.ARCHITECT:
                return self.actions.get("implement")
        return None

    async def act(self, action: Action, context: Dict) -> Message:
        result = await action.run(context)
        return Message(self.role, result, target_roles=action.target_roles)


class Environment:
    def __init__(self, topology: TeamTopology):
        self.agents: Dict[Role, Agent] = {}
        self.message_queue: List[Message] = []
        self.context: Dict = {}
        self.topology = topology

    def add_agent(self, agent: Agent):
        self.agents[agent.role] = agent

    async def run(self, initial_message: str, max_rounds: int = 5):
        self.message_queue.append(Message(Role.PRODUCT_MANAGER, initial_message))

        for round_num in range(max_rounds):
            if not self.message_queue:
                break

            current_message = self.message_queue.pop(0)
            print(f"\nRound {round_num + 1}:")
            print(f"Message from {current_message.role.value}: {current_message.content}")

            # Determine message recipients based on topology and target_roles
            valid_receivers = self.topology.get_receivers(current_message.role)
            if current_message.target_roles:
                valid_receivers = valid_receivers.intersection(current_message.target_roles)

            # Broadcast message to valid receivers only
            for role, agent in self.agents.items():
                if role in valid_receivers:
                    print(f"- {role.value} received the message")
                    await agent.observe(current_message)

                    # Let agent think and act
                    action = await agent.think()
                    if action:
                        new_message = await agent.act(action, self.context)
                        self.message_queue.append(new_message)


async def main():
    # Define team topology
    topology = TeamTopology()

    # Define communication paths
    topology.add_communication_path(Role.PRODUCT_MANAGER, Role.ARCHITECT, bidirectional=False)
    topology.add_communication_path(Role.ARCHITECT, Role.ENGINEER, bidirectional=False)
    topology.add_communication_path(Role.ENGINEER, Role.REVIEWER, bidirectional=True)
    topology.add_communication_path(Role.REVIEWER, Role.PRODUCT_MANAGER, bidirectional=True)

    # Create actions with specific target roles
    pm_actions = [Action("write_prd", "Write product requirements", {Role.ARCHITECT})]
    architect_actions = [Action("design_system", "Design system architecture", {Role.ENGINEER})]
    engineer_actions = [Action("implement", "Implement the solution", {Role.REVIEWER})]
    reviewer_actions = [Action("review", "Review the implementation", {Role.PRODUCT_MANAGER, Role.ENGINEER})]

    # Create agents
    pm = Agent(Role.PRODUCT_MANAGER, pm_actions)
    architect = Agent(Role.ARCHITECT, architect_actions)
    engineer = Agent(Role.ENGINEER, engineer_actions)
    reviewer = Agent(Role.REVIEWER, reviewer_actions)

    # Set up environment with topology
    env = Environment(topology)
    env.add_agent(pm)
    env.add_agent(architect)
    env.add_agent(engineer)
    env.add_agent(reviewer)

    # Run the simulation
    await env.run("Create a todo list application")


if __name__ == "__main__":
    asyncio.run(main())