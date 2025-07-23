import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from core.messaging import Message, MessageType, Role


@dataclass
class InteractionRecord:
    """Record of a single interaction between agents"""
    timestamp: float
    sender: str
    receiver: str
    message_type: str
    task_context: str  # Current GTA problem (trait-condition)
    step_context: str  # Current action unit if applicable
    duration: float = 0.0  # Time until response received

    def to_dict(self):
        return asdict(self)


class CollaborationTracker:
    """Tracks agent interactions and collaboration patterns"""

    def __init__(self, output_dir: str = "./output/collaboration_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Current session data
        self.interactions: List[InteractionRecord] = []
        self.pending_requests: Dict[Tuple[str, str, str], float] = {}  # (sender, receiver, type) -> timestamp
        self.current_task: Optional[str] = None
        self.current_step: Optional[str] = None
        self.session_start: float = time.time()

        # Aggregated statistics
        self.interaction_counts: Dict[Tuple[str, str], int] = defaultdict(int)  # (sender, receiver) -> count
        self.message_type_counts: Dict[str, int] = defaultdict(int)
        self.agent_activity: Dict[str, int] = defaultdict(int)  # agent -> total messages
        self.task_interactions: Dict[str, List[InteractionRecord]] = defaultdict(list)

    def set_task_context(self, trait: str, condition: Optional[str] = None):
        """Set the current GTA problem being processed"""
        self.current_task = f"{trait}-{condition}" if condition else trait

    def set_step_context(self, step: Optional[str]):
        """Set the current action unit being executed"""
        self.current_step = step

    def record_message(self, message: Message, receivers: List[Role]):
        """Record a message being sent from one agent to others"""
        sender = message.role.value
        message_type = message.type.value
        timestamp = time.time()

        for receiver_role in receivers:
            receiver = receiver_role.value

            # Create interaction record
            interaction = InteractionRecord(
                timestamp=timestamp,
                sender=sender,
                receiver=receiver,
                message_type=message_type,
                task_context=self.current_task or "unknown",
                step_context=self.current_step or "unknown"
            )

            # If this is a request, store it to calculate duration later
            if MessageType.is_request(message):
                key = (sender, receiver, message_type)
                self.pending_requests[key] = timestamp

            # If this is a response, calculate duration from corresponding request
            elif MessageType.is_response(message):
                # Find corresponding request type
                request_type = message_type.replace('_response', '_request')
                key = (receiver, sender, request_type)  # Note: reversed for response
                if key in self.pending_requests:
                    request_time = self.pending_requests.pop(key)
                    interaction.duration = timestamp - request_time

            # Store interaction
            self.interactions.append(interaction)

            # Update statistics
            self.interaction_counts[(sender, receiver)] += 1
            self.message_type_counts[message_type] += 1
            self.agent_activity[sender] += 1
            if self.current_task:
                self.task_interactions[self.current_task].append(interaction)

    def save_session_data(self, session_id: str):
        """Save all interaction data for the current session"""
        session_data = {
            "session_id": session_id,
            "start_time": self.session_start,
            "end_time": time.time(),
            "duration": time.time() - self.session_start,
            "interactions": [i.to_dict() for i in self.interactions],
            "statistics": {
                "total_interactions": len(self.interactions),
                "interaction_counts": {f"{k[0]}->{k[1]}": v for k, v in self.interaction_counts.items()},
                "message_type_counts": dict(self.message_type_counts),
                "agent_activity": dict(self.agent_activity),
                "tasks_processed": list(self.task_interactions.keys())
            }
        }

        # Save to file
        filename = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)

        return filepath

    def get_interaction_matrix(self) -> Dict[str, Dict[str, int]]:
        """Get interaction counts as a matrix for visualization"""
        agents = set()
        for (sender, receiver) in self.interaction_counts:
            agents.add(sender)
            agents.add(receiver)

        matrix = {agent: {other: 0 for other in agents} for agent in agents}
        for (sender, receiver), count in self.interaction_counts.items():
            matrix[sender][receiver] = count

        return matrix

    def get_temporal_patterns(self) -> List[Dict]:
        """Get temporal interaction patterns for timeline visualization"""
        patterns = []
        for interaction in self.interactions:
            patterns.append({
                "time": interaction.timestamp - self.session_start,
                "sender": interaction.sender,
                "receiver": interaction.receiver,
                "type": interaction.message_type,
                "task": interaction.task_context,
                "duration": interaction.duration
            })
        return patterns