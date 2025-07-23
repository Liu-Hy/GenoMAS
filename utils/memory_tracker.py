import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

from core.messaging import Role


@dataclass
class MemoryUsageRecord:
    """Record of a single code snippet memory usage"""
    timestamp: float
    agent_role: str
    action_unit: str
    task_context: str  # Current GTA problem (trait-condition)
    cohort: str  # Dataset cohort being processed
    reuse_type: str  # 'direct_reuse', 'adapted_reuse', 'new_generation'
    time_saved: float  # Estimated time saved through reuse (0 for new generation)
    code_length: int  # Length of code snippet
    modifications: int  # Number of modifications for adapted reuse
    success: bool  # Whether the code executed successfully

    def to_dict(self):
        return asdict(self)


@dataclass
class ActionUnitMemory:
    """Memory statistics for a specific action unit"""
    total_executions: int = 0
    direct_reuses: int = 0
    adapted_reuses: int = 0
    new_generations: int = 0
    total_time_saved: float = 0.0
    avg_generation_time: float = 0.0
    snippets_stored: Set[str] = None  # Track unique snippets

    def __post_init__(self):
        if self.snippets_stored is None:
            self.snippets_stored = set()

    def to_dict(self):
        return {
            "total_executions": self.total_executions,
            "direct_reuses": self.direct_reuses,
            "adapted_reuses": self.adapted_reuses,
            "new_generations": self.new_generations,
            "total_time_saved": self.total_time_saved,
            "avg_generation_time": self.avg_generation_time,
            "unique_snippets": len(self.snippets_stored)
        }


class MemoryTracker:
    """Tracks code snippet memory usage and reuse patterns"""

    def __init__(self, output_dir: str = "./output/memory_tracking"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Current session data
        self.usage_records: List[MemoryUsageRecord] = []
        self.session_start: float = time.time()
        self.current_task: Optional[str] = None
        self.current_cohort: Optional[str] = None

        # Memory statistics by agent and action unit
        self.memory_stats: Dict[str, Dict[str, ActionUnitMemory]] = defaultdict(lambda: defaultdict(ActionUnitMemory))

        # Track generation times for estimating time saved
        self.generation_times: Dict[Tuple[str, str], List[float]] = defaultdict(list)  # (agent, action_unit) -> times

        # Track code snippets for similarity analysis
        self.stored_snippets: Dict[Tuple[str, str], str] = {}  # (agent, action_unit) -> snippet

        # Track pending records that need success status update
        self.pending_record_index: Optional[int] = None

    def set_task_context(self, trait: str, condition: Optional[str] = None):
        """Set the current GTA problem being processed"""
        self.current_task = f"{trait}-{condition}" if condition else trait

    def set_cohort_context(self, cohort: str):
        """Set the current cohort being processed"""
        self.current_cohort = cohort

    def record_memory_usage(self, agent_role: Role, action_unit: str, reuse_type: str,
                            code: str, generation_time: float, success: bool,
                            modifications: int = 0):
        """Record a memory usage event"""
        agent_name = agent_role.value

        # Calculate time saved based on average generation time
        time_saved = 0.0
        if reuse_type in ['direct_reuse', 'adapted_reuse']:
            avg_gen_time = self._get_avg_generation_time(agent_name, action_unit)
            if reuse_type == 'direct_reuse':
                time_saved = avg_gen_time
            else:  # adapted_reuse
                time_saved = avg_gen_time * 0.7  # Assume adaptation saves 70% of generation time

        # Create usage record
        record = MemoryUsageRecord(
            timestamp=time.time(),
            agent_role=agent_name,
            action_unit=action_unit,
            task_context=self.current_task or "unknown",
            cohort=self.current_cohort or "unknown",
            reuse_type=reuse_type,
            time_saved=time_saved,
            code_length=len(code),
            modifications=modifications,
            success=success
        )

        self.usage_records.append(record)

        # Update memory statistics
        stats = self.memory_stats[agent_name][action_unit]
        stats.total_executions += 1

        if reuse_type == 'direct_reuse':
            stats.direct_reuses += 1
        elif reuse_type == 'adapted_reuse':
            stats.adapted_reuses += 1
        else:  # new_generation
            stats.new_generations += 1
            if success:
                self.generation_times[(agent_name, action_unit)].append(generation_time)
                stats.avg_generation_time = self._get_avg_generation_time(agent_name, action_unit)

        stats.total_time_saved += time_saved

        # Store snippet if successful and new
        if success and reuse_type == 'new_generation':
            self.stored_snippets[(agent_name, action_unit)] = code
            stats.snippets_stored.add(hash(code))

        # Return the index of this record
        return len(self.usage_records) - 1

    def update_record_success(self, record_index: int, success: bool, code: str,
                              generation_time: Optional[float] = None):
        """Update the success status of a previously recorded event"""
        if 0 <= record_index < len(self.usage_records):
            record = self.usage_records[record_index]
            agent_name = record.agent_role
            action_unit = record.action_unit

            # Update success status
            record.success = success

            # Update code length (in case it changed during execution)
            record.code_length = len(code)

            # Update statistics if needed
            stats = self.memory_stats[agent_name][action_unit]

            # If this was a new generation that succeeded, update timing data and store snippet
            if success and record.reuse_type == 'new_generation':
                if generation_time is not None:
                    self.generation_times[(agent_name, action_unit)].append(generation_time)
                    stats.avg_generation_time = self._get_avg_generation_time(agent_name, action_unit)

                # Store the snippet for future similarity checks
                self.stored_snippets[(agent_name, action_unit)] = code
                stats.snippets_stored.add(hash(code))

            # Recalculate time saved if needed
            if record.reuse_type in ['direct_reuse', 'adapted_reuse']:
                avg_gen_time = self._get_avg_generation_time(agent_name, action_unit)
                if record.reuse_type == 'direct_reuse':
                    record.time_saved = avg_gen_time
                else:  # adapted_reuse
                    record.time_saved = avg_gen_time * 0.7

                # Update total time saved
                stats.total_time_saved = sum(r.time_saved for r in self.usage_records
                                             if r.agent_role == agent_name and r.action_unit == action_unit)

    def create_pending_record(self, agent_role: Role, action_unit: str, reuse_type: str,
                              code: str, modifications: int = 0) -> int:
        """Create a pending record that will be updated after execution"""
        # Create record with temporary values
        record_index = self.record_memory_usage(
            agent_role=agent_role,
            action_unit=action_unit,
            reuse_type=reuse_type,
            code=code,
            generation_time=0.0,  # Will be updated later
            success=False,  # Will be updated later
            modifications=modifications
        )
        self.pending_record_index = record_index
        return record_index

    def get_stored_snippet(self, agent: str, action_unit: str) -> Optional[str]:
        """Get previously stored snippet for an agent/action unit pair"""
        return self.stored_snippets.get((agent, action_unit))

    def _get_avg_generation_time(self, agent: str, action_unit: str) -> float:
        """Get average generation time for an action unit"""
        times = self.generation_times.get((agent, action_unit), [])
        return sum(times) / len(times) if times else 30.0  # Default 30s if no data

    def calculate_code_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code snippets (0-1 scale)"""
        # Simple line-based similarity for now
        lines1 = set(line.strip() for line in code1.splitlines() if line.strip())
        lines2 = set(line.strip() for line in code2.splitlines() if line.strip())

        if not lines1 or not lines2:
            return 0.0

        intersection = len(lines1 & lines2)
        union = len(lines1 | lines2)

        return intersection / union if union > 0 else 0.0

    def get_reuse_efficiency_over_time(self) -> List[Dict]:
        """Get cumulative time savings over problem execution"""
        cumulative_data = []
        cumulative_saved = 0.0
        cumulative_total = 0.0

        for i, record in enumerate(self.usage_records):
            cumulative_saved += record.time_saved
            if record.reuse_type == 'new_generation':
                cumulative_total += self._get_avg_generation_time(record.agent_role, record.action_unit)
            else:
                cumulative_total += record.time_saved

            cumulative_data.append({
                "problem_index": i,
                "timestamp": record.timestamp - self.session_start,
                "task": record.task_context,
                "cumulative_time_saved": cumulative_saved,
                "cumulative_time_total": cumulative_total,
                "efficiency_ratio": cumulative_saved / cumulative_total if cumulative_total > 0 else 0,
                "reuse_type": record.reuse_type,
                "agent": record.agent_role,
                "action_unit": record.action_unit
            })

        return cumulative_data

    def get_reuse_patterns_by_action_unit(self) -> Dict[str, Dict]:
        """Get reuse patterns grouped by action unit"""
        patterns = {}

        for agent, action_units in self.memory_stats.items():
            for action_unit, stats in action_units.items():
                key = f"{agent}_{action_unit}"
                patterns[key] = {
                    "agent": agent,
                    "action_unit": action_unit,
                    "total_executions": stats.total_executions,
                    "direct_reuse_rate": stats.direct_reuses / stats.total_executions if stats.total_executions > 0 else 0,
                    "adapted_reuse_rate": stats.adapted_reuses / stats.total_executions if stats.total_executions > 0 else 0,
                    "new_generation_rate": stats.new_generations / stats.total_executions if stats.total_executions > 0 else 0,
                    "avg_time_saved_per_execution": stats.total_time_saved / stats.total_executions if stats.total_executions > 0 else 0,
                    "unique_snippets": len(stats.snippets_stored)
                }

        return patterns

    def save_session_data(self, session_id: str):
        """Save all memory tracking data for the current session"""
        session_data = {
            "session_id": session_id,
            "start_time": self.session_start,
            "end_time": time.time(),
            "duration": time.time() - self.session_start,
            "usage_records": [r.to_dict() for r in self.usage_records],
            "statistics": {
                "total_memory_uses": len(self.usage_records),
                "total_direct_reuses": sum(1 for r in self.usage_records if r.reuse_type == 'direct_reuse'),
                "total_adapted_reuses": sum(1 for r in self.usage_records if r.reuse_type == 'adapted_reuse'),
                "total_new_generations": sum(1 for r in self.usage_records if r.reuse_type == 'new_generation'),
                "total_time_saved": sum(r.time_saved for r in self.usage_records),
                "memory_stats_by_agent": {
                    agent: {au: stats.to_dict() for au, stats in units.items()}
                    for agent, units in self.memory_stats.items()
                },
                "reuse_efficiency_timeline": self.get_reuse_efficiency_over_time(),
                "reuse_patterns": self.get_reuse_patterns_by_action_unit()
            }
        }

        # Save to file
        filename = f"memory_tracking_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)

        return filepath