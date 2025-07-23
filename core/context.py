from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class StepType(Enum):
    REGULAR = "regular"
    DEBUG = "debug"


@dataclass
class Step:
    type: StepType
    index: int
    action_name: str
    instruction: Optional[str]
    code: str
    stdout: str
    error_trace: Optional[str]


class ActionUnit:
    def __init__(self,
                 name: str,
                 instruction: str,
                 code_snippet: str = ""):
        self.name = name
        self.instruction = instruction
        self.code_snippet = code_snippet
        self.code_snippet_buffer = []

    def __str__(self):
        return f"Name: {self.name}\nInstruction: {self.instruction}"

    @property
    def requires_domain_knowledge(self):
        return self.name in ["Dataset Analysis and Clinical Feature Extraction",
                             "Gene Identifier Review",
                             "Gene Identifier Mapping",
                             "Initial Data Selection and Loading",
                             "Find Candidate Demographic Features",
                             "Select Demographic Features"]

    @property
    def no_history(self):
        return self.name in ["Dataset Analysis and Clinical Feature Extraction",
                             "Gene Identifier Review",
                             "Initial Data Selection and Loading",
                             "Find Candidate Demographic Features",
                             "Select Demographic Features"
                             ]


class TaskContext:
    def __init__(self):
        self.history: List[Step] = []
        self.current_step = 0
        self.debug_step = 0
        self.setup_code = None

    def set_setup_code(self, setup_code: str):
        self.setup_code = setup_code

    def add_step(self, step_type: StepType, **kwargs):
        """Add a new step to history"""

        if step_type == StepType.REGULAR:
            self.current_step += 1
        else:
            self.debug_step += 1

        step = Step(
            type=step_type,
            index=self.current_step if step_type == StepType.REGULAR else self.debug_step,
            **kwargs
        )
        self.history.append(step)

    def get_last_step(self) -> Optional[Step]:
        """Get the most recent step"""
        return self.history[-1] if self.history else None

    def get_regular_step(self, index: int) -> Optional[Step]:
        """Get step by index from regular steps"""
        for step in self.history:
            if step.type == StepType.REGULAR and step.index == index:
                return step
        return None

    def clear_debug_steps(self):
        """Remove all debug steps from history"""
        self.history = [step for step in self.history if step.type == StepType.REGULAR]
        self.debug_step = 0
    
    def preserve_first_n_steps(self, n: int):
        self.clear_debug_steps()
        self.current_step = n
        self.history = self.history[:n]

    def concatenate_snippets(self, end_step: Optional[int] = None, include_setup: bool = False) -> str:
        """Concatenate code snippets up to specified step"""
        if end_step is None:
            end_step = self.current_step

        if include_setup:
            snippets = [self.setup_code]
        else:
            snippets = []
        for step in self.history:
            if step.type == StepType.REGULAR and step.index <= end_step:
                # Prepend each snippet with a header showing step number and action name
                header = f"# Step {step.index}: {step.action_name}"
                formatted_snippet = f"{header}\n{step.code}"
                snippets.append(formatted_snippet)
        return "\n\n".join(snippets)

    def merge_revision(self):
        """Merge the latest debug step into the main step"""
        last_step = self.get_last_step()
        if last_step and last_step.type == StepType.DEBUG:
            main_step = self.get_regular_step(self.current_step)
            for attrib in ['code', 'stdout', 'error_trace']:
                setattr(main_step, attrib, getattr(last_step, attrib))
            self.clear_debug_steps()

    def format_context(self, mode: str = "all", domain_focus: bool = False) -> str:
        """Format the context for prompts.

        Args:
            mode (str): Controls which steps to include in the context:
                - "all": Display all steps
                - "past": Display only previous steps
                - "last": Display only the last step
            domain_focus (bool): If True, only include context starting from the current step

        Returns:
            str: The formatted context string
        """
        assert mode in ["all", "past", "last"]
        start_id = self.current_step - 1 if domain_focus else None

        if mode == "all":
            steps_to_show = self.history[start_id:]
        elif mode == "past":
            steps_to_show = self.history[start_id:-1]
        elif mode == "last":
            steps_to_show = self.history[-1:]

        formatted_context = []
        for step in steps_to_show:
            if step.type == StepType.DEBUG:
                formatted_context.append(f"Debugging Attempt {step.index}")
            else:
                formatted_context.append(f"STEP {step.index}")
                if step.instruction:
                    formatted_context.append(f"[Chosen action unit]: {step.action_name}")
                    formatted_context.append(f"[Instruction]:\n{step.instruction}")
                    if domain_focus:
                        # Add output from previous step for domain expert
                        prev_step = self.get_regular_step(self.current_step - 1)
                        if prev_step:
                            formatted_context.append(prev_step.stdout)

            formatted_context.append(f"[Code]:\n{step.code}")
            formatted_context.append(f"[Output]:\n{step.stdout}")

            if step.error_trace:
                formatted_context.append(f"[Error Trace]:\n{step.error_trace}")

            formatted_context.append(
                "-" * 50 if step.type == StepType.DEBUG else "=" * 50
            )

        return "\n".join(formatted_context)
