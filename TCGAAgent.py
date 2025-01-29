import argparse
import asyncio
import io
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, List, Dict, Set, Tuple

from prompts import *
from tools.statistics import normalize_trait
from utils.llm import LLMClient, get_llm_client
from utils.path_config import TCGAPathConfig
from utils.utils import extract_function_code

GLOBAL_MAX_TIME = 600.0  # 1000000.0


class Role(Enum):
    PI = "PI"
    GEO_AGENT = "GEO Agent"
    TCGA_AGENT = "TCGA Agent"
    STATISTICIAN_AGENT = "Statistician Agent"
    CODE_REVIEWER = "Code Reviewer"
    DOMAIN_EXPERT = "Domain Expert"


class MessageType(Enum):
    TASK_REQUEST = "task_request"
    CODE_WRITING_REQUEST = "code_writing_request"
    CODE_REVIEW_REQUEST = "code_review_request"
    CODE_REVISION_REQUEST = "code_revision_request"
    PLANNING_REQUEST = "planning_request"

    TASK_RESPONSE = "task_response"
    CODE_WRITING_RESPONSE = "code_writing_response"
    CODE_REVIEW_RESPONSE = "code_review_response"
    CODE_REVISION_RESPONSE = "code_revision_response"
    PLANNING_RESPONSE = "planning_response"

    # System
    TIMEOUT = "timeout"

    @classmethod
    def get_type(cls, message: Union['MessageType', 'Message']) -> 'MessageType':
        if isinstance(message, MessageType):
            return message
        elif isinstance(message, Message):
            return message.type
        else:
            raise ValueError(f"Invalid input type: {type(message)}")

    @classmethod
    def is_request(cls, message: Union['MessageType', 'Message']) -> bool:
        message_type = cls.get_type(message)
        return message_type.value.endswith('_request')

    @classmethod
    def is_response(cls, message: Union['MessageType', 'Message']) -> bool:
        message_type = cls.get_type(message)
        return message_type.value.endswith('_response')

    @classmethod
    def is_system(cls, message: Union['MessageType', 'Message']) -> bool:
        return not (cls.is_request(message) or cls.is_response(message))

    @classmethod
    def get_response_type(cls, request: Union['MessageType', 'Message']) -> Optional['MessageType']:
        request_type = cls.get_type(request)
        if not cls.is_request(request_type):
            return None

        response_name = request_type.value.replace('_request', '_response')
        return next((t for t in cls if t.value == response_name), None)


@dataclass
class Message:
    role: Role
    type: MessageType
    content: str
    target_roles: Union[Role, List[Role], Tuple[Role, ...], Set[Role]]

    def __post_init__(self):
        # Convert target_roles to Set[Role] regardless of input type
        if isinstance(self.target_roles, Role):
            self.target_roles = {self.target_roles}
        elif isinstance(self.target_roles, (list, tuple)):
            self.target_roles = set(self.target_roles)
        elif not isinstance(self.target_roles, set):
            raise ValueError(
                f"target_roles must be Role, List[Role], Tuple[Role], or Set[Role], got {type(self.target_roles)}")


CODE_INDUCER = """
NOTE: 
ONLY IMPLEMENT CODE FOR THE CURRENT STEP. MAKE SURE THE CODE CAN BE CONCATENATED WITH THE CODE FROM PREVIOUS STEPS AND CORRECTLY EXECUTED.

FORMAT:
```python
[your_code]
```

NO text outside the code block.
Your code:
"""


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
                             "Gene Identifier Review", "Gene Identifier Mapping"]

    @property
    def no_history(self):
        return self.name in ["Dataset Analysis and Clinical Feature Extraction",
                             "Gene Identifier Review"]


class Logger(logging.Logger):
    def __init__(self, log_file: Optional[str] = None, name: Optional[str] = "Script", max_msg_length: int = 5000):
        super().__init__(name=name)
        self.start_time = time.time()
        self.api_calls = []
        self.total_cost = 0.0  # Track cumulative cost
        self.cost_tracking_enabled = True  # Flag to indicate if cost tracking is possible
        self.max_msg_length = max_msg_length  # Set this before any logging calls

        # Add timestamp to log file name if provided
        if log_file:
            base, ext = os.path.splitext(log_file)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_file = f"{base}_{timestamp}{ext}"
        else:
            self.log_file = None

        # Set up handler based on whether log_file is provided
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if self.log_file:
            handler = logging.FileHandler(self.log_file, mode='w')
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.addHandler(handler)

        # Initialize the log with a header
        self.info("=== Log started ===")
        self.max_msg_length = max_msg_length

    def _truncate_message(self, message: str) -> str:
        """Truncate message if it exceeds max_msg_length"""
        msg = str(message)
        if len(msg) > self.max_msg_length:
            return msg[:self.max_msg_length] + "... [truncated]"
        return msg

    def debug(self, msg, *args, **kwargs):
        super().debug(self._truncate_message(msg), *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        super().info(self._truncate_message(msg), *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        super().warning(self._truncate_message(msg), *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        super().error(self._truncate_message(msg), *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        super().critical(self._truncate_message(msg), *args, **kwargs)

    def log_api_call(self, duration: float, input_tokens: int, output_tokens: int, cost: float):
        call_data = {
            'timestamp': time.time(),
            'duration': duration,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost
        }
        self.api_calls.append(call_data)

        # If we get a negative cost, disable cost tracking for the entire session
        if cost < 0:
            self.cost_tracking_enabled = False
            self.total_cost = -1.0
        elif self.cost_tracking_enabled:
            self.total_cost += cost

        # Log API call
        base_log = f"API Call - Duration: {call_data['duration']:.2f}s, " \
                   f"Input Tokens: {call_data['input_tokens']}, Output Tokens: {call_data['output_tokens']}"

        if self.cost_tracking_enabled:
            base_log += f", Cost: ${cost:.2f}, Cumulative Cost: ${self.total_cost:.2f}"

        self.info(base_log)

    def save(self):
        """Write final summary statistics to the log file"""
        end_time = time.time()
        total_duration = end_time - self.start_time
        total_api_duration = sum(call['duration'] for call in self.api_calls)
        total_input_tokens = sum(call['input_tokens'] for call in self.api_calls)
        total_output_tokens = sum(call['output_tokens'] for call in self.api_calls)

        self.info("\n=== Summary Statistics ===")
        self.info(f"Total Duration: {total_duration:.2f} seconds")
        self.info(f"Total API Call Duration: {total_api_duration:.2f} seconds")
        self.info(f"Total Input Tokens: {total_input_tokens}")
        self.info(f"Total Output Tokens: {total_output_tokens}")
        if self.cost_tracking_enabled:
            self.info(f"Total API Cost: ${self.total_cost:.2f}")
        self.info("=== Log ended ===")


@dataclass
class ExecutionResult:
    stdout: str
    error_trace: Optional[str] = None
    error: Optional[Exception] = None
    is_timeout: bool = False


class CodeExecutor:
    def __init__(self):
        self.namespace: Dict = {}
        self.setup_code: Optional[str] = None

    def set_setup_code(self, setup_code: str):
        self.setup_code = setup_code

    def clear_namespace(self):
        self.namespace.clear()

    async def execute(self, code: str, timeout: float = GLOBAL_MAX_TIME) -> ExecutionResult:
        # Initialize namespace with environment setup if empty
        if (not self.namespace) and self.setup_code:
            exec(self.setup_code, self.namespace)

        stdout = io.StringIO()

        try:
            async def exec_with_timeout():
                with redirect_stdout(stdout):
                    exec(code, self.namespace)

            await asyncio.wait_for(exec_with_timeout(), timeout)

            return ExecutionResult(
                stdout=stdout.getvalue()
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


class TaskContext:
    def __init__(self):
        self.history: List[Step] = []
        self.current_step = 0
        self.debug_step = 0

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

    def concatenate_snippets(self, end_step: Optional[int] = None) -> str:
        """Concatenate code snippets up to specified step"""
        if end_step is None:
            end_step = self.current_step

        snippets = []
        for step in self.history:
            if step.type == StepType.REGULAR and step.index <= end_step:
                snippets.append(step.code)
        return "\n".join(snippets)

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


class BaseAgent:
    def __init__(self, role: Role, client: LLMClient, logger: Logger, role_prompt: Optional[str] = None,
                 guidelines: Optional[str] = None, tools: Optional[str] = None, setups: Optional[str] = None,
                 action_units: Optional[List[ActionUnit]] = None, args: Optional[argparse.Namespace] = None):
        self.role = role
        self.client = client
        self.logger = logger
        if not role_prompt:
            role_prompt = f"You are a {role.value} in this project."
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
            self.logger.error(error_msg)
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


class PIAgent(BaseAgent):
    def __init__(self, client: LLMClient, logger: Logger, role_prompt: Optional[str] = None,
                 guidelines: Optional[str] = None,
                 args: Optional[argparse.Namespace] = None):
        super().__init__(Role.PI, client, logger=logger, role_prompt=role_prompt, guidelines=guidelines, args=args)

    async def act(self, message: Optional[Message] = None) -> Optional[Message]:
        if message is None:
            return Message(
                role=self.role,
                type=MessageType.TASK_REQUEST,
                content="Process this trait.",
                target_roles={Role.TCGA_AGENT}
            )
        return None


class TCGAAgent(BaseAgent):
    def __init__(self, client: LLMClient, logger: Logger, role_prompt: str, guidelines: str, tools: str, setups: str,
                 action_units: List[ActionUnit], args: argparse.Namespace):
        super().__init__(Role.TCGA_AGENT, client, logger, role_prompt=role_prompt, guidelines=guidelines, tools=tools,
                         setups=setups, action_units=action_units, args=args)
        self.max_rounds = args.max_rounds
        self.review_round = 0
        self.task_context = TaskContext()
        self.executor = CodeExecutor()
        self.current_action = None
        self.task_completed = False
        self.include_domain_expert = args.de
        self.use_code_snippet = args.cs
        self.enable_planning = args.plan
        self.max_retractions = args.max_retract
        self.remaining_retractions = args.max_retract

    def update_path_config(self, path_config: TCGAPathConfig):
        """Update path configuration and related prompts for a new cohort."""
        self.setups = MULTI_STEP_SETUPS.format(path_setup=path_config.get_setup_prompt())
        self.executor.set_setup_code(path_config.get_setup_code())

    def prepare_prompt(self, mode="all", domain_focus=False):
        assert mode in ["all", "past", "last"], "Unsupported mode: must be one of 'all', 'past', 'last'."
        formatted_prompt = []
        if (not domain_focus) and (mode != "last"):
            formatted_prompt.append(
                "To help you prepare, I will provide you with the following: the task guidelines, "
                "the function tools, the programming setups, and the history of previous steps "
                "taken, including the instructions, code, and execution output of each step.")
            formatted_prompt.append(f"**General Guidelines**: \n{self.guidelines}\n")
            formatted_prompt.append(f"**Function Tools**: \n{self.tools['full']}\n")
            formatted_prompt.append(f"**Programming Setups**: \n{self.setups}\n")
            formatted_prompt.append(f"**Task History**: \n{self.task_context.format_context(mode, domain_focus)}")
        else:
            if mode != "last":
                formatted_prompt.append(f"**Function Tools**: \n{self.tools['domain_focus']}\n")
                formatted_prompt.append(f"**Programming Setups**: \n{self.setups}\n")
            formatted_prompt.append(self.task_context.format_context(mode, domain_focus))

        return "\n".join(formatted_prompt)

    def prepare_code_writing_prompt(self) -> str:
        """Prepare prompt for code writing"""
        if self.current_action.no_history and self.include_domain_expert:
            # For steps 2 and 4
            last_step = self.task_context.get_last_step()
            prev_output = last_step.stdout if last_step else ""
            prompt = (
                f"**Function Tools**: \n{self.tools['domain_focus']}\n"
                f"**Programming Setups**: \n{self.setups}\n"
                f"**Task**: \n{str(self.current_action)}\n\n{prev_output}\n\n{CODE_INDUCER}"
            )
        else:
            prompt = self.prepare_prompt()
            prompt += f"\n**TO DO: Programming** \nNow that you've been familiar with the task setups and current status" \
                      f", please write the code following the instructions:\n\n{str(self.current_action)}\n{CODE_INDUCER}"
        return prompt

    def prepare_code_review_prompt(self) -> str:
        """Prepare prompt for code review"""
        formatted_prompt = []
        domain_focus = self.current_action.no_history and self.include_domain_expert
        formatted_prompt.append(self.prepare_prompt(mode="past", domain_focus=domain_focus))

        formatted_prompt.append("\n**TO DO: Code Review**\n"
                                "The following code is the latest attempt for the current step and requires your review. "
                                "If previous attempts have been included in the task history above, their presence"
                                " does not indicate they succeeded or failed, though you can refer to their execution "
                                "outputs for context. \nOnly review the latest code attempt provided below.\n")

        formatted_prompt.append(self.prepare_prompt(mode="last", domain_focus=domain_focus))

        if self.current_action.requires_domain_knowledge and self.include_domain_expert:
            domain_trigger = ("Some tasks may involve understanding the data and making inferences based on biomedical "
                              "knowledge. These inferences might require assumptions, which do not need to be fully "
                              "validated, though they need to be reasonable. ")
        else:
            domain_trigger = ""

        formatted_prompt.append(f"\nPlease review the code according to the following criteria:\n"
                                "1. *Functionality*: Can the code be successfully executed in the current setting?\n"
                                f"2. *Conformance*: Does the code conform to the given instructions? {domain_trigger}\n"
                                "Provide suggestions for revision and improvement if necessary.\n"
                                "*NOTE*:\n"
                                "1. Your review is not concerned with engineering code quality. The code is a quick "
                                "demo for a research project, so the standards should not be strict.\n"
                                "2. If you provide suggestions, please limit them to 1 to 3 key suggestions. Focus on "
                                "the most important aspects, such as how to solve the execution errors or make the code "
                                "conform to the instructions.\n\n"
                                "State your decision exactly in the format: \"Final Decision: Approved\" or "
                                "\"Final Decision: Rejected.\"")

        return "\n".join(formatted_prompt)

    def prepare_code_revision_prompt(self, feedback: str) -> str:
        """Prepare prompt for code revision"""
        domain_focus = self.current_action.no_history and self.include_domain_expert

        prompt = []
        prompt.append(self.prepare_prompt(mode="past", domain_focus=domain_focus))
        prompt.append(
            f"\nThe following code is the latest attempt for the current step and requires correction. "
            "If previous attempts have been included in the task history above, their presence"
            " does not indicate they succeeded or failed, though you can refer to their execution "
            "outputs for context. \nOnly correct the latest code attempt provided below.\n")
        prompt.append(self.prepare_prompt(mode="last", domain_focus=domain_focus))
        prompt.append(f"Use the reviewer's feedback to help debug and identify logical errors in the code. "
                      f"While the feedback is generally reliable, it might occasionally include errors or "
                      f"suggest changes that are impractical in the current context. Make revisions where "
                      f"you agree with the feedback, but retain the original code where you do not.\n")
        prompt.append(f"\nReviewer's feedback:\n{feedback}\n")
        prompt.append(CODE_INDUCER)

        return "\n".join(prompt)

    def prepare_planning_prompt(self) -> str:
        """Prepare prompt for planning the next step"""
        workflow_context = [
            f"You are a {self.role} working on a multi-step workflow for exploring gene expression datasets.",
            "Each step involves performing an Action Unit by writing code following specific instructions.",
            "The execution results of each step often serve as input for subsequent steps.",
            "Your task is to analyze the current context and choose the most appropriate Action Unit for the next step.",
            "",
        ]

        if self.remaining_retractions > 0:
            workflow_context.extend([
                "If you discover a critical error in choosing an Action Unit for a previous step that significantly impacts",
                "the workflow, you may choose to retract to that step. Retraction will remove all execution results from",
                "that step onward, allowing you to choose a different Action Unit and proceed again from there.",
                "However, retraction is an expensive operation and should be used sparingly.",
                "",
            ])

        workflow_context.append(f"High-level Guidelines for this Task:\n{self.guidelines}\n")

        action_units_formatted = "\n".join([f"- {str(unit)}" for unit in self.action_units.values()])

        task_context = [
            "Current Context:",
            f"Current step: {self.task_context.current_step}",
            "Task History:",
            self.task_context.format_context(mode="all", domain_focus=False)
        ]

        planning_instructions = [
            "\nPlanning Instructions:",
            "1. Review the task guidelines and history carefully",
            "2. Choose ONE action unit for the next step"
        ]

        if self.remaining_retractions > 0:
            planning_instructions.extend([
                f"3. You have {self.remaining_retractions} retraction opportunities remaining. Use them only for correcting critical mistakes in choosing action units.",
                "4. You may only retract to regular steps (steps not labelled as \"Debugging Attempt\")"
            ])

        planning_instructions.extend([
            "",
            "Available Action Units:",
            action_units_formatted,
            "",
            "Your response should include the exact name of the action unit you choose, one of the below:",
            "[" + ", ".join(f'"{name}"' for name in self.action_units.keys()) + "]",
            "",
            "Your response must strictly follow the JSON format below with no extra text:",
            "Response Format:",
            "{\n  \"action\": \"<action_unit_name>\","
        ])

        if self.remaining_retractions > 0:
            planning_instructions.extend([
                "  \"retract\": <true/false>,",
                "  \"retract_to_step\": <step_number or null>,"
            ])

        planning_instructions.extend([
            "  \"reasoning\": \"<brief explanation of your decision>\"\n}",
            "",
            "Your response:"
        ])

        return "\n".join(workflow_context + task_context + planning_instructions)

    def get_default_planning_response(self) -> str:
        """Generate default planning response that follows predefined order"""
        action_names = list(self.action_units.keys())
        next_action = action_names[self.task_context.current_step]
        return json.dumps({
            "action": next_action,
            "retract": False,
            "retract_to_step": None,
            "reasoning": "Following predefined action unit order"
        })

    async def retract_to_step(self, step: int) -> None:
        """Retract the task context and executor namespace to a previous step
        
        Args:
            step: The step number to retract to. After retraction, the task will continue
                 from this step (i.e., this step will be redone with a new action unit).
                 The history will be truncated to contain only steps 0 to step-1.
        """
        if step < 0 or step >= self.task_context.current_step:
            raise ValueError(f"Invalid step number for retraction: {step}")

        # Get the regular step to validate it's not a debug step
        try:
            self.task_context.get_regular_step(step)
        except ValueError as e:
            raise ValueError(f"Cannot retract to debug step: {step}")

        # Truncate history to keep only steps 1 to step-1
        # The next step (step N) will be added with the new action unit
        self.task_context.clear_debug_steps()
        self.task_context.current_step = step - 1
        self.task_context.history = self.task_context.history[:step - 1]

        # Reset executor namespace and re-execute code up to step-1
        prev_result = await self.executor.reset_and_execute_to_step(
            self.task_context,
            step - 1,
            self.get_remaining_time()
        )
        if prev_result and prev_result.is_timeout:
            return self.create_timeout_message("retraction recovery")

        # Decrement remaining retractions
        self.remaining_retractions -= 1

    async def start_next_step(self) -> Optional[Message]:
        """Handle progression to next step"""
        self.task_context.merge_revision()
        self.review_round = 0

        return Message(
            role=self.role,
            type=MessageType.PLANNING_REQUEST,
            content=self.prepare_planning_prompt(),
            target_roles={self.role}
        )

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings.
        Returns a score between 0 and 1, where 1 means exact match (ignoring case).
        """
        # Convert to lowercase and remove spaces for comparison
        s1 = str1.lower().replace(" ", "")
        s2 = str2.lower().replace(" ", "")

        # Exact match
        if s1 == s2:
            return 1.0

        # One string contains the other
        if s1 in s2 or s2 in s1:
            return 0.9

        # Count matching characters
        matches = sum(1 for c1, c2 in zip(s1, s2) if c1 == c2)
        max_len = max(len(s1), len(s2))

        return matches / max_len if max_len > 0 else 0.0

    def parse_planning_response(self, content: str) -> Tuple[str, bool, Optional[int], str]:
        """Parse the planning response from the agent"""
        try:
            content = content.strip()

            # Find the first '{' and last '}' to extract just the JSON object
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                content = content[start:end + 1]

            response = json.loads(content)

            if "action" not in response:
                raise KeyError("Missing required field: 'action'")

            proposed_action = response["action"]

            # Try fuzzy matching if exact match fails
            if proposed_action not in self.action_units:
                best_match = None
                best_score = 0
                for action_name in self.action_units.keys():
                    score = self._calculate_similarity(proposed_action, action_name)
                    if score > best_score and score > 0.8:  # 0.8 threshold for similarity
                        best_score = score
                        best_match = action_name

                if best_match:
                    self.logger.info(
                        f"Fuzzy matched action '{proposed_action}' to '{best_match}' (similarity: {best_score:.2f})")
                    proposed_action = best_match
                else:
                    raise KeyError(f"No matching action unit found for: {proposed_action}")

            # Get optional fields with defaults
            retract = bool(response.get("retract", False))
            retract_to_step = response.get("retract_to_step")
            if retract_to_step is not None:
                retract_to_step = int(retract_to_step)
            reasoning = str(response.get("reasoning", "No reasoning provided"))

            return proposed_action, retract, retract_to_step, reasoning
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse planning response: {e}\nOriginal content: {content}")
            # Default to next action in sequence as fallback
            action_names = list(self.action_units.keys())
            next_action = action_names[self.task_context.current_step]
            return next_action, False, None, "Error in parsing response, using default next action"

    def parse_code_review_response(self, content: str) -> bool:
        # Rule 1: Check alphanumeric-only lowercase string
        alpha_only = ''.join(c.lower() for c in content if c.isalpha())
        if 'finaldecisionapproved' in alpha_only:
            return True
        if 'finaldecisionrejected' in alpha_only:
            return False

        # Rule 2: Check whitespace-stripped string with colons
        clean_content = ''.join(content.split())
        if ':Approved' in clean_content:
            return True
        if ':Rejected' in clean_content:
            return False

        # Rule 3: Default to False for all other cases
        return False

    async def act(self, message: Message) -> Optional[Message]:
        """Unified method for handling messages and generating responses"""
        if message.role == Role.PI:
            return await self.start_next_step()

        if message.type in [MessageType.CODE_WRITING_REQUEST, MessageType.CODE_REVISION_REQUEST,
                            MessageType.PLANNING_REQUEST]:
            if message.type == MessageType.CODE_WRITING_REQUEST and self.current_action.code_snippet:
                response = self.current_action.code_snippet
            elif message.type == MessageType.PLANNING_REQUEST and not self.enable_planning:
                response = self.get_default_planning_response()
            else:
                response = await self.ask_llm(message.content)
            return Message(
                role=self.role,
                type=MessageType.get_response_type(message),
                content=response,
                target_roles={message.role}
            )

        if message.type == MessageType.PLANNING_RESPONSE:
            action_name, do_retract, retract_step, reasoning = self.parse_planning_response(message.content)

            if do_retract and self.remaining_retractions > 0 and retract_step is not None:
                try:
                    retract_result = await self.retract_to_step(retract_step)
                    if isinstance(retract_result,
                                  Message) and retract_result.type == MessageType.TIMEOUT:  # Timeout occurred
                        return retract_result
                    self.logger.info(f"Retracted to step {retract_step}. Reasoning: {reasoning}")
                except ValueError as e:
                    self.logger.error(f"Failed to retract: {e}")
                    # Continue with chosen action without retraction

            self.current_action = self.action_units[action_name]
            if self.current_action.name == "TASK COMPLETED":
                self.task_completed = True
                return None

            return Message(
                role=self.role,
                type=MessageType.CODE_WRITING_REQUEST,
                content=self.prepare_code_writing_prompt(),
                target_roles={Role.DOMAIN_EXPERT} if (
                        self.current_action.requires_domain_knowledge and self.include_domain_expert) else {self.role}
            )

        if message.type in [MessageType.CODE_WRITING_RESPONSE, MessageType.CODE_REVISION_RESPONSE]:
            if message.type == MessageType.CODE_REVISION_RESPONSE:
                self.review_round += 1
                prev_result = await self.executor.reset_and_execute_to_step(
                    self.task_context,
                    self.task_context.current_step - 1,
                    self.get_remaining_time()
                )
                if prev_result and prev_result.is_timeout:
                    return self.create_timeout_message("error recovery")

            # Handle new or revised code (from self or domain expert)
            code = self.parse_code(message.content)
            result = await self.executor.execute(code, timeout=self.get_remaining_time())
            if result.is_timeout:
                return self.create_timeout_message("code processing")

            # Add step to context
            step_type = StepType.DEBUG if message.type == MessageType.CODE_REVISION_RESPONSE else StepType.REGULAR
            self.task_context.add_step(
                step_type=step_type,
                action_name=self.current_action.name,
                instruction=self.current_action.instruction,
                code=code,
                stdout=result.stdout,
                error_trace=result.error_trace
            )

            self.logger.debug(self.task_context.format_context(mode="last"))
            # Decide whether to send for review
            do_review = self.review_round < self.max_rounds

            if do_review:
                target_role = Role.DOMAIN_EXPERT if (
                        self.current_action.requires_domain_knowledge and self.include_domain_expert) else Role.CODE_REVIEWER
                return Message(
                    role=self.role,
                    type=MessageType.CODE_REVIEW_REQUEST,
                    content=self.prepare_code_review_prompt(),
                    target_roles={target_role}
                )
            else:
                return await self.start_next_step()

        if message.type == MessageType.CODE_REVIEW_RESPONSE:
            if self.parse_code_review_response(message.content):
                if self.use_code_snippet and (not self.current_action.requires_domain_knowledge):
                    self.current_action.code_snippet = self.task_context.get_last_step().code
                return await self.start_next_step()
            else:
                # Request code revision
                target_role = Role.DOMAIN_EXPERT if (
                        self.current_action.requires_domain_knowledge and self.include_domain_expert) else self.role
                return Message(
                    role=self.role,
                    type=MessageType.CODE_REVISION_REQUEST,
                    content=self.prepare_code_revision_prompt(message.content),
                    target_roles={target_role}
                )

        return None


class CodeReviewerAgent(BaseAgent):
    def __init__(self, client: LLMClient, logger: Logger, role_prompt: Optional[str] = None,
                 guidelines: Optional[str] = None,
                 args: Optional[argparse.Namespace] = None):
        super().__init__(Role.CODE_REVIEWER, client, logger, role_prompt=role_prompt, guidelines=guidelines, args=args)
        self.role_prompt = "You are a code reviewer in this project."

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


class DomainExpertAgent(BaseAgent):
    def __init__(self, client: LLMClient, logger: Logger, role_prompt: Optional[str] = None,
                 guidelines: Optional[str] = None,
                 args: Optional[argparse.Namespace] = None):
        super().__init__(Role.DOMAIN_EXPERT, client, logger, role_prompt=role_prompt, guidelines=guidelines, args=args)
        self.role_prompt = "You are a domain expert in this biomedical research project."

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


class Environment:
    def __init__(self, logger: Logger, agents: Optional[List[BaseAgent]] = None,
                 args: Optional[argparse.Namespace] = None):
        self.logger = logger
        self.agents: Dict[Role, BaseAgent] = {}
        if agents:
            for agent in agents:
                self.add_agent(agent)

        self.topology = {
            Role.PI: {Role.PI, Role.GEO_AGENT, Role.TCGA_AGENT, Role.STATISTICIAN_AGENT, Role.CODE_REVIEWER,
                      Role.DOMAIN_EXPERT},
            Role.GEO_AGENT: {Role.PI, Role.GEO_AGENT, Role.CODE_REVIEWER, Role.DOMAIN_EXPERT},
            Role.TCGA_AGENT: {Role.PI, Role.TCGA_AGENT, Role.CODE_REVIEWER, Role.DOMAIN_EXPERT},
            Role.STATISTICIAN_AGENT: {Role.PI, Role.STATISTICIAN_AGENT, Role.CODE_REVIEWER},
            Role.CODE_REVIEWER: {Role.PI, Role.GEO_AGENT, Role.TCGA_AGENT, Role.STATISTICIAN_AGENT},
            Role.DOMAIN_EXPERT: {Role.PI, Role.GEO_AGENT, Role.TCGA_AGENT}
        }

        self.message_queue: List[Message] = []
        self.start_time: Optional[float] = None

    def add_agent(self, agent: BaseAgent):
        self.agents[agent.role] = agent

    async def run(self) -> Optional[str]:
        """Run the multi-agent system"""
        self.start_time = time.time()

        # Start with PI's message (now using None as trigger)
        initial_message = await self.agents[Role.PI].act(None)
        if initial_message:
            self.message_queue.append(initial_message)

        while self.message_queue:
            # Check global timeout
            if time.time() - self.start_time > GLOBAL_MAX_TIME:
                self.logger.error(f"TimeoutError! {GLOBAL_MAX_TIME}s exceeded, early stopped this cohort.")
                break

            current_message = self.message_queue.pop(0)

            valid_receivers = current_message.target_roles.intersection(
                self.topology[current_message.role]
            )

            # Let valid receivers observe and act
            for role in valid_receivers:
                if role in self.agents:
                    new_message = await self.agents[role].act(current_message)
                    if new_message:
                        if not MessageType.is_request(new_message):
                            msg_preview = '\n'.join(new_message.content.splitlines()[-30:])
                            self.logger.debug(f"\n【{new_message.role.value}】 ({new_message.type.value})\n{msg_preview}")
                        if new_message.type == MessageType.TIMEOUT:
                            self.logger.error(
                                f"TimeoutError! {new_message.role.value} reported timeout, early stopped this cohort.")
                            self.message_queue.clear()
                            break
                        self.message_queue.append(new_message)

            tcga_agent = self.agents[Role.TCGA_AGENT]
            if tcga_agent.task_completed:
                break

        return self.agents[Role.TCGA_AGENT].task_context.concatenate_snippets()

    def clear_states(self):
        """Clear states before processing new cohort"""
        for agent in self.agents.values():
            agent.clear_states()
        self.message_queue.clear()
        self.start_time = None


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="TCGA cohort data wrangling experiments with LLM-based agents.")
    parser.add_argument('--max_rounds', type=int, default=2, help='Maximum number of revision rounds.')
    parser.add_argument('--de', type=lambda x: (str(x).lower() == 'true'), default=True, help='Include domain expert.')
    parser.add_argument('--cs', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use code snippet.')
    parser.add_argument('--version', type=str, required=True, help='Version string for the current run of experiment.')
    parser.add_argument('--resume', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Continue from next cohort.')
    parser.add_argument('--model', type=str, required=True, help='Name of LLM.')
    parser.add_argument('--provider', type=str, default='none',
                        choices=['none', 'openai', 'anthropic', 'google', 'meta', 'deepseek'],
                        help='Provider of LLM. Use "none" to auto-detect from model name.')
    parser.add_argument('--api', type=int, default=None,
                        help='Index of API configuration to use (1-based). If not provided, uses default API keys.')
    parser.add_argument('--use-api', action='store_true',
                        help='Use API service for open source models (e.g., Novita for Llama)')
    parser.add_argument('--max-retract', type=int, default=1,
                        help='Maximum number of times allowed to retract to previous steps')
    parser.add_argument('--plan', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Enable context-aware planning. Use "true" or "false".')
    parser.add_argument('--max-time', type=float, default=600,
                        help='Maximum time (in seconds) allowed for the task')
    return parser


def load_last_cohort_info(version_dir):
    try:
        with open(os.path.join(version_dir, "last_cohort_info.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_last_cohort_info(version_dir, cohort_info):
    with open(os.path.join(version_dir, "last_cohort_info.json"), "w") as f:
        json.dump(cohort_info, f)


def delete_corrupted_files(output_dir, cohort):
    out_gene_dir = os.path.join(output_dir, 'gene_data')
    out_clinical_dir = os.path.join(output_dir, 'clinical_data')
    out_code_dir = os.path.join(output_dir, 'code')
    for this_dir in [output_dir, out_gene_dir, out_clinical_dir]:
        file_path = os.path.join(this_dir, f"{cohort}.csv")
        if os.path.exists(file_path):
            os.remove(file_path)
    code_path = os.path.join(out_code_dir, f"{cohort}.py")
    if os.path.exists(code_path):
        os.remove(code_path)


async def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    client = get_llm_client(args)

    # Read tools code
    tool_file = "./tools/preprocess.py"
    with open(tool_file, 'r') as file:
        tools_code_full = file.read()
    extracted_code = extract_function_code(tool_file, ["tcga_convert_trait",
                                                       "tcga_convert_age",
                                                       "tcga_convert_gender",
                                                       "tcga_select_clinical_features",
                                                       "preview_df"])
    tools = {"full": PREPROCESS_TOOLS.format(tools_code=tools_code_full),
             "domain_focus": PREPROCESS_TOOLS.format(tools_code=extracted_code)}
    # Load traits with special ordering
    with open("./metadata/all_traits.json", "r") as f:
        all_traits = json.load(f)
    all_traits = [normalize_trait(t) for t in all_traits]
    special_traits = ["Breast_Cancer", "Epilepsy", "Atherosclerosis"]
    all_traits = [t for t in all_traits if t not in special_traits]
    all_traits = special_traits + all_traits

    all_traits = ['Breast_Cancer', 'Lung_Cancer', 'Esophageal Cancer', 'Rectal Cancer', 'Ovarian Cancer', 'Testicular Cancer']
    # all_traits = [
    #     "Age-Related Macular Degeneration",
    #     "Alcohol Flush Reaction",
    #     "Allergies",
    #     "Alopecia",
    #     "Alzheimer's Disease",
    #     "Amyotrophic Lateral Sclerosis"]

    input_dir = '/media/techt/DATA/TCGA' if os.path.exists('/media/techt/DATA/TCGA') else '../DATA/TCGA'
    output_root = './output/preprocess/'
    version = args.version
    out_version_dir = os.path.join(output_root, version)

    if not args.resume:
        confirm_delete = input(
            f"Do you want to delete all the output data in '{out_version_dir}'? [yes/no]: ")
        if confirm_delete.lower() == 'yes':
            shutil.rmtree(out_version_dir)

    os.makedirs(out_version_dir, exist_ok=True)
    log_file = os.path.join(out_version_dir, 'log.txt')
    logger = Logger(log_file=log_file, max_msg_length=5000)

    action_units = [
        ActionUnit("Initial Data Selection and Loading",
                   TCGA_DATA_LOADING_PROMPT.format(list_subdirs=os.listdir(input_dir))),
        ActionUnit("Find Candidate Demographic Features", TCGA_FIND_CANDIDATE_DEMOGRAPHIC_PROMPT),
        ActionUnit("Select Demographic Features", TCGA_SELECT_DEMOGRAPHIC_PROMPT),
        ActionUnit("Feature Engineering and Validation", TCGA_FEATURE_ENGINEERING_PROMPT),
        ActionUnit("TASK COMPLETED", TASK_COMPLETED_PROMPT)
    ]

    tcga_agent = TCGAAgent(
        client=client,
        logger=logger,
        role_prompt=TCGA_ROLE_PROMPT,
        guidelines=TCGA_GUIDELINES,
        tools=tools,
        setups='',
        action_units=action_units,
        args=args
    )

    agents = [
        PIAgent(client=client, logger=logger, args=args),
        tcga_agent,
        CodeReviewerAgent(client=client, logger=logger, args=args),
        DomainExpertAgent(client=client, logger=logger, args=args)
    ]

    env = Environment(logger=logger, args=args)
    for agent in agents:
        env.add_agent(agent)

    last_cohort_info = load_last_cohort_info(out_version_dir)
    for index, trait in enumerate(all_traits):
        out_trait_dir = os.path.join(out_version_dir, trait)
        os.makedirs(out_trait_dir, exist_ok=True)
        out_gene_dir = os.path.join(out_trait_dir, 'gene_data')
        out_clinical_dir = os.path.join(out_trait_dir, 'clinical_data')
        out_log_dir = os.path.join(out_trait_dir, 'log')
        out_code_dir = os.path.join(out_trait_dir, 'code')
        for this_dir in [out_gene_dir, out_clinical_dir, out_log_dir, out_code_dir]:
            os.makedirs(this_dir, exist_ok=True)

        json_path = os.path.join(out_trait_dir, "cohort_info.json")
        cohort = "TCGA"

        out_data_file = os.path.join(out_trait_dir, f"{cohort}.csv")
        out_gene_data_file = os.path.join(out_gene_dir, f"{cohort}.csv")
        out_clinical_data_file = os.path.join(out_clinical_dir, f"{cohort}.csv")

        try:
            # Create path configuration for this cohort
            path_config = TCGAPathConfig(
                trait=trait,
                tcga_root_dir=input_dir,
                out_data_file=out_data_file,
                out_gene_data_file=out_gene_data_file,
                out_clinical_data_file=out_clinical_data_file,
                json_path=json_path,
            )

            tcga_agent.update_path_config(path_config)

            # Clear states before processing new cohort
            env.clear_states()

            # Run the environment
            code = await asyncio.wait_for(env.run(), timeout=GLOBAL_MAX_TIME)

            # Save the final code to a file
            code_file = os.path.join(out_code_dir, f"{cohort}.py")
            with open(code_file, "w") as cf:
                cf.write(code)

            # Save the current state
            save_last_cohort_info(out_version_dir, {'trait': trait, 'cohort': cohort})
            logger.summarize()

        except asyncio.TimeoutError:
            logger.error(f"Timeout error occurred while processing trait {trait}, cohort {cohort}")
        except Exception as e:
            logger.error(
                f"Error occurred while processing trait {trait}, cohort {cohort}\n {type(e).__name__}: {str(e)}\n{traceback.format_exc()}")
        finally:
            # Reset stdout and stderr to default
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if __name__ == "__main__":
    asyncio.run(main())
