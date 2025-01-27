import argparse
import json
from typing import Dict, List, Optional, Tuple

from core.context import ActionUnit, TaskContext, StepType
from core.execution import CodeExecutor
from core.messaging import Role, Message, MessageType
from prompts import *
from utils.llm import LLMClient
from utils.logger import Logger
from utils.path_config import PathConfig
from .base_agent import BaseAgent


class MultiStepProgrammingAgent(BaseAgent):
    def __init__(self, role: Role, client: LLMClient, logger: Logger, role_prompt: str, guidelines: str,
                 tools: Dict[str, str],
                 setups: str,
                 action_units: List[ActionUnit], args: argparse.Namespace):
        super().__init__(role, client, logger, role_prompt=role_prompt, guidelines=guidelines, tools=tools,
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

    def set_path_config(self, path_config: PathConfig):
        """Set up path configuration and related prompts for a new cohort."""
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
            # do_review = self.review_round < self.max_rounds
            do_review = self.review_round < self.max_rounds and (
                ((not self.role == Role.STATISTICIAN_AGENT) or result.error or
                 (not self.current_action.code_snippet) or self.task_context.debug_step != 0)
            )

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
                if self.use_code_snippet and not (self.current_action.requires_domain_knowledge
                                                  and self.include_domain_expert):
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
