import sys
import io
import os
from contextlib import redirect_stdout, redirect_stderr
import json
import re
import copy
import time
import argparse
import shutil
import signal

from openai import AzureOpenAI
import pandas as pd
import numpy as np
import ast
import subprocess
from utils.statistics import normalize_trait
from prompts.preprocess import *

# Azure OpenAI client setup
client = AzureOpenAI(
    api_key="57983a2e88fa4d6b81205a8d55d9bd46",
    api_version="2023-10-01-preview",
    azure_endpoint="https://haoyang2.openai.azure.com/"
)


# Define the timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Processing this cohort took too long!")


# Set the signal handler for the alarm signal
signal.signal(signal.SIGALRM, timeout_handler)

CODE_INDUCER = """
NOTE: 
ONLY IMPLEMENT CODE FOR THE CURRENT STEP. MAKE SURE THE CODE CAN BE CONCATENATED WITH THE CODE FROM PREVIOUS STEPS AND CORRECTLY EXECUTED.

Return ```python your_code_here ``` with NO other texts. your_code_here is a placeholder.
your code:

"""

CODE_INDUCER2 = """
NOTE: 
ONLY IMPLEMENT CODE FOR THE CURRENT STEP. MAKE SURE THE CODE CAN BE CONCATENATED WITH THE CODE FROM PREVIOUS STEPS AND CORRECTLY EXECUTED.

Please fill in the following code template:
```python 
# Initialize variables. Remember to change their values if they are available.
is_gene_available = False
trait_row = age_row = gender_row = None  # set to different values when applicable
convert_trait = convert_age = convert_age = None  # define the functions when applicable

your_code_here 
```with NO other texts. your_code_here is a placeholder.
your code:

"""


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.start_time = time.time()
        self.total_duration = 0
        self.api_calls = []
        self.token_consumption = []

    def log_api_call(self, elapsed_time, tokens_consumed):
        self.api_calls.append(elapsed_time)
        self.token_consumption.append(tokens_consumed)

    def log_message(self, message):
        with open(self.log_file, "a") as log_f:
            log_f.write(f"{message}\n")

    def finalize(self):
        end_time = time.time()
        self.total_duration = end_time - self.start_time
        total_tokens = sum(self.token_consumption)
        total_api_duration = sum(self.api_calls)

        with open(self.log_file, "a") as log_f:
            log_f.write(f"\nTotal API Call Duration: {total_api_duration} seconds\n")
            log_f.write(f"Total Tokens Consumed: {total_tokens}\n")
            log_f.write(f"Total Processing Duration: {self.total_duration} seconds\n")


def call_openai_gpt(prompt, sys_prompt=None, logger=None):
    if sys_prompt is None:
        sys_prompt = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]
    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-4",  # Adjust the model name as needed
        messages=messages
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    tokens_consumed = response.usage.total_tokens
    if logger:
        logger.log_api_call(elapsed_time, tokens_consumed)
    return response.choices[0].message.content


class TaskContext:
    def __init__(self):
        self.history = []
        self.current_step = 0
        self.debug_step = 0

    def add_step(self, debug, code_snippet, stdout, stderr, error=None, action_unit_name=None, instruction=None):
        if debug:
            self.debug_step += 1
            action_unit_name = instruction = None
        else:
            assert action_unit_name is not None and instruction is not None, "For non-debugging steps, the name and " \
                                                                             "instruction of the action unit must be specified."
            self.current_step += 1
        step = {
            'debug': debug,
            'index': self.debug_step if debug else self.current_step,
            'action_unit_name': action_unit_name,
            'instruction': instruction,
            'code_snippet': code_snippet,
            'stdout': stdout,
            'stderr': stderr,
            'error': error
        }
        self.history.append(step)

    def display(self, mode="all", domain_focus=False):
        assert mode in ["all", "past", "last"], "Unsupported mode: must be one of 'all', 'past', 'last'."
        start_id = self.current_step - 1 if domain_focus else None
        if mode == "all":
            contxt_to_display = self.history[start_id:]
        elif mode == "past":
            contxt_to_display = self.history[start_id:-1]
        elif mode == "last":
            contxt_to_display = self.history[-1:]
        formatted_context = []
        for step in contxt_to_display:
            debug = step['debug']
            if debug:
                formatted_context.append(f"Debugging Attempt {step['index']}")
            else:
                formatted_context.append(f"STEP {step['index']}")
                formatted_context.append(f"[Chosen action unit]: {step['action_unit_name']}")
                formatted_context.append(f"[Instruction]:\n{step['instruction']}")
                if domain_focus:
                    formatted_context.append(self.history[start_id - 1]['stdout'])
            formatted_context.append(f"[Code]:\n{step['code_snippet']}")
            formatted_context.append(f"[Output]:\n{step['stdout']}")
            if step['stderr']:
                formatted_context.append(f"[Errors]:\n{step['stderr']}")
            if step['error']:
                formatted_context.append(f"[Execution Error]:\n{step['error']}")
            if debug:
                formatted_context.append("-" * 50)
            else:
                formatted_context.append("=" * 50)
        return "\n".join(formatted_context)

    def concatenate_snippets(self, mode="all"):
        assert mode in ["all", "previous"], "Unsupported mode: must be one of 'all', 'previous'."
        if mode == "all":
            end_idx = self.current_step
        elif mode == "previous":
            end_idx = self.current_step - 1
        return "\n".join([step['code_snippet'] for step in self.history[:end_idx]])


class ActionUnit:
    def __init__(self, name, instruction, code_snippet=""):
        self.name = name
        self.instruction = instruction
        self.code_snippet = code_snippet
        self.code_snippet_buffer = []

    def __str__(self):
        return f"Name: {self.name}\nInstruction: {self.instruction}"


class GEOAgent:
    def __init__(self, role_prompt, guidelines, tools, setups, action_units, logger, max_rounds=2,
                 include_domain_expert=True, use_code_snippet=False):
        self.role_prompt = role_prompt
        self.guidelines = guidelines
        self.tools = tools
        self.setups = setups
        self.action_units = {unit.name: unit for unit in action_units}
        self.task_context = TaskContext()
        self.current_exec_state = {}
        self.max_rounds = max_rounds
        self.logger = logger
        self.one_history_only = ['2', '4'] if include_domain_expert else []
        self.need_biomedical_knowledge = ['2', '4', '6'] if include_domain_expert else []
        if not use_code_snippet:
            for unit in self.action_units.values():
                unit.code_snippet = ""

    def ask(self, prompt):
        return call_openai_gpt(prompt, self.role_prompt, self.logger)

    def clear_states(self, context=False, exe_state=False):
        if context:
            del self.task_context
            self.task_context = TaskContext()
        if exe_state:
            del self.current_exec_state
            self.current_exec_state = {}

    def prepare_prompt(self, include_tool_setups=True, mode="all", domain_focus=False):
        assert mode in ["all", "past", "last"], "Unsupported mode: must be one of 'all', 'past', 'last'."
        formatted_prompt = []
        if (not domain_focus) and (mode != "last"):
            if include_tool_setups:
                formatted_prompt.append(
                    "To help you prepare, I will provide you with the following: the task guidelines, "
                    "the function tools, the programming setups, and the history of previous steps "
                    "taken, including the instructions, code, and execution output of each step.")
            else:
                formatted_prompt.append(
                    "To help you prepare, I will provide you with the following: the task guidelines, "
                    "and the history of previous steps taken, including the "
                    "instructions, code, and execution output of each step.")
            formatted_prompt.append(f"**General guidelines**: \n{self.guidelines}\n")
            if include_tool_setups:
                formatted_prompt.append(f"**Function tools**: \n{self.tools}\n")
                formatted_prompt.append(f"**Programming setups**: \n{self.setups}\n")
            formatted_prompt.append(f"**Task history**: \n{self.task_context.display(mode, domain_focus)}")
        else:
            formatted_prompt.append(self.task_context.display(mode, domain_focus))

        return "\n".join(formatted_prompt)

    def merge_revision_into_context(self):
        current_step = self.task_context.current_step
        debug_step = self.task_context.debug_step
        if debug_step != 0:
            for key in ['code_snippet', 'stdout', 'stderr', 'error']:
                self.task_context.history[current_step - 1][key] = self.task_context.history[-1][key]
        self.task_context.history = self.task_context.history[: current_step]
        self.task_context.debug_step = 0

    def check_code_snippet_buffer(self):
        for unit in self.action_units.values():
            if len(unit.code_snippet_buffer) >= 3:
                modified_snippet = self.aggregate_code_snippets(unit)
                unit.code_snippet = modified_snippet
                unit.code_snippet_buffer.clear()

    def aggregate_code_snippets(self, unit):
        original_snippet = unit.code_snippet
        revised_versions = unit.code_snippet_buffer
        formatted_versions = [f"*VERSION {i + 1}*: \n{version}" for i, version in enumerate(revised_versions)]
        formatted_versions = '\n\n'.join(formatted_versions)
        prompt = []
        prompt.append(
            "Given background information and the original code, please read the candidate revised versions of the code"
            " to understand the revisions made, and either select the best one or combine their advantages to write a "
            "single revised version of the code. I will provide you with all the necessary information below.\n")
        prompt.append("Gold guidelines of the overall task:\n")
        prompt.append(f"{self.guidelines}\n")
        prompt.append("Current understanding of the subtask (which may or may not be correct):\n")
        prompt.append(f"{unit.instruction}\n")
        prompt.append("Current code snippet (which didn't work as expected):\n")
        prompt.append(f"{original_snippet}\n")
        prompt.append("Candidate revised versions that worked:\n")
        prompt.append(f"{formatted_versions}\n")
        prompt.append(f"{CODE_INDUCER}")
        prompt = "\n".join(prompt)
        response = self.ask(prompt)
        code = self.parse_code(response)
        return code

    def choose_action_unit(self):
        return str(self.task_context.current_step + 1)

    def run_snippet(self, snippet, namespace):
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(snippet, namespace)
            return stdout.getvalue(), stderr.getvalue(), None
        except Exception as e:
            return stdout.getvalue(), stderr.getvalue(), e

    def write_initial_code(self, action_unit):
        code_inducer = CODE_INDUCER2 if action_unit.name == "2" else CODE_INDUCER
        if action_unit.name not in self.one_history_only:
            prompt = self.prepare_prompt()
            prompt += f"\n**TO DO: Programming** \nNow that you've been familiar with the task setups and current status" \
                      f", please write the code following the instructions:\n\n{action_unit.instruction}\n"
            prompt = prompt + code_inducer
            if action_unit.name not in self.need_biomedical_knowledge:
                response = self.ask(prompt)
            else:
                expert = DomainExpertAgent(logger=self.logger)
                response = expert.ask(prompt)
        else:
            prompt = f"{action_unit.instruction}\n" \
                     f"{self.task_context.history[-1]['stdout']}\n\n" \
                     f"{code_inducer}"  # CODE_INDUCER 1 AND 2, TWO TYPES
            expert = DomainExpertAgent(logger=self.logger)
            response = expert.ask(prompt)
        code = self.parse_code(response)
        return code

    def send_code_for_review(self, action_unit):
        formatted_prompt = []
        domain_focus = action_unit.name in self.one_history_only
        if domain_focus:
            formatted_prompt.append("The detailed task instructions are provided below. Sometimes the record of"
                                    "previous attempts are also provided")
        formatted_prompt.append(self.prepare_prompt(mode="past", domain_focus=domain_focus))
        if action_unit.name in self.need_biomedical_knowledge:
            domain_trigger = "Some tasks may involve understanding the data and making inferences based on biomedical " \
                             "knowledge. These inferences might require assumptions, which do not need to be fully " \
                             "validated, though they need to be reasonable. "
        else:
            domain_trigger = "\n"

        formatted_prompt.append("\n**TO DO: Code Review**\n"
                                "The following code is the latest attempt for the current step and requires your review. "
                                "If previous attempts have been included in the task history above, their presence"
                                " does not indicate they succeeded or failed, though you can refer to their execution "
                                "outputs for context. \nOnly review the latest code attempt provided below.\n")
        formatted_prompt.append(self.prepare_prompt(mode="last", domain_focus=domain_focus))
        formatted_prompt.append(f"\nPlease review the code according to the following criteria:\n"
                                "1. *Functionality*: Can the code be successfully executed in the current setting?\n"
                                f"2. *Conformance*: Does the code conform to the given instructions? {domain_trigger}"
                                "Provide suggestions for revision and improvement if necessary.\n"
                                "*NOTE*:\n"
                                "1. Your review is not concerned with engineering code quality. The code is a quick "
                                "demo for a research project, so the standards should not be strict.\n"
                                "2. If you provide suggestions, please limit them to 1 to 3 key suggestions. Focus on "
                                "the most important aspects, such as how to solve the execution errors or make the code "
                                "conform to the instructions.\n\n"
                                "Return your decision in the format: \"Final Decision: Approved\" or \"Final Decision: "
                                "Rejected.\"")
        prompt = "\n".join(formatted_prompt)
        if action_unit.name not in self.need_biomedical_knowledge:
            reviewer = CodeReviewerAgent(logger=self.logger)
        else:
            reviewer = DomainExpertAgent(logger=self.logger)
        response = reviewer.ask(prompt)
        return response

    def correct_code(self, action_unit, feedback):
        code_inducer = CODE_INDUCER2 if action_unit.name == "2" else CODE_INDUCER
        formatted_prompt = []
        domain_focus = action_unit.name in self.one_history_only
        if domain_focus:
            formatted_prompt.append("The detailed task instructions are provided below. Sometimes the record of"
                                    "previous attempts are also provided")
        formatted_prompt.append(self.prepare_prompt(mode="past", domain_focus=domain_focus))
        formatted_prompt.append(
            f"\nThe following code is the latest attempt for the current step and requires correction. "
            "If previous attempts have been included in the task history above, their presence"
            " does not indicate they succeeded or failed, though you can refer to their execution "
            "outputs for context. \nOnly correct the latest code attempt provided below.\n")
        formatted_prompt.append(self.prepare_prompt(mode="last", domain_focus=domain_focus))
        formatted_prompt.append(f"Use the reviewer's feedback to help debug and identify logical errors in the code. "
                                f"While the feedback is generally reliable, it might occasionally include errors or "
                                f"suggest changes that are impractical in the current context. Make corrections where "
                                f"you agree with the feedback, but retain the original code where you do not.\n")
        formatted_prompt.append(f"\nReviewer's feedback:\n{feedback}\n")
        formatted_prompt.append(code_inducer)
        prompt = "\n".join(formatted_prompt)
        if action_unit.name not in self.need_biomedical_knowledge:
            response = self.ask(prompt)
        else:
            expert = DomainExpertAgent(logger=self.logger)
            response = expert.ask(prompt)

        code = self.parse_code(response)
        return code

    def review_and_correct(self, action_unit):
        round_counter = 0
        while round_counter < self.max_rounds:
            last_step = self.task_context.history[-1]
            stderr, error = last_step["stderr"], last_step["error"]
            # Send code for review
            feedback = self.send_code_for_review(action_unit)
            print(feedback)
            if ":approved" in feedback.lower().replace(" ", "") and not stderr and not error:
                break

            self.clear_states(exe_state=True)
            code_to_repeat = self.task_context.concatenate_snippets(mode="previous")
            _, _, _ = self.run_snippet(code_to_repeat, self.current_exec_state)

            # Correct the code based on feedback
            new_code_snippet = self.correct_code(action_unit, feedback)
            stdout, stderr, error = self.run_snippet(new_code_snippet, self.current_exec_state)

            self.task_context.add_step(
                debug=True,
                code_snippet=new_code_snippet,
                stdout=stdout,
                stderr=stderr,
                error=str(error) if error else None
            )
            print(self.task_context.display(mode="last"))

            round_counter += 1

        if round_counter >= self.max_rounds:
            print(f"Maximum revision attempts {self.max_rounds} reached. Use the code from latest attempt without "
                  f"review.")
        self.merge_revision_into_context()

    def execute_action_unit(self, action_unit_name):
        action_unit = self.action_units[action_unit_name]
        code_snippet = action_unit.code_snippet
        if not code_snippet:
            code_snippet = self.write_initial_code(action_unit)

        stdout, stderr, error = self.run_snippet(code_snippet, self.current_exec_state)

        self.task_context.add_step(
            debug=False,
            action_unit_name=action_unit_name,
            instruction=action_unit.instruction,
            code_snippet=code_snippet,
            stdout=stdout,
            stderr=stderr,
            error=str(error) if error else None
        )
        print(self.task_context.display(mode="last"))

        if stderr or error or not action_unit.code_snippet:
            self.review_and_correct(action_unit)

    def run_task(self):
        self.clear_states(context=True, exe_state=True)
        self.check_code_snippet_buffer()
        while True:
            action_unit_name = self.choose_action_unit()
            if action_unit_name == "8":
                break
            self.execute_action_unit(action_unit_name)
            if action_unit_name == "2":
                is_gene_available = self.current_exec_state.get("is_gene_available")
                trait_row = self.current_exec_state.get("trait_row")
                if is_gene_available is False or trait_row is None:
                    print("Cohort not usable. Early stop triggered")
                    break
        return self.task_context.concatenate_snippets()

    @staticmethod
    def parse_code(rsp):
        pattern = r"```python(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text


class CodeReviewerAgent:
    def __init__(self, role_prompt="You are a code reviewer in this project.", logger=None):
        self.role_prompt = role_prompt
        self.logger = logger

    def ask(self, prompt):
        return call_openai_gpt(prompt, self.role_prompt, self.logger)


class DomainExpertAgent:
    def __init__(self, role_prompt="You are a domain expert in this biomedical research project.", logger=None):
        self.role_prompt = role_prompt
        self.logger = logger

    def ask(self, prompt):
        return call_openai_gpt(prompt, self.role_prompt, self.logger)


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="GEO cohort data wrangling experiments with LLM-based agents.")
    parser.add_argument('--max_rounds', type=int, default=2, help='Maximum number of revision rounds.')
    parser.add_argument('--de', type=lambda x: (str(x).lower() == 'true'), default=True, help='Include domain expert.')
    parser.add_argument('--cs', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use code snippet.')
    parser.add_argument('--version', type=str, required=True, help='Version string for the current run of experiment.')
    parser.add_argument('--resume', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Continue from next cohort.')
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
    out_trait_dir = os.path.join(output_dir, 'trait_data')
    out_code_dir = os.path.join(output_dir, 'code')
    for this_dir in [output_dir, out_gene_dir, out_trait_dir]:
        file_path = os.path.join(this_dir, f"{cohort}.csv")
        if os.path.exists(file_path):
            os.remove(file_path)
    code_path = os.path.join(out_code_dir, f"{cohort}.py")
    if os.path.exists(code_path):
        os.remove(code_path)


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    all_traits = pd.read_csv("all_traits.csv")["Trait"].tolist()
    all_traits = [normalize_trait(t) for t in all_traits]
    all_traits = [t for t in all_traits if t not in ["Breast_Cancer", "Epilepsy"]]
    all_traits = ["Breast_Cancer", "Epilepsy"] + all_traits
    input_dir = '/media/techt/DATA/GEO' if os.path.exists('/media/techt/DATA/GEO') else '../DATA/GEO'

    output_root = './output/preprocess/'
    version = args.version
    version_dir = os.path.join(output_root, version)

    utils_code = "".join(open("utils/preprocess.py", 'r').readlines())
    tools = TOOLS.format(utils_code=utils_code)

    last_cohort_info = load_last_cohort_info(version_dir)

    for index, trait in enumerate(all_traits):
        in_trait_dir = os.path.join(input_dir, trait)
        if not os.path.isdir(in_trait_dir):
            print(f"Trait directory not found: {in_trait_dir}")
            continue
        output_dir = os.path.join(version_dir, trait)
        os.makedirs(output_dir, exist_ok=True)
        out_gene_dir = os.path.join(output_dir, 'gene_data')
        out_trait_dir = os.path.join(output_dir, 'trait_data')
        out_log_dir = os.path.join(output_dir, 'log')
        out_code_dir = os.path.join(output_dir, 'code')
        for this_dir in [out_gene_dir, out_trait_dir, out_log_dir, out_code_dir]:
            os.makedirs(this_dir, exist_ok=True)

        json_path = os.path.join(output_dir, "cohort_info.json")

        cohorts = os.listdir(in_trait_dir)
        for cohort in cohorts:
            if args.resume and last_cohort_info:
                if last_cohort_info['trait'] == trait and last_cohort_info['cohort'] == cohort:
                    delete_corrupted_files(output_dir, cohort)
                    last_cohort_info = None  # Reset last_cohort_info to avoid skipping further cohorts
                    continue

            try:
                signal.alarm(600)  # Set a timeout alarm for 600 seconds
                in_cohort_dir = os.path.join(in_trait_dir, cohort)
                if not os.path.isdir(in_cohort_dir):
                    print(f"Cohort directory not found: {in_cohort_dir}")
                    continue

                if not args.resume:
                    # Prompt to confirm deletion
                    confirm_delete = input(f"Do you want to delete all the output data in '{version_dir}'? [yes/no]: ")
                    if confirm_delete.lower() == 'yes':
                        shutil.rmtree(version_dir)

                setups = SETUPS.format(in_cohort_dir=in_cohort_dir, output_dir=output_dir)
                out_data_file = os.path.join(output_dir, f"{cohort}.csv")
                out_gene_data_file = os.path.join(out_gene_dir, f"{cohort}.csv")
                out_trait_data_file = os.path.join(out_trait_dir, f"{cohort}.csv")

                action_units = [
                    ActionUnit("1", INSTRUCTION_STEP1, CODE_STEP1.format(in_cohort_dir=in_cohort_dir)),
                    ActionUnit("2", INSTRUCTION_STEP2.format(trait=trait, out_trait_data_file=out_trait_data_file,
                                                             cohort=cohort, json_path=json_path)),
                    ActionUnit("3", INSTRUCTION_STEP3, CODE_STEP3),
                    ActionUnit("4", INSTRUCTION_STEP4),
                    ActionUnit("5", INSTRUCTION_STEP5, CODE_STEP5),
                    ActionUnit("6", INSTRUCTION_STEP6),
                    ActionUnit("7", INSTRUCTION_STEP7.format(trait=trait, cohort=cohort, json_path=json_path,
                                                             out_data_file=out_data_file,
                                                             out_gene_data_file=out_gene_data_file),
                               CODE_STEP7.format(trait=trait, cohort=cohort, json_path=json_path,
                                                 out_data_file=out_data_file, out_gene_data_file=out_gene_data_file)),
                    ActionUnit("8", "Task completed, you don't need to write any code.")
                ]

                log_file = os.path.join(out_log_dir, f"{cohort}.log")
                logger = Logger(log_file)

                geo_agent = GEOAgent(ROLE_PROMPT, GUIDELINES, tools, setups, action_units, logger, args.max_rounds,
                                     args.de, args.cs)

                with open(log_file, "a") as log_f:
                    sys.stdout = log_f
                    sys.stderr = log_f
                    code = geo_agent.run_task()
                    # Save the final code to a file
                    code_file = os.path.join(out_code_dir, f"{cohort}.py")
                    with open(code_file, "w") as cf:
                        cf.write(code)

                # Save the current state
                save_last_cohort_info(version_dir, {'trait': trait, 'cohort': cohort})
                logger.finalize()
                signal.alarm(0)  # Disable the alarm

            except TimeoutError:
                print(f"Timeout reached for cohort {cohort}. Skipping to the next one.")
                continue
            except Exception as e:
                print(e)
                continue
            finally:
                # Reset stdout and stderr to default
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                signal.alarm(0)  # Ensure the alarm is disabled after processing


if __name__ == "__main__":
    main()
