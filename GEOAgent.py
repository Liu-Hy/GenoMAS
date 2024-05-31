import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import json
import re
import copy
import time
from openai import AzureOpenAI

# Azure OpenAI client setup
client = AzureOpenAI(
    api_key="57983a2e88fa4d6b81205a8d55d9bd46",
    api_version="2023-10-01-preview",
    azure_endpoint="https://haoyang2.openai.azure.com/"
)

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
trait_row = None  # set to a value if trait is available or can be inferred
convert_trait = None  # define the function if trait is available or can be inferred

your_code_here 
```with NO other texts. your_code_here is a placeholder.
your code:

"""
def call_openai_gpt(prompt, sys_prompt=None):
    if sys_prompt is None:
        sys_prompt = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4",  # Adjust the model name as needed
        messages=messages
    )
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
        else:
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
                    formatted_context.append(self.history[start_id-1]['stdout'])
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

    def concatenate_snippets(self, up_to_index):
        return "\n".join([step['code_snippet'] for step in self.history[:up_to_index + 1]])


class ActionUnit:
    def __init__(self, name, instruction, code_snippet=""):
        self.name = name
        self.instruction = instruction
        self.code_snippet = code_snippet
        self.code_snippet_buffer = []

    def __str__(self):
        return f"Name: {self.name}\nInstruction: {self.instruction}"


class GEOAgent:
    def __init__(self, role_prompt, guidelines, tools, setups, action_units, max_rounds=2):
        self.role_prompt = role_prompt
        self.guidelines = guidelines
        self.tools = tools
        self.setups = setups
        self.action_units = {unit.name: unit for unit in action_units}
        self.task_context = TaskContext()
        self.current_exec_state = {}
        self.backup_exec_state = {}
        self.max_rounds = max_rounds
        self.one_history_only = ['2', '4']
        self.need_biomedical_knowledge = ['2', '4', '6']

    def ask(self, prompt):
        return call_openai_gpt(prompt, self.role_prompt)

    def clear_states(self, backup_only=False):
        del self.backup_exec_state
        self.backup_exec_state = {}
        if not backup_only:
            del self.current_exec_state
            self.current_exec_state = {}
            del self.task_context
            self.task_context = TaskContext()

    def prepare_prompt(self, include_tools=True, mode="all"):
        formatted_prompt = []
        if include_tools:
            formatted_prompt.append("To help you prepare, I will provide you with the following: the task guidelines, "
                                    "the function tools, the programming setups, and the history of previous steps "
                                    "taken, including the instructions, code, and execution output of each step.")
        else:
            formatted_prompt.append("To help you prepare, I will provide you with the following: the task guidelines, "
                                    "the programming setups, and the history of previous steps taken, including the "
                                    "instructions, code, and execution output of each step.")
        formatted_prompt.append(f"**General guidelines**: \n{self.guidelines}\n")
        if include_tools:
            formatted_prompt.append(f"**Function tools**: \n{self.tools}\n")
        formatted_prompt.append(f"**Programming setups**: \n{self.setups}\n")
        formatted_prompt.append(f"**Task history**: \n{self.task_context.display(mode)}")

        return "\n".join(formatted_prompt)

    def merge_revision_into_context(self):
        current_step = self.task_context.current_step
        debug_step = self.task_context.debug_step
        if debug_step != 0:
            for key in ['code_snippet', 'stdout', 'stderr', 'error']:
                self.task_context.history[current_step-1][key] = self.task_context.history[-1][key]
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
        # action_units_formatted = "\n".join([str(unit) for unit in self.action_units.values()])
        # prompt = []
        # prompt.append(
        #     "Please read the following information and choose one action unit for the next step. I will provide you "
        #     "with the general guidelines, task history, and the available action units below.\n\n")
        # prompt.append("**General Guidelines**:")
        # prompt.append(f"{self.guidelines}\n")
        # prompt.append("**Task History**:")
        # prompt.append(f"{self.task_context.display()}\n")
        # prompt.append("**Available Action Units**:")
        # prompt.append(f"{action_units_formatted}\n")
        # prompt.append(
        #     "**TO DO**: \nBased on this information, please choose one and only one action unit for the next step. Please"
        #     " only answer with the name of the unit, chosen from the list below:")
        # prompt.append(f"{[unit.name for unit in self.action_units.values()]}")
        # prompt.append("Your answer:")
        # prompt = "\n".join(prompt)
        # response = self.ask(prompt)
        # return response.strip()
        return str(self.task_context.current_step + 1)

    def execute_action_unit(self, action_unit_name):
        action_unit = self.action_units[action_unit_name]
        code_snippet = action_unit.code_snippet
        if not code_snippet:
            code_snippet = self.write_initial_code(action_unit)

        # Backup current state before execution
        start_time = time.time()
        self.backup_exec_state = copy.deepcopy(self.current_exec_state)
        print(f"Creating backup took {round(time.time() - start_time, 2)} seconds")

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

        self.clear_states(backup_only=True)

    def run_snippet(self, snippet, namespace):
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(snippet, namespace)
            return stdout.getvalue(), stderr.getvalue(), None
        except Exception as e:
            return stdout.getvalue(), stderr.getvalue(), e

    def review_and_correct(self, action_unit):
        round_counter = 0
        while round_counter < self.max_rounds:
            last_step = self.task_context.history[-1]
            stderr, error = last_step["stderr"], last_step["error"]
            # Send code for review
            feedback = self.send_code_for_review(
                action_unit
            )
            print(feedback)
            if "approved" in feedback.lower() and not stderr and not error:
                # if not self.action_units[action_unit_name].code_snippet:
                #     self.action_units[action_unit_name].code_snippet = self.task_context.history[-1]['code_snippet']
                # else:
                #     self.action_units[action_unit_name].code_snippet_buffer.append(
                #         self.task_context.history[-1]['code_snippet'])
                break

            # Restore state if code rejected and needs correction
            del self.current_exec_state
            start_time = time.time()
            self.current_exec_state = copy.deepcopy(self.backup_exec_state)
            print(f"Restoring from backup took {round(time.time() - start_time, 2)} seconds")
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


    def correct_code(self, action_unit, feedback):
        formatted_prompt = []
        if action_unit.name in self.one_history_only:
            formatted_prompt.append(self.task_context.display(mode="past"))
            formatted_prompt.append("The detailed task instructions are provided below. Sometimes the record of"
                                    "previous attempts are also provided")
        else:
            formatted_prompt.append(self.prepare_prompt(mode="past"))
        formatted_prompt.append(f"\nThe following code is the latest attempt for the current step and requires correction. "
                                "If previous attempts have been included in the task history above, their presence"
                                " does not indicate they succeeded or failed, though you can refer to their execution "
                                "outputs for context. \nOnly correct the latest code attempt provided below.\n")
        formatted_prompt.append(self.task_context.display(mode="last"))
        formatted_prompt.append(f"\nReviewer's feedback:\n{feedback}\n")
        formatted_prompt.append(f"Based on the reviewer's feedback, write a corrected version of the code\n")
        formatted_prompt.append(CODE_INDUCER)
        prompt = "\n".join(formatted_prompt)
        if action_unit.name not in self.need_biomedical_knowledge:
            response = self.ask(prompt)
        else:
            expert = DomainExpertAgent()
            response = expert.ask(prompt)

        code = self.parse_code(response)
        return code

    def send_code_for_review(self, action_unit):
        formatted_prompt = []
        if action_unit.name in self.one_history_only:
            formatted_prompt.append(self.task_context.display(mode="past"))
            formatted_prompt.append("The detailed task instructions are provided below. Sometimes the record of"
                                    "previous attempts are also provided")
        else:
            formatted_prompt.append(self.prepare_prompt(mode="past"))
        formatted_prompt.append("\n**TO DO: Code Review**\n"
                                "The following code is the latest attempt for the current step and requires your review. "
                                "If previous attempts have been included in the task history above, their presence"
                                " does not indicate they succeeded or failed, though you can refer to their execution "
                                "outputs for context. \nOnly review the latest code attempt provided below.\n")
        formatted_prompt.append(self.task_context.display(mode="last"))
        formatted_prompt.append("\nPlease review the code according to the following criteria:\n"
                                "1. *Functionality*: Can the code be successfully executed in the current setting?\n"
                                "2. *Conformance*: Does the code conform to the given instructions?\n"
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
            reviewer = CodeReviewerAgent()
        else:
            reviewer = DomainExpertAgent()
        response = reviewer.ask(prompt)
        return response

    def write_initial_code(self, action_unit):
        if action_unit.name not in self.one_history_only:
            prompt = self.prepare_prompt()
            prompt += f"\n**TO DO: Programming** \nNow that you've been familiar with the task setups and current status" \
                      f", please write the code following the instructions:\n\n{action_unit.instruction}\n"
            prompt = prompt + CODE_INDUCER
            if action_unit.name not in self.need_biomedical_knowledge:
                response = self.ask(prompt)
            else:
                expert = DomainExpertAgent()
                response = expert.ask(prompt)
        else:
            prompt = f"{action_unit.instruction}\n" \
                     f"{self.task_context.history[-1]['stdout']}\n\n" \
                     f"{CODE_INDUCER}"  # CODE_INDUCER 1 AND 2, TWO TYPES
            expert = DomainExpertAgent()
            response = expert.ask(prompt)
        code = self.parse_code(response)
        return code

    def run_task(self):
        self.clear_states()
        self.check_code_snippet_buffer()
        while True:
            action_unit_name = self.choose_action_unit()
            print('h\n'*10 + action_unit_name)
            if action_unit_name == "7":
                break
            self.execute_action_unit(action_unit_name)
        return self.task_context.history

    @staticmethod
    def parse_code(rsp):
        pattern = r"```python(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text


class CodeReviewerAgent:
    def __init__(self, role_prompt="You are a code reviewer in this project."):
        self.role_prompt = role_prompt

    def ask(self, prompt):
        return call_openai_gpt(prompt, self.role_prompt)


class DomainExpertAgent:
    def __init__(self, role_prompt="You are a domain expert in this biomedical research project."):
        self.role_prompt = role_prompt

    def ask(self, prompt):
        return call_openai_gpt(prompt, self.role_prompt)


if __name__ == "__main__":
    import os
    from openai import AzureOpenAI
    import pandas as pd
    import numpy as np
    import json
    import ast
    import re
    import subprocess
    import tqdm
    from utils.statistics import normalize_trait, read_json_to_dataframe


    ROLE_PROMPT: str = \
        """You are a data engineer in a biomedical research team. Your goal is to write code for wrangling biomedical data. 
In this project, you will focus on wrangling Series data from the GEO database."""

    GUIDELINES: str = \
        """In this project, your job is to implement statistical models to identify significant genes related 
to traits. 
There are three types of problems to solve. The steps you should take depend on the problem type. 
- Unconditional one-step regression. Identify the significant genes related to a trait.
- Conditional one-step regression. Identify the significant genes related to a trait while accounting for the influence of 'age' or 'gender'. When solving such a problem, this attribute should be available in the dataset.
- Conditional two-step regression. Identify the significant genes related to a trait while accounting for the influence of a condition, which is a trait other than age or gender. We will need to combine the information from two datasets, and conduct two-step regression. 

"""

    TOOLS: str = \
        """Tools:
"utils.preprocess" provides lots of well-developed helper functions for this project. Henceforth, it will be referred to as "the library." Please import and use functions from the library when possible. Below is the source code:
{utils_code}
        """

    SETUPS: str = \
        """
Setups:
1. Path to the raw GEO dataset for preprocessing: {in_cohort_dir}
2. Directory to save the preprocessed GEO dataset: {output_dir}

NOTICE 1: The overall preprocessing requires multiple code snippets, each based on the execution results of the previous snippets. 
Consequently, the instructions will be divided into multiple STEPS. Each STEP requires you to write a code snippet, and then the execution result will be given to you for either revision of the current STEP or progression to the next STEP.

NOTICE 2: Please import all functions and classes in 'utils.preprocess' at the beginning of the code:
'from utils.preprocess import *'

Based on the context, write code to follow the instructions.
"""
# Also, the trait name has been assigned to the string variable "trait." To make the code more general, please use the variable instead of the string literal.
    INSTRUCTION_STEP1: str = \
        """
        STEP1:
1. Identify the paths to the soft file and the matrix file, and assign them to the variables 'soft_file' and 'matrix_file'.
2. Read the matrix file to obtain background information about the dataset and sample characteristics data through the 'get_background_and_clinical_data' function from the library. For the input parameters to the function,
'background_prefixes' should be a list consisting of strings '!Series_title', '!Series_summary', and '!Series_overall_design'. 'clinical_prefixes' should be a list consisting of '!Sample_geo_accession' and '!Sample_characteristics_ch1'.
3. Obtain the sample characteristics dictionary from the clinical dataframe via the 'get_unique_values_by_row' function from the utils
4. Explicitly print out the all the background information and the sample characteristics dictionary. This information is for STEP2 to further write code.
        """

    INSTRUCTION_STEP2: str = \
        """
STEP 2: Dataset Analysis and Questions

As a biomedical research team, we are analyzing datasets to study the association between the human trait '{trait}' and genetic factors, considering the possible influence of age and gender. After searching the GEO database and parsing the matrix file of a series, STEP 1 has provided background information and sample characteristics data. Please review the output from STEP 1 and answer the following questions regarding this dataset:

1. Gene Expression Data Availability
   - Does this dataset contain gene expression data? (Note: Pure miRNA data is not suitable.)
     - If YES, set `is_gene_available` to `True`.
     - If NO, set `is_gene_available` to `False`.

2. Variable Availability and Data Type Conversion
   For each of the variables '{trait}', 'age', and 'gender', address the following points:

   **2.1 Data Availability**
     - Is there human data available for this variable?

   **2.2 Key Identification**
     - If data is available, identify the key in the sample characteristics dictionary where unique values of this variable are recorded. The key is an integer. The variable information might be explicitly recorded or inferred from the field with biomedical knowledge or understanding of the dataset background.

   **2.3 Data Type Conversion**
     - Choose an appropriate data type for each variable ('continuous' or 'binary').
     - If the data type is binary, convert values to 0 and 1.
     - Write a Python function to convert any given value of the variable to this data type. Typically, a colon (':') separates the header and the value in each cell, so ensure to extract the value after the colon in the function. Unknown values should be converted to `None`.
     - When data is not explicitly given but can be inferred, carefully observe the unique values in the sample characteristics dictionary and design a heuristic rule to convert those values into the chosen type. If you are 90% sure that some cases should be mapped to certain value, please do that instead of giving `None`. 
     - Name the functions `convert_trait`, `convert_age`, and `convert_gender`, respectively.

3. If `is_gene_available` is False or `trait_row` is None, it means the dataset is not usable. Then, please save this information and end the task. Hint: Follow the function call format strictly:
   ```python
   save_cohort_info(cohort, json_path, is_gene_available, trait_row is not None)
   ```
   Otherwise, please continue with the below steps.

4. Selecting Clinical Features
   - Use the `geo_select_clinical_features` function to obtain the output `selected_clinical_data` from the input dataframe. Follow the function call format strictly:
     ```python
     selected_clinical_data = geo_select_clinical_features(clinical_data, '{trait}', trait_row, convert_trait, age_row, convert_age, gender_row, convert_gender)
     ```
   - Note: `clinical_data` has been previously defined. Do not comment out the function call.

5. Previewing Selected Clinical Data
   - Use the `preview_df` function to print a preview of `selected_clinical_data`. Follow the function call format strictly:
     ```python
     print(preview_df(selected_clinical_data))
     ```
   - Do not comment out the function call.

[Output of STEP 1]
        """

    INSTRUCTION_STEP3: str = \
        """
STEP3:
1. Use the get_genetic_data function from the library to get the genetic_data from the matrix_file previously defined.
2. print the first 20 row ids for following step.
        """

    INSTRUCTION_STEP4: str = \
        """
STEP4:
Given the row headers (from STEP3) of a gene expression dataset in GEO. Based on your biomedical knowledge, are they human gene symbols, or are they some other identifiers that need to be mapped to gene symbols? Your answer should be concluded by starting a new line and strictly following this format:
requires_gene_mapping = (True or False)

[Output of STEP3]
        """

    INSTRUCTION_STEP5: str = \
        """
STEP5:
If requires_gene_mapping is True, do the following substeps 1-2; otherwise, skip STEP5.
    1. Use the 'get_gene_annotation' function from the library to get gene annotation data from the soft file.
    2. Use the 'preview_df' function from the library to preview the data and print out the results for the following step. Append the printing with a header like "Gene annotation"
        """

    INSTRUCTION_STEP6: str = \
        """
STEP6:
If requires_gene_mapping is True, do the following 1-3 substeps; otherwise, MUST SKIP the following substeps.
    1. When analyzing a gene expression dataset, we need to map some identifiers of genes to actual gene symbols. STEP3 prints out some of those identifiers, 
    and STEP5 prints out part of the gene annotation data converted to a Python dictionary. 
    Please read the dictionary and decide which key stores the same kind of identifiers as in STEP3, and which key stores the gene symbols. Please strictly follow this format in your answer:
    identifier_key = 'key_name1'
    gene_symbol_key = 'key_name2'
    2. Get the gen mapping with the 'get_gene_mapping' function from the library. 
    3. Get the gene data with the 'apply_gene_mapping' function from the library. 
Please DO the following substeps 4-8 anyway:   
4. Normalize the obtained gene data with the 'normalize_gene_symbols_in_index' function from the library. 
5. Merge the clinical and genetic data with the 'geo_merge_clinical_genetic_data' function from the library, and assign the merged data to a variable 'merged_data'.
6. Determine whether the trait '{trait}' and some demographic attributes in the data is severely biased, and remove biased attributes with the 'judge_and_remove_biased_features' function from the library.
7. Save the cohort information with the 'save_cohort_info' function from the library. Hint: set the 'json_path' variable to '{json_path}', and assuming 'is_trait_biased' indicates whether the trait is biased, use the following code to save the cohort information: save_cohort_info(cohort, json_path, True, True, is_trait_biased, merged_data).
8. If the trait in the data is not severely biased (regardless of whether the other attributes are biased), save the merged data to a csv file, in the path '{out_data_file}'. Otherwise, you must not save it.
        """

    all_traits = pd.read_csv("all_traits.csv")["Trait"].tolist()
    all_traits = [normalize_trait(t) for t in all_traits]
    input_dir = '/media/techt/DATA/GEO' if os.path.exists('/media/techt/DATA/GEO') else '../DATA/GEO'

    output_root = './output/preprocessed'
    version = 'gs1'
    version_dir = os.path.join(output_root, version)

    utils_code = "".join(open("utils/preprocess.py", 'r').readlines())
    tools = TOOLS.format(utils_code=utils_code)
    for index, trait in enumerate(all_traits):
        try:
            in_trait_dir = os.path.join(input_dir, trait)
            output_dir = os.path.join(version_dir, trait)
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, "cohort_info.json")

            if not os.path.isdir(in_trait_dir):
                print(f"Trait directory not found: {in_trait_dir}")
                continue
            for cohort in os.listdir(in_trait_dir):
                in_cohort_dir = os.path.join(in_trait_dir, cohort)
                if not os.path.isdir(in_cohort_dir):
                    print(f"Cohort directory not found: {in_cohort_dir}")
                    continue
                setups = SETUPS.format(in_cohort_dir=in_cohort_dir, output_dir=output_dir)
                out_data_file = os.path.join(output_dir, f"{cohort}.csv")

                action_units = [
                    ActionUnit("1", INSTRUCTION_STEP1),
                    ActionUnit("2", INSTRUCTION_STEP2.format(trait=trait)),
                    ActionUnit("3", INSTRUCTION_STEP3),
                    ActionUnit("4", INSTRUCTION_STEP4),
                    ActionUnit("5", INSTRUCTION_STEP5),
                    ActionUnit("6", INSTRUCTION_STEP6.format(trait=trait, json_path=json_path,
                                                       out_data_file=out_data_file)),
                    ActionUnit("7", "Task completed, you don't need to write any code.")
                ]

            geo_agent = GEOAgent(ROLE_PROMPT, GUIDELINES, tools, setups, action_units)
            task_context = geo_agent.run_task()
        except Exception as e:
            print(e)
            continue

