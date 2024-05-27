import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import json
import re
from openai import AzureOpenAI

# Azure OpenAI client setup
client = AzureOpenAI(
    api_key="57983a2e88fa4d6b81205a8d55d9bd46",
    api_version="2023-10-01-preview",
    azure_endpoint="https://haoyang2.openai.azure.com/"
)

CODE_INDUCER: str = \
"""
Return ```python your_code_here ``` with NO other texts. your_code_here is a placeholder.
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

    def add_step(self, action_unit_name, instruction, code_snippet, stdout, stderr, error=None):
        step = {
            'index': self.current_step,
            'action_unit_name': action_unit_name,
            'instruction': instruction,
            'code_snippet': code_snippet,
            'stdout': stdout,
            'stderr': stderr
        }
        if error:
            step['error'] = error
        self.history.append(step)
        self.current_step += 1

    def display(self, last_only=False):
        contxt_to_display = self.history[-1:] if last_only else self.history
        formatted_context = []
        for step in contxt_to_display:
            formatted_context.append(f"STEP {step['index']}")
            formatted_context.append(f"Chosen action unit: {step['action_unit_name']}")
            formatted_context.append(f"Instruction: {step['instruction']}")
            formatted_context.append(f"Code:\n{step['code_snippet']}")
            formatted_context.append(f"Output:\n{step['stdout']}")
            if step['stderr']:
                formatted_context.append(f"Errors:\n{step['stderr']}")
            if 'error' in step:
                formatted_context.append(f"Execution Error:\n{step['error']}")
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


class DataScientistAgent:
    def __init__(self, role_prompt, tools, setups, guidelines, action_units, max_rounds=3):
        self.role_prompt = role_prompt
        self.tools = tools
        self.setups = setups
        self.guidelines = guidelines
        self.action_units = {unit.name: unit for unit in action_units}
        self.task_context = TaskContext()
        self.current_state = {}
        self.max_rounds = max_rounds

    def ask(self, prompt):
        return call_openai_gpt(prompt, self.role_prompt)

    def prepare_prompt(self, include_tools=True):
        formatted_prompt = []
        if include_tools:
            formatted_prompt.append("To help you get prepared, I will provide you with the general guidelines for "
                                    "the task, the task setups, the function tools, and the current context including the"
                                    "instructions, code, and execution output of all previous steps taken.")
        else:
            formatted_prompt.append("To help you get prepared, I will provide you with the general guidelines for "
                                    "the task, the task setups, and the current context including the"
                                    "instructions, code, and execution output of all previous steps taken.")
        formatted_prompt.append(f"General guidelines: \n{self.guidelines}")
        formatted_prompt.append(f"Task setups: \n{self.setups}")
        if include_tools:
            formatted_prompt.append(f"Function tools: \n{self.tools}")
        formatted_prompt.append(f"Task context: \n{self.task_context.display()}\n")

        return "\n".join(formatted_prompt)


    def check_code_snippet_buffer(self):
        for unit in self.action_units.values():
            if len(unit.code_snippet_buffer) == 3:
                modified_snippet = self.aggregate_code_snippets(unit)
                unit.code_snippet = modified_snippet
                unit.code_snippet_buffer.clear()

    def aggregate_code_snippets(self, unit):
        original_snippet = unit.code_snippet
        revised_versions = unit.code_snippet_buffer
        formatted_versions = [f"[version {i + 1}]: \n{version}" for i, version in enumerate(revised_versions)]
        formatted_versions = '\n\n'.join(formatted_versions)
        prompt = (
            f"We want to perform a task, with the gold guideline given below:\n\n"
            f"{self.guidelines}\n\n"
            f"Now we want to improve the code for a specific subtask. Below is a description of this subtask:\n\n"
            f"{unit.instruction}\n\n"
            f"Here is the original code snippet which didn't seem to work:\n\n"
            f"{original_snippet}\n\n"
            f"Here are the candidate revised versions that worked:\n\n"
            f"{formatted_versions}\n\n"
            f"Please read the candidate revised versions to understand the revisions made, either select the best one or combine their advantages, to write a single revised version."
        )
        prompt = prompt + CODE_INDUCER
        response = self.ask(prompt)
        code = self.parse_code(response)
        return code

    def choose_action_unit(self):
        action_units_formatted = "\n".join([str(unit) for unit in self.action_units.values()])
        prompt = f"Here is the general guideline for your task:\n\n{self.guidelines}\n\n" \
                 f"Here is your current task context:\n\n{self.task_context.display()}\n\n" \
                 f"Here are the action units you can choose for the next step, with their names and instructions:\n\n{action_units_formatted}\n\n" \
                 f"Based on this information, please choose one and only one action unit. Please only answer the name of the unit, chosen from the below list" \
                 f"{[unit.name for unit in self.action_units.values()]}" \
                 f"\n\nYour answer:"
        response = self.ask(prompt)
        return response.strip()

    def execute_action_unit(self, action_unit_name):
        action_unit = self.action_units[action_unit_name]
        code_snippet = action_unit.code_snippet
        if not code_snippet:
            code_snippet = self.write_initial_code(action_unit)

        stdout, stderr, error = self.run_snippet(code_snippet, self.current_state)

        self.task_context.add_step(
            action_unit_name=action_unit_name,
            instruction=action_unit.instruction,
            code_snippet=code_snippet,
            stdout=stdout,
            stderr=stderr,
            error=str(error) if error else None
        )
        print(self.task_context.display(last_only=True))

        if stderr or error or not action_unit.code_snippet:
            self.review_and_correct(action_unit_name)

    def run_snippet(self, snippet, namespace):
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(snippet, namespace)
            return stdout.getvalue(), stderr.getvalue(), None
        except Exception as e:
            return stdout.getvalue(), stderr.getvalue(), e

    def review_and_correct(self, action_unit_name):
        round_counter = 0
        while round_counter < self.max_rounds:
            # Send code for review
            reviewer = CodeReviewerAgent("You are a code reviewer in this project.")
            full_context = self.task_context.display()
            feedback = reviewer.review_code(self.guidelines, full_context, self.task_context.history[-1])

            if "approved" in feedback.lower():
                if not self.action_units[action_unit_name].code_snippet:
                    self.action_units[action_unit_name].code_snippet = self.task_context.history[-1]['code_snippet']
                else:
                    self.action_units[action_unit_name].code_snippet_buffer.append(
                        self.task_context.history[-1]['code_snippet'])
                break

            # Correct the code based on feedback
            new_code_snippet = self.correct_code(action_unit_name, feedback)
            stdout, stderr, error = self.run_snippet(new_code_snippet, self.current_state)

            self.task_context.add_step(
                action_unit_name=f"Debugging Attempt {round_counter}",
                instruction="(Omitted)",
                code_snippet=new_code_snippet,
                stdout=stdout,
                stderr=stderr,
                error=str(error) if error else None
            )
            print(self.task_context.display(last_only=True))

            # if not stderr and not error:
            #     self.action_units[action_unit_name].code_snippet_buffer.append(new_code_snippet)
            #     break

            round_counter += 1

    def correct_code(self, action_unit_name, feedback):
        formatted_prompt = []
        formatted_prompt.append(f"I will provide you with detailed information about the task. Please read relevant "
                                f"information and correct the code for the last step.")
        formatted_prompt.append(self.prepare_prompt())
        formatted_prompt.append(f"Reviewer's feedback\n: {feedback}")
        formatted_prompt.append(f"Based on the reviewer's feedback, write a corrected version of the code below: \n"
                                f"{self.task_context.history[-1]['code_snippet']}")
        formatted_prompt.append(CODE_INDUCER)
        prompt = "\n".join(formatted_prompt)
        response = self.ask(prompt)
        code = self.parse_code(response)
        return code

    def write_initial_code(self, action_unit):
        prompt = self.prepare_prompt()
        prompt += f"\nNow that you've been familiar with the task setups and current status, please write the code " \
                 f"following the instructions:\n\n{action_unit.instruction}\n"
        prompt = prompt + CODE_INDUCER
        response = self.ask(prompt)
        code = self.parse_code(response)
        return code

    def run_task(self):
        self.check_code_snippet_buffer()
        while True:
            action_unit_name = self.choose_action_unit()
            if action_unit_name == "task_completed":
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
    def __init__(self, role_prompt):
        self.role_prompt = role_prompt
    def ask(self, prompt):
        return call_openai_gpt(prompt, self.role_prompt)
    def review_code(self, guideline, full_context, step):
        prompt = f"{guideline}" \
                 f"Review the following code snippet in the context of the task:\n\nTask context:\n{full_context}\n\n" \
                 f"Code:\n{step['code_snippet']}\n\n" \
                 f"Execution result:\nStdout:\n{step['stdout']}\nStderr:\n{step['stderr']}\n\n" \
                 f"Your feedback:"
        response = self.ask(prompt)
        print(response)
        return response

    def format_context(self, context):
        formatted_context = []
        for step in context.history:
            formatted_context.append(f"STEP {step['index']}: {step['action_unit_name']}")
            formatted_context.append(f"Instruction: {step['instruction']}")
            formatted_context.append("Code:")
            formatted_context.append(step['code_snippet'])
            formatted_context.append("Output:")
            formatted_context.append(step['stdout'])
            if step['stderr']:
                formatted_context.append("Errors:")
                formatted_context.append(step['stderr'])
            if 'error' in step:
                formatted_context.append("Execution Error:")
                formatted_context.append(step['error'])
            formatted_context.append("=" * 50)
        return "\n".join(formatted_context)


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

    pairs_df = pd.read_csv("trait_condition_pairs.csv")
    pair_rows = list(pairs_df.iterrows())
    all_pairs = []
    seen_traits = set()
    for i, row in pair_rows:
        trait, condition = row['Trait'], row['Condition']
        if trait not in seen_traits:
            seen_traits.add(trait)
            all_pairs.append((trait, None))
        all_pairs.append((trait, condition))

    gene_info_path = './trait_related_genes.csv'
    data_root = '/home/techt/Desktop/a4s/gold_subset'
    output_root = './output_agent'

    role_prompt = """You are a statistician in a biomedical research team, and your main goal is to write code to do statistical 
        analysis on biomedical datasets.
        In this project, you will explore gene expression datasets to identify the significant genes related to a trait, 
        optionally controlling for a condition."""

    guidelines = """In this project, your job is to implement statistical models to identify significant genes related 
to traits. 
There are three types of problems to solve. The steps you should take depend on the problem type. 
- Unconditional one-step regression. Identify the significant genes related to a trait.
- Conditional one-step regression. Identify the significant genes related to a trait while accounting for the influence of 'age' or 'gender'. When solving such a problem, this attribute should be available in the dataset.
- Conditional two-step regression. Identify the significant genes related to a trait while accounting for the influence of a condition, which is a trait other than age or gender. We will need to combine the information from two datasets, and conduct two-step regression. 

"""

    TOOLS: str = \
        """
        Tools: 
        In "utils.statistics", there are lots of well-developed helper functions for this project. Please import 
        and use them when possible. Hereafter I will call it "the library". Below is the source code.
        {utils_code}
        """

    SETUPS : str = \
        """
        Setups:
        1. All input data are stored in the directory: '{data_root}'.
        2. The output should be saved to the directory '{output_root}', under a subdirectory named after the trait.
        3. External knowledge about genes related to each trait is available in a file '{gene_info_path}'.

        NOTICE1: Please import all the functions in 'utils.statistics' at the beginning of the code, and feel free to use '*' in the import statement.
        NOTICE2: The overall preprocessing requires multiple code snippets and each code snippet is based on the execution results of the previous code snippets. 
        Consequently, the instruction will be divided into multiple STEPS, each STEP requires you to write a code snippet, then the execution result will be given to you for either revision of the current STEP or go to the next STEP.

        Based on the context, write code to follow the instructions.
        """

    UNCONDITIONAL_ONE_STEP_PROMPT = """
    Instruction: Write code to solve the following research question: What are the genetic factors related to the trait 
    '{trait}'?
    Based on the context and the following instructions, write code that is elegant and easy to read.
    1. Select the best input data about the trait into a dataframe, and load the data.
    2. Remove the columns 'Age' and 'Gender' if either is present.
    3. Select the data in relevant columns for regression analysis. We need numpy arrays X and Y. Y is the trait data from the column '{trait}', and X is the rest of the data.
    4. Check whether the feature X shows batch effect. Hint: you may use the 'detect_batch_effect' function from the library.
    5. Select appropriate models based on whether the dataset has batch effect. If yes, use an LMM (Linear Mixed Model); 
       Otherwise, use a Lasso model.
    6. Perform a hyperparameter search on integer powers of 10 from 1e-6 to 1e0 (inclusive). Record the best hyperparameter setting for the chosen model, and the cross-validation performance. Hint: please use the tune_hyperparameters() function from the library.
    7. Normalize X to have a mean of 0 and standard deviation of 1.
    8. Train a model with the best hyperparameter on the whole dataset.
    9. Interpret the trained model to identify the effect of the condition and significant genes. Hint: You may use the 'interpret_result' function from the library, and use the output_dir given.
    10. Save the model output and cross-validation performance. Hint: you may use the 'save_result' function from the library
            """

    CONDITIONAL_ONE_STEP_PROMPT = """
    Instruction: Write code to solve the following research question: What are the genetic factors related to the trait 
    '{trait}' when considering the influence of the condition '{condition}'?
    Based on the context and the following instructions, write code that is elegant and easy to read.
    1. Select the best input data about the trait into a dataframe, and load the data.
    2. We need only one condition from 'Age' and 'Gender'. Remove the redundant column if present.
    3. Select the data in relevant columns for regression analysis. We need three numpy arrays X, Y and Z. Y is the trait data from the column '{trait}', Z is the condition data from the column '{condition}', and X is the rest of the data.
    4. Check whether the feature X shows batch effect. Hint: you may use the 'detect_batch_effect' function from the library.
    5. Select appropriate models based on whether the dataset has batch effect. If yes, use an LMM (Linear Mixed Model); 
       Otherwise, use a Lasso model.
    6. Perform a hyperparameter search on integer powers of 10 from 1e-6 to 1e0 (inclusive). Record the best hyperparameter setting for the chosen model, and the cross-validation performance. Hint: please use the tune_hyperparameters() function from the library.
    7. Normalize the X and Z to have a mean of 0 and standard deviation of 1.
    8. Train a model with the best hyperparameter on the whole dataset. The model should conduct residualization to account for the confounder Z.
    9. Interpret the trained model to identify the effect of the condition and significant genes. Hint: You may use the 'interpret_result' function from the library, and use the output_dir given.
    10. Save the model output and cross-validation performance. Hint: you may use the 'save_result' function from the library
            """

    TWO_STEP_PROMPT = """
    Instruction: Write code to solve the following research question: What are the genetic factors related to the trait 
    '{trait}' when considering the influence of the condition '{condition}'?
    When we don't have data about the trait and condition from the same group of people, we can still solve the problem by
    two-step regression. With two datasets for the trait and the condition respectively, we find common gene features among
    them that are known related to the condition. We then use those those genes to fit a regression model on the condition 
    dataset, to predict the condition of samples in the trait dataset. Then we can do regression on the trait dataset to 
    solve the question.

    Below are more detailed instructions. Based on the context and the instructions, write code that is elegant and easy to read.
    1. Select the best input data about the trait and the condition into two separate dataframe, and load the data and common gene regressors.
    2. From the trait dataset, remove the columns 'Age' and 'Gender' if either is present.
    3. From the condition dataframe, select the columns corresponding to the gene regressors as 'X_condition', and the column corresponding to the condition value as 'Y_condition', and convert them to numpy arrays.
    4. Determine the data type of the condition, which is either 'binary' or 'continuous', by seeing whether the array of condition values has two unique values.
    ## The first step regression
    5. Please choose an appropriate regression model for the condition. 
       - If the condition is a binary variable, then use the LogisticRegression model. Use L1 penalty if 'X_condition' has more columns than rows.
       - If the condition is a continuous variable, then choose Lasso or LinearRegression depending on whether 'X_condition' has more columns than rows.
       Normalize 'X_condition' to a mean of 0 and std of 1. With the model you chose, fit the model on 'normalized_X_condition' and 'Y_condition'
    6. From the trait dataframe, select the columns corresponding to the common gene regressors to get a numpy array, and normalize it to a mean of 0 and std of 1.
    7. With the model trained in Step 5, predict the condition of the samples in the trait dataframe based on the normalized gene regressors. 
      If the condition is a continuous variable, use the predict() method of the model to get the predicted values of the condition; otherwise, use the predict_proba() 
      method and select the column corresponding to the positive label, to get the predicted probability of the condition being true. 
      Add a column named {condition} to the trait dataframe, storing predicted condition values. Drop the columns about the common gene regressors.
    ## The second step regression
    8. From the trait dataframe, select the data in relevant columns for regression analysis. We need three numpy arrays X, Y and Z. Y is the trait data from the column '{trait}', Z is the condition data from the column '{condition}', and X is the rest of the data. We want to analyze and find the genetic factors related to the trait when considering the influence of the condition.
    9. Check whether the feature X shows batch effect. Hint: you may use the 'detect_batch_effect' function from the library.
    10. Select appropriate models based on whether the dataset has batch effect. If yes, use an LMM (Linear Mixed Model); 
       Otherwise, use a Lasso model.
    11. Perform a hyperparameter search on integer powers of 10 from 1e-6 to 1e0 (inclusive). Record the best hyperparameter setting for the chosen model, and the cross-validation performance. Hint: please use the tune_hyperparameters() function from the library.
    12. Normalize the X and Z to have a mean of 0 and standard deviation of 1. Hint: you may use the 'normalize_data' function from the library to normalize X and Z in two seperate lines.
    13. Train a model with the best hyperparameter on the whole dataset. The model should conduct residualization to account for the confounder Z.
    14. Interpret the trained model to identify the effect of the condition and significant genes. Hint: You may use the 'interpret_result' function from the library, and use the output_dir given.
    15. Save the model output and cross-validation performance. Hint: you may use the 'save_result' function from the library
            """

    utils_code = "".join(open("utils/statistics.py", 'r').readlines())

    for index, pair in enumerate(all_pairs):
        trait, condition = pair
        if index < 3: continue
        if condition is None or condition in ['Age', 'Gender'] or 'Endometriosis' in [trait, condition]: continue
        #if condition is not None: continue
        question = f"\nThe question to solve is: What are the genetic factors related to the trait '{trait}' when considering the influence of the " \
                   f"condition '{condition}'?"
        print(trait, condition)
        # if trait != 'Adrenocortical_Cancer' or condition != 'Anxiety_disorder': continue
        tools = TOOLS.format(utils_code=utils_code)
        setups = SETUPS.format(data_root=data_root, output_root=output_root, gene_info_path=gene_info_path)

        action_units = [
            ActionUnit("unconditional one-step regression", UNCONDITIONAL_ONE_STEP_PROMPT.format(trait=trait, condition=condition,
                                                     )),
            ActionUnit("conditional one-step regression", CONDITIONAL_ONE_STEP_PROMPT.format(trait=trait, condition=condition,
                                                     )),
            ActionUnit("two-step regression", TWO_STEP_PROMPT.format(trait=trait, condition=condition,
                                                     )),
            ActionUnit("task_completed", "Task completed, you don't need to write any code.")
        ]

        data_scientist = DataScientistAgent(role_prompt, tools, setups, guidelines + question, action_units)
        task_context = data_scientist.run_task()

        for step in task_context:
            print(json.dumps(step, indent=4))
