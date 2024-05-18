import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import json
from openai import AzureOpenAI


client = AzureOpenAI(
    api_key="57983a2e88fa4d6b81205a8d55d9bd46",
    api_version="2023-10-01-preview",
    azure_endpoint="https://haoyang2.openai.azure.com/"
)


def call_openai_gpt(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
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

    def add_step(self, action_unit, instruction, code_snippet, stdout, stderr, error=None):
        step = {
            'index': self.current_step,
            'action_unit': action_unit,
            'instruction': instruction,
            'code_snippet': code_snippet,
            'stdout': stdout,
            'stderr': stderr
        }
        if error:
            step['error'] = error
        self.history.append(step)
        self.current_step += 1

    def display(self):
        for step in self.history:
            print(f"Step {step['index']}: {step['action_unit']}")
            print(f"Instruction: {step['instruction']}")
            print("Code:")
            print(step['code_snippet'])
            print("Output:")
            print(step['stdout'])
            if step['stderr']:
                print("Errors:")
                print(step['stderr'])
            if 'error' in step:
                print("Execution Error:")
                print(step['error'])
            print("="*50)

    def concatenate_snippets(self, up_to_index):
        return "\n".join([step['code_snippet'] for step in self.history[:up_to_index + 1]])


class DataScientistAgent:
    def __init__(self, guidelines, action_units):
        self.guidelines = guidelines
        self.action_units = action_units
        self.task_context = TaskContext()
        self.code_snippet_buffer = {unit: [] for unit in action_units.keys()}
        self.current_state = {}

    def check_code_snippet_buffer(self):
        for unit, buffer in self.code_snippet_buffer.items():
            if len(buffer) == 3:
                modified_snippet = self.aggregate_code_snippets(unit)
                self.action_units[unit]['code_snippet'] = modified_snippet
                buffer.clear()

    def aggregate_code_snippets(self, unit):
        original_snippet = self.action_units[unit]['code_snippet']
        revised_versions = self.code_snippet_buffer[unit]
        prompt = f"Here is the original code snippet:\n\n{original_snippet}\n\nHere are the revised versions:\n\n{'\n\n'.join(revised_versions)}\n\nPlease combine these to create a modified version that addresses the errors."
        response = call_openai_gpt(prompt)
        return response

    def choose_action_unit(self):
        prompt = f"You are a data scientist agent. Here is the guideline for your task: {self.guidelines}. Here is your current task context:\n\n{self.task_context.display()}\n\nHere are the action units you can choose, with their names and instructions:\n\n{json.dumps(self.action_units, indent=4)}\n\nBased on this information, please choose one and only one action unit. Please only answer the name of the unit.\n\nYour answer:"
        response = call_openai_gpt(prompt)
        return response.strip()

    def execute_action_unit(self, action_unit):
        code_snippet = self.action_units[action_unit]['code_snippet']
        if not code_snippet:
            code_snippet = self.write_initial_code(action_unit)

        stdout, stderr, error = self.run_snippet(code_snippet, self.current_state)

        self.task_context.add_step(
            action_unit=action_unit,
            instruction=self.action_units[action_unit]['instruction'],
            code_snippet=code_snippet,
            stdout=stdout,
            stderr=stderr,
            error=str(error) if error else None
        )

        if error:
            self.handle_execution_result(action_unit, error=str(error))
        else:
            self.action_units[action_unit]['completed'] = True

    def run_snippet(self, snippet, namespace):
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(snippet, namespace)
            return stdout.getvalue(), stderr.getvalue(), None
        except Exception as e:
            return stdout.getvalue(), stderr.getvalue(), e

    def handle_execution_result(self, action_unit, error=None):
        if error:
            self.code_snippet_buffer[action_unit].append(self.task_context.history[-1]['code_snippet'])
            if len(self.code_snippet_buffer[action_unit]) >= 3:
                self.check_code_snippet_buffer()
                self.reset_and_retry(action_unit)
            else:
                new_code_snippet = self.rewrite_code(action_unit, error)
                self.action_units[action_unit]['code_snippet'] = new_code_snippet
                self.execute_action_unit(action_unit)
        else:
            self.action_units[action_unit]['completed'] = True

    def rewrite_code(self, action_unit, error):
        prompt = f"Rewrite the following code to fix the error:\n\n{self.task_context.history[-1]['code_snippet']}\n\nError: {error}"
        response = call_openai_gpt(prompt)
        return response

    def reset_and_retry(self, action_unit):
        self.current_state.clear()
        concatenated_code = self.task_context.concatenate_snippets(up_to_index=len(self.task_context.history) - 1)
        stdout, stderr, error = self.run_snippet(concatenated_code, self.current_state)

        if error:
            self.task_context.add_step(
                action_unit=action_unit,
                instruction=self.action_units[action_unit]['instruction'],
                code_snippet=concatenated_code,
                stdout=stdout,
                stderr=stderr,
                error=str(error)
            )
            self.execute_action_unit(action_unit)
        else:
            self.action_units[action_unit]['completed'] = True

    def write_initial_code(self, action_unit):
        prompt = f"Write the initial code for the following instruction:\n\n{self.action_units[action_unit]['instruction']}"
        response = call_openai_gpt(prompt)
        return response

    def run_task(self):
        self.check_code_snippet_buffer()
        while True:
            action_unit = self.choose_action_unit()
            if action_unit == "task_completed":
                break
            self.execute_action_unit(action_unit)
        self.task_context.display()
        return self.task_context.history


class CodeReviewerAgent:
    def review_code(self, code_snippet):
        prompt = f"Review the following code snippet:\n\n{code_snippet}"
        response = call_openai_gpt(prompt)
        return response


# Example usage:
guidelines = "High-level guidelines for performing data analysis tasks."
action_units = {
    "load_data": {"instruction": "Load the dataset.", "code_snippet": "", "completed": False},
    "preprocess_data": {"instruction": "Preprocess the dataset.", "code_snippet": "", "completed": False},
    "train_model": {"instruction": "Train the machine learning model.", "code_snippet": "", "completed": False},
    "evaluate_model": {"instruction": "Evaluate the model performance.", "code_snippet": "", "completed": False}
}

data_scientist = DataScientistAgent(guidelines, action_units)
task_context = data_scientist.run_task()

for step in task_context:
    print(json.dumps(step, indent=4))
