import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import json
from openai import AzureOpenAI

# Azure OpenAI client setup
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
            print("=" * 50)

    def concatenate_snippets(self, up_to_index):
        return "\n".join([step['code_snippet'] for step in self.history[:up_to_index + 1]])


class DataScientistAgent:
    def __init__(self, guidelines, action_units, max_rounds=3):
        self.guidelines = guidelines
        self.action_units = action_units
        self.task_context = TaskContext()
        self.code_snippet_buffer = {unit: [] for unit in action_units.keys()}
        self.current_state = {}
        self.max_rounds = max_rounds

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
            code_snippet = self.write_initial_code(action_unit)  # TO DO: initialize the action unit with the code here?

        stdout, stderr, error = self.run_snippet(code_snippet, self.current_state)

        self.task_context.add_step(
            action_unit=action_unit,
            instruction=self.action_units[action_unit]['instruction'],
            code_snippet=code_snippet,
            stdout=stdout,
            stderr=stderr,
            error=str(error) if error else None
        )

        if stderr or error:
            self.review_and_correct(action_unit)

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
            # Send code for review
            reviewer = CodeReviewerAgent()
            last_step = self.task_context.history[-1]
            feedback = reviewer.review_code(last_step['code_snippet'])

            # Check for approval or provide feedback
            if "approved" in feedback.lower():
                self.code_snippet_buffer[action_unit].append(last_step['code_snippet'])
                break

            # Correct the code based on feedback
            new_code_snippet = self.correct_code(action_unit, feedback)
            stdout, stderr, error = self.run_snippet(new_code_snippet, self.current_state)

            self.task_context.add_step(
                action_unit=action_unit,
                instruction=self.action_units[action_unit]['instruction'],
                code_snippet=new_code_snippet,
                stdout=stdout,
                stderr=stderr,
                error=str(error) if error else None
            )

            if not stderr and not error:
                self.code_snippet_buffer[action_unit].append(new_code_snippet)
                break

            round_counter += 1

    def correct_code(self, action_unit, feedback):
        prompt = f"Based on the following feedback, correct the code:\n\nFeedback: {feedback}\n\nCode:\n{self.task_context.history[-1]['code_snippet']}"
        response = call_openai_gpt(prompt)
        return response

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
    "load_data": {"instruction": "Load the dataset.", "code_snippet": ""},
    "preprocess_data": {"instruction": "Preprocess the dataset.", "code_snippet": ""},
    "train_model": {"instruction": "Train the machine learning model.", "code_snippet": ""},
    "evaluate_model": {"instruction": "Evaluate the model performance.", "code_snippet": ""}
}

data_scientist = DataScientistAgent(guidelines, action_units)
task_context = data_scientist.run_task()

for step in task_context:
    print(json.dumps(step, indent=4))
