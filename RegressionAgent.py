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


class DataScientistAgent:
    def __init__(self, guidelines, action_units):
        self.guidelines = guidelines
        self.action_units = action_units
        self.task_context = []
        """TO DO: 
        please define a TaskContext class, that holds the all staffs, and handles common functions such as
        the display of the context, concatenation of previous code, etc. The context should be formatted in a nice way 
        for humans to read. Just think of how a Jupyter notebook displays the markdown cells, code cells, and output 
        nicely. You can use special characters or even ascii art to show the structure of the context. Of course, our
        design is different from jupyter notebook. We need things like the index of the current step, the name of
        the action unit chosen, the instruction of the action unit, the code, the stdout, and stderr, for each step.
        """

        self.code_snippet_buffer = {unit: [] for unit in action_units.keys()}
        self.current_state = {}

    def check_code_snippet_buffer(self):
        for unit, buffer in self.code_snippet_buffer.items():
            if len(buffer) == 3:
                modified_snippet = self.aggregate_code_snippets(buffer)
                self.action_units[unit]['code_snippet'] = modified_snippet
                buffer.clear()

    def aggregate_code_snippets(self, buffer):
        """TO DO:
        I didn't mean "aggregate" in the literal way.
        The agent should read and compare the different versions of the code snippet, including the original one and
        the different revised versions in the buffer, to get a modified version as the new code snippet for this
        action unit.
        """
        raise NotImplementedError

    def choose_action_unit(self):
        """TO DO:
        Analyze current task context to choose the next action unit. Use a prompt to ask the agent itself like
        the below, but you should format it much more nicely.
        prompt = f"You are a data scientist agent. Here is the guideline for your task {guideline}. Here is your current
        task context: {self.context.history}. Here are the action units you can choose, with their names and
        instructions: {}
        Based on these information, please choose one and only one action unit. Please only answer the name of the unit.
        Your answer:\n"
        """
        raise NotImplementedError

    def execute_action_unit(self, action_unit):
        code_snippet = self.action_units[action_unit]['code_snippet']
        if not code_snippet:
            code_snippet = self.write_initial_code(action_unit)

        self.task_context.append({
            'action_unit': action_unit,
            'instruction': self.action_units[action_unit]['instruction'],
            'code_snippet': code_snippet
        })

        stdout, stderr, error = self.run_snippet(code_snippet, self.current_state)

        self.task_context[-1].update({
            'stdout': stdout,
            'stderr': stderr
        })

        if error:
            self.task_context[-1]['error'] = str(error)

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
            self.code_snippet_buffer[action_unit].append(self.task_context[-1]['code_snippet'])
            if len(self.code_snippet_buffer[action_unit]) >= 3:
                self.check_code_snippet_buffer()
                self.reset_and_retry(action_unit)
            else:
                new_code_snippet = self.rewrite_code(action_unit, error)
                self.task_context[-1]['code_snippet'] = new_code_snippet
                self.execute_action_unit(action_unit)
        else:
            self.action_units[action_unit]['completed'] = True

    def rewrite_code(self, action_unit, error):
        prompt = f"Rewrite the following code to fix the error:\n\n{self.task_context[-1]['code_snippet']}\n\nError: {error}"
        response = call_openai_gpt(prompt)
        return response

    def concatenate_snippets(self, up_to_index):
        return "\n".join([step['code_snippet'] for step in self.task_context[:up_to_index + 1]])

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
        return self.task_context


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
