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

    def display(self):
        formatted_context = []
        for step in self.history:
            formatted_context.append(f"STEP {step['index']}")
            formatted_context.append(f"Chosen action unit: {step['action_unit_name']}")
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
    def __init__(self, guidelines, action_units, max_rounds=3):
        self.guidelines = guidelines
        self.action_units = {unit.name: unit for unit in action_units}
        self.task_context = TaskContext()
        self.current_state = {}
        self.max_rounds = max_rounds

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
        prompt = (
            f"We want to perform a task, with the gold guideline given below:\n\n{self.guidelines}\n\n"
            f"Now we want to improve the code for a specific subtask. Below is a description of this subtask:\n\n"
            f"{unit.instruction}\n\n"
            f"Here is the original code snippet which didn't seem to work:\n\n{original_snippet}\n\n"
            f"Here are the candidate revised versions that worked:\n\n{'\n\n'.join(formatted_versions)}\n\n"
            f"Please read the candidate revised versions to understand the revisions made, either select the best one or combine their advantages, to write a single revised version."
        )
        response = call_openai_gpt(prompt)
        return response

    def choose_action_unit(self):
        action_units_formatted = "\n".join([str(unit) for unit in self.action_units.values()])
        prompt = f"You are a data scientist agent. Here is the general guideline for your task:\n\n{self.guidelines}\n\n" \
                 f"Here is your current task context:\n\n{self.task_context.display()}\n\n" \
                 f"Here are the action units you can choose for the next step, with their names and instructions:\n\n{action_units_formatted}\n\n" \
                 f"Based on this information, please choose one and only one action unit. Please only answer the name of the unit.\n\nYour answer:"
        response = call_openai_gpt(prompt)
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
            reviewer = CodeReviewerAgent()
            full_context = self.task_context.display()
            feedback = reviewer.review_code(full_context, self.task_context.history[-1])

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

            if not stderr and not error:
                self.action_units[action_unit_name].code_snippet_buffer.append(new_code_snippet)
                break

            round_counter += 1

    def correct_code(self, action_unit_name, feedback):
        prompt = f"Based on the following feedback, correct the code:\n\nFeedback: {feedback}\n\nCode:\n{self.task_context.history[-1]['code_snippet']}"
        response = call_openai_gpt(prompt)
        return response

    def write_initial_code(self, action_unit):
        prompt = f"Write the initial code for the following instruction:\n\n{action_unit.instruction}"
        response = call_openai_gpt(prompt)
        return response

    def run_task(self):
        self.check_code_snippet_buffer()
        while True:
            action_unit_name = self.choose_action_unit()
            if action_unit_name == "task_completed":
                break
            self.execute_action_unit(action_unit_name)
        print(self.task_context.display())
        return self.task_context.history


class CodeReviewerAgent:
    def review_code(self, full_context, step):
        prompt = f"Review the following code snippet in the context of the task:\n\nTask context:\n{full_context}\n\n" \
                 f"Code:\n{step['code_snippet']}\n\n" \
                 f"Execution result:\nStdout:\n{step['stdout']}\nStderr:\n{step['stderr']}\n\n" \
                 f"Your feedback:"
        response = call_openai_gpt(prompt)
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
    guidelines = "High-level guidelines for performing data analysis tasks."
    action_units = [
        ActionUnit("load_data", "Load the dataset."),
        ActionUnit("preprocess_data", "Preprocess the dataset."),
        ActionUnit("train_model", "Train the machine learning model."),
        ActionUnit("evaluate_model", "Evaluate the model performance.")
    ]

    data_scientist = DataScientistAgent(guidelines, action_units)
    task_context = data_scientist.run_task()

    for step in task_context:
        print(json.dumps(step, indent=4))
