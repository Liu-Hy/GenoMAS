import subprocess
from openai import AzureOpenAI

class Agent:
    def __init__(self, guidelines):
        self.client = None
        self.guidelines = guidelines
        self.context = TaskContext()
        self.action_units = {}
        self.buffer = {}
        self.subprocess = Subprocess()
        self.initialize_action_units()

    def initialize_action_units(self):
        # Define initial action units with names and instructions
        action_unit_names = ["data_loading", "data_cleaning", "data_analysis", "data_visualization", "exit_with_error", "task_completed"]
        for name in action_unit_names:
            instructions = f"Instructions for {name}"
            self.action_units[name] = ActionUnit(name, instructions)

    def initialize_client(self):
        self.client = AzureOpenAI(
            api_key="57983a2e88fa4d6b81205a8d55d9bd46",
            api_version="2023-10-01-preview",
            azure_endpoint="https://haoyang2.openai.azure.com/"
        )

    def choose_action_unit(self):
        # Analyze current task context to choose the next action unit
        prompt = f"You are a data analysis agent. Here is your current task context: {self.context.history}. Based on this context, what should be the next action unit to perform?"
        action_unit_name = self.call_openai_gpt(prompt).strip()
        return self.action_units[action_unit_name]

    def execute_action_unit(self, action_unit):
        result = self.subprocess.execute(action_unit.code_snippet)
        self.context.record_step(action_unit.name, action_unit.code_snippet, result)
        return result

    def troubleshoot_and_debug(self, action_unit):
        for _ in range(3):  # Maximum of 3 debugging attempts
            result = self.execute_action_unit(action_unit)
            if result['status'] == 'success':
                action_unit.add_revised_version(action_unit.code_snippet)
                return result
            else:
                error_message = result['error']
                prompt = f"The following code snippet failed to execute: {action_unit.code_snippet}. Here is the error message: {error_message}. Please suggest a revised code snippet to fix this issue."
                revised_code_snippet = self.call_openai_gpt(prompt).strip()
                action_unit.add_code_snippet(revised_code_snippet)

        self.context.record_step("exit_with_error", "", "Max debugging attempts reached. Exiting with error.")
        return "exit_with_error"

    def update_code_snippet(self, action_unit):
        if len(action_unit.revised_versions) == 3:
            # Aggregate buffer and update the code snippet
            new_code_snippet = self.aggregate_code_snippets(action_unit.revised_versions)
            action_unit.add_code_snippet(new_code_snippet)
            action_unit.revised_versions = []

    def modify_instructions(self, action_unit):
        # Modify instructions based on new and old code snippets if needed
        pass

    def call_openai_gpt(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": prompt}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4",  # Adjust the model name as needed
            messages=messages
        )
        return response.choices[0].message.content

    def aggregate_code_snippets(self, snippets):
        # Placeholder for aggregating code snippets
        return snippets[-1]

class TaskContext:
    def __init__(self):
        self.history = []
        self.current_step = None

    def record_step(self, action_unit_name, code_snippet, result):
        step = {
            "action_unit": action_unit_name,
            "code_snippet": code_snippet,
            "result": result
        }
        self.history.append(step)
        self.current_step = step

    def get_last_step(self):
        if self.history:
            return self.history[-1]
        return None

class ActionUnit:
    def __init__(self, name, instructions):
        self.name = name
        self.instructions = instructions
        self.code_snippet = ""
        self.revised_versions = []

    def add_code_snippet(self, code_snippet):
        self.code_snippet = code_snippet

    def add_revised_version(self, code_snippet):
        self.revised_versions.append(code_snippet)

class Subprocess:
    def __init__(self):
        self.process = None

    def execute(self, code):
        try:
            # Using eval to simulate code execution (to be replaced with actual execution logic)
            exec_globals = {}
            exec(code, exec_globals)
            result = exec_globals
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def reset(self):
        self.process = None

# Example guidelines (to be extended as needed)
guidelines = """
1. Always validate input data before analysis.
2. Use appropriate visualization techniques for different data types.
3. Ensure reproducibility of analysis by maintaining a clear record of steps.
"""

# Initialize the agent with guidelines
agent = Agent(guidelines)
