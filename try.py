import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Define your code snippets
snippets = [
    """
x = 10
y = 20
result = x + y
print(f'Result of snippet 1: {result}')
""",
    """
z = result * 2
print(f'Result of snippet 2: {z}')
""",
    """
final_result = z + 10
print(f'Final result: {final_result}')
"""
]

# Function to execute a snippet in the given namespace and capture stdout and stderr
def execute_snippet(snippet, namespace):
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(snippet, namespace)
        return stdout.getvalue(), stderr.getvalue(), None
    except Exception as e:
        return stdout.getvalue(), stderr.getvalue(), e

# Function to concatenate code snippets
def concatenate_snippets(snippets, up_to_index):
    return '\n'.join(snippets[:up_to_index+1])

# Main function to execute snippets with error handling and capture
def main(snippets):
    namespace = {}
    for i, snippet in enumerate(snippets):
        stdout, stderr, error = execute_snippet(snippet, namespace)
        print(f"Snippet {i+1} stdout:\n{stdout}")
        print(f"Snippet {i+1} stderr:\n{stderr}")
        if error:
            print(f"Error in snippet {i+1}: {error}")
            # Clear namespace
            namespace = {}
            # Concatenate snippets up to the current step
            concatenated_snippet = concatenate_snippets(snippets, i)
            print(f"Retrying with concatenated snippets up to step {i+1}...")
            stdout, stderr, error = execute_snippet(concatenated_snippet, namespace)
            print(f"Retry stdout:\n{stdout}")
            print(f"Retry stderr:\n{stderr}")
            if error:
                print(f"Failed again in concatenated snippets up to step {i+1}: {error}")
                break

# Execute the main function
if __name__ == "__main__":
    main(snippets)
