PREPROCESS_TOOLS: str = \
"""Tools:
"tools.preprocess" provides well-developed helper functions for this project. Henceforth, it will be referred to as 
"the library". Please import and use functions from the library when possible. But if none of these function satisfy
your needs, feel free to adapt the implementation or write your own. Below is the source code:
{tools_code}
"""

STATISTICIAN_TOOLS: str = \
"""Tools:
"tools.statistics" provides well-developed helper functions for this project. Henceforth, it will be referred to as 
"the library". Please import and use functions from the library when possible. But if none of these function satisfy
your needs, feel free to adapt the implementation or write your own. Below is the source code:
{tools_code}
"""

MULTI_STEP_SETUPS: str = \
"""
Programming Environment Setup:
{path_setup}

NOTE: The overall preprocessing requires multiple steps, where each step often depends on the execution results of 
previous steps. In each step, you perform an Action Unit by writing a code snippet following instructions, and then the 
execution result will be given to you for either revision of the current step or progression to the next step.

Based on the context, write code to follow the instructions.
"""

TASK_COMPLETED_PROMPT: str = \
"""
Mark the task as completed when you have finished processing a dataset or you find you should stop early, either because 
there is no input data available, or that the gene expression or clinical data is missing from the dataset.
"""

CODE_INDUCER = """
NOTE: 
ONLY IMPLEMENT CODE FOR THE CURRENT STEP. MAKE SURE THE CODE CAN BE CONCATENATED WITH THE CODE FROM PREVIOUS STEPS AND CORRECTLY EXECUTED.

FORMAT:
```python
[your_code]
```

NO text outside the code block.
Your code:
"""