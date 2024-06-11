import nbformat as nbf
import glob
import os

def create_notebook_from_python_file(file_path, output_directory):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize a new notebook
    nb = nbf.v4.new_notebook()
    cells = []

    step_code_blocks = []
    current_code_block = []

    for line in lines:
        if line.startswith("# STEP") or line.startswith("# Initialize variables"):
            if current_code_block:
                step_code_blocks.append(current_code_block)
                current_code_block = []
            if line.startswith("# STEP"):
                step_number = line.strip().split("# STEP")[1].strip()
                cells.append(nbf.v4.new_markdown_cell(f"## Step {step_number}"))
            elif line.startswith("# Initialize variables"):
                cells.append(nbf.v4.new_markdown_cell("## Step 2: Initialize Variables"))
        current_code_block.append(line)

    if current_code_block:
        step_code_blocks.append(current_code_block)

    for block in step_code_blocks:
        code = ''.join(block)
        cells.append(nbf.v4.new_code_cell(code))

    nb['cells'] = cells

    # Create output path
    os.makedirs(output_directory, exist_ok=True)
    notebook_path = os.path.join(output_directory, os.path.basename(file_path).replace('.py', '.ipynb'))

    # Write the notebook to a file
    with open(notebook_path, 'w') as notebook_file:
        nbf.write(nb, notebook_file)
    print(f"Converted {file_path} to {notebook_path}")

# Define input and output directories
input_base_dir = 'output/preprocess/gs2'
output_base_dir = 'code'

# Process all Python files in the input directory
for trait_name in os.listdir(input_base_dir):
    input_dir = os.path.join(input_base_dir, trait_name, 'code')
    if os.path.isdir(input_dir):
        for python_file in glob.glob(os.path.join(input_dir, '*.py')):
            output_dir = os.path.join(output_base_dir, trait_name)
            create_notebook_from_python_file(python_file, output_dir)
