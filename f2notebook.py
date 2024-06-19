import nbformat as nbf
import glob
import os

# Mapping of step numbers to their respective headings
step_headings = {
    "1": "Initial Data Loading",
    "2": "Dataset Analysis and Clinical Feature Extraction",
    "3": "Gene Data Extraction",
    "4": "Gene Identifier Review",
    "5": "Gene Annotation (Conditional)",
    "6": "Gene Identifier Mapping",
    "7": "Data Normalization and Merging"
}

def remove_trailing_empty_lines(code_block):
    while code_block and code_block[-1].strip() == '':
        code_block.pop()
    return code_block

def create_notebook_from_python_file(file_path, output_directory):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize a new notebook
    nb = nbf.v4.new_notebook()
    cells = []

    current_code_block = []
    inside_code_block = False
    step_marker = ""

    for line in lines:
        if line.startswith("# STEP") or line.startswith("# Initialize variables") or line.startswith("requires_gene_mapping ="):
            # End the previous code block if it exists
            if current_code_block:
                if step_marker:
                    cells.append(nbf.v4.new_markdown_cell(step_marker))
                cleaned_code_block = remove_trailing_empty_lines(current_code_block)
                cells.append(nbf.v4.new_code_cell(''.join(cleaned_code_block)))
                current_code_block = []
                inside_code_block = False

            # Set the step marker with the appropriate heading
            if line.startswith("# STEP"):
                step_number = line.strip().split("# STEP")[1].strip()
                step_heading = step_headings.get(step_number, "Unknown Step")
                step_marker = f"### Step {step_number}: {step_heading}"
                inside_code_block = True
            elif line.startswith("# Initialize variables"):
                step_number = "2"
                step_heading = step_headings.get(step_number, "Unknown Step")
                step_marker = f"### Step {step_number}: {step_heading}"
                inside_code_block = True
            elif line.startswith("requires_gene_mapping ="):
                step_number = "4"
                step_heading = step_headings.get(step_number, "Unknown Step")
                step_marker = f"### Step {step_number}: {step_heading}"
                current_code_block.append(line)
                inside_code_block = True

            continue

        if inside_code_block:
            if step_marker.startswith("### Step 1") and "../DATA/" in line:
                line = line.replace("../DATA/", "/media/techt/DATA/")
            if "is_trait_biased" in line:
                line = line.replace("is_trait_biased", "trait_biased")
            if "output/preprocess/gs2" in line:
                line = line.replace("output/preprocess/gs2", "preprocessed")
            current_code_block.append(line)

    # Add the last code block
    if current_code_block:
        if step_marker:
            cells.append(nbf.v4.new_markdown_cell(step_marker))
        cleaned_code_block = remove_trailing_empty_lines(current_code_block)
        cells.append(nbf.v4.new_code_cell(''.join(cleaned_code_block)))

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
output_base_dir = 'code2'

# Process all Python files in the input directory
for trait_name in os.listdir(input_base_dir):
    input_dir = os.path.join(input_base_dir, trait_name, 'code')
    if os.path.isdir(input_dir):
        for python_file in glob.glob(os.path.join(input_dir, '*.py')):
            output_dir = os.path.join(output_base_dir, trait_name)
            create_notebook_from_python_file(python_file, output_dir)