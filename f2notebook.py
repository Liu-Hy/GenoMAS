import os
import re
import nbformat as nbf

# Define the code templates for the odd-numbered steps
TEMPLATES = {
    1: """# STEP1
from utils.preprocess import *
# 1. Identify the paths to the soft file and the matrix file
cohort_dir = '{in_cohort_dir}'
soft_file, matrix_file = geo_get_relevant_filepaths(cohort_dir)

# 2. Read the matrix file to obtain background information and sample characteristics data
background_prefixes = ['!Series_title', '!Series_summary', '!Series_overall_design']
clinical_prefixes = ['!Sample_geo_accession', '!Sample_characteristics_ch1']
background_info, clinical_data = get_background_and_clinical_data(matrix_file, background_prefixes, clinical_prefixes)

# 3. Obtain the sample characteristics dictionary from the clinical dataframe
sample_characteristics_dict = get_unique_values_by_row(clinical_data)

# 4. Explicitly print out all the background information and the sample characteristics dictionary
print("Background Information:")
print(background_info)
print("Sample Characteristics Dictionary:")
print(sample_characteristics_dict)
""",
    3: """# STEP3
# 1. Use the get_genetic_data function from the library to get the gene_data from the matrix_file previously defined.
gene_data = get_genetic_data(matrix_file)

# 2. Print the first 20 row ids for the following step.
print(gene_data.index[:20])
""",
    5: """# STEP5
# 1. Use the 'get_gene_annotation' function from the library to get gene annotation data from the soft file.
gene_annotation = get_gene_annotation(soft_file)

# 2. Use the 'preview_df' function from the library to preview the data and print out the results.
print("Gene annotation preview:")
print(preview_df(gene_annotation))
""",
    7: """# STEP7
# 1. Normalize the obtained gene data with the 'normalize_gene_symbols_in_index' function from the library.
normalized_gene_data = normalize_gene_symbols_in_index(gene_data)
gene_csv_path = '{out_gene_data_file}'
normalized_gene_data.to_csv(gene_csv_path)

# 2. Merge the clinical and genetic data with the 'geo_merge_clinical_genetic_data' function from the library.
merged_data = geo_merge_clinical_genetic_data(selected_clinical_data, normalized_gene_data)

# 3. Determine whether the trait and some demographic attributes in the data is severely biased, and remove biased attributes.
trait_biased, unbiased_merged_data = judge_and_remove_biased_features(merged_data, '{trait}')

# If the trait is not severely biased, save the cohort information and the merged data.

# 4. Save the cohort information.
save_cohort_info('{cohort}', '{json_path}', True, True, is_trait_biased, merged_data)

if not trait_biased:
    # 5. If the trait is not severely biased, save the merged data to a csv file.
    csv_path = '{out_data_file}'
    unbiased_merged_data.to_csv(csv_path)
"""
}


# Function to create a regex pattern for a filled template
def create_filled_template_pattern(template):
    # Escape the template to handle special characters
    template_escaped = re.escape(template)

    # Replace placeholders with a regex pattern that matches non-greedy any character sequences
    pattern = re.sub(r'\\\{.*?\\\}', r'.*?', template_escaped)

    # Adjust pattern to allow for more flexibility with line breaks and comments
    pattern = pattern.replace(r'\#', r'#').replace(r'\n', r'\s*?\n').replace(r'\ ', r'\s*')

    return re.compile(pattern, re.DOTALL)


# Create regex patterns for all templates
PATTERNS = {step: create_filled_template_pattern(template) for step, template in TEMPLATES.items()}


# Function to create a Jupyter notebook from a Python file
def create_notebook_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Initialize a new notebook
    nb = nbf.v4.new_notebook()
    cells = []

    # Initialize variables to track the positions of code blocks
    previous_position = 0
    step_number = 1

    # Iterate over the templates
    while step_number <= 7:
        pattern = PATTERNS.get(step_number)
        if pattern:
            match = pattern.search(content, previous_position)
            if match:
                # Add the code block for the even step (if any)
                if previous_position < match.start():
                    even_step_code = content[previous_position:match.start()].strip()
                    if even_step_code:
                        cells.append(nbf.v4.new_markdown_cell(f"## Step {step_number - 1}"))
                        cells.append(nbf.v4.new_code_cell(even_step_code))

                # Add the code block for the odd step
                cells.append(nbf.v4.new_markdown_cell(f"## Step {step_number}"))
                cells.append(nbf.v4.new_code_cell(content[match.start():match.end()].strip()))

                # Update the previous_position for the next iteration
                previous_position = match.end()
                step_number += 2
            else:
                print(f"Pattern for STEP {step_number} not found.")
                break

    # If there is remaining content after the last predefined step, add it as the last cell
    if previous_position < len(content):
        remaining_code = content[previous_position:].strip()
        if remaining_code:
            cells.append(nbf.v4.new_markdown_cell(f"## Step {step_number - 1}"))
            cells.append(nbf.v4.new_code_cell(remaining_code))

    # Add the cells to the notebook
    nb['cells'] = cells

    # Define the output path for the notebook
    notebook_path = file_path.replace('.py', '.ipynb')

    # Write the notebook to a file
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    print(f"Notebook created: {notebook_path}")


# Path to the directory containing the Python files
directory_path = 'code'

# Process each file in the directory
for filename in os.listdir(directory_path):
    if '11024' not in filename:
        continue
    if filename.endswith('.py'):
        create_notebook_from_file(os.path.join(directory_path, filename))
