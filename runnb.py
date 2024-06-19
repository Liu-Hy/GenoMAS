import os
import glob
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def add_sys_path_cell(notebook, path):
    sys_path_code = f"""
import sys
sys.path.append('{path}')
"""
    sys_path_cell = nbf.v4.new_code_cell(sys_path_code)
    notebook['cells'].insert(0, sys_path_cell)
    return notebook

def execute_notebook(notebook_path, utils_path, working_dir):
    with open(notebook_path, 'r') as nb_file:
        nb_content = nb_file.read()

    notebook = nbf.reads(nb_content, as_version=4)
    notebook = add_sys_path_cell(notebook, utils_path)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        ep.preprocess(notebook, {'metadata': {'path': working_dir}})
    except Exception as e:
        print(f"Error executing the notebook {notebook_path}: {str(e)}")
        return False

    with open(notebook_path, 'w') as nb_file:
        nbf.write(notebook, nb_file)

    print(f"Executed notebook: {notebook_path}")
    return True

# Define the output base directory
output_base_dir = 'code2'
utils_path = '/'  # Change this to the correct path
working_dir = '/'  # Change this to the desired working directory

out_data_dir = './preprocessed'


# Process all Jupyter notebooks in the output directory
for trait_name in os.listdir(output_base_dir):
    out_trait_data_dir = os.path.join(out_data_dir, trait_name)
    os.makedirs(out_trait_data_dir, exist_ok=True)
    os.makedirs(os.path.join(out_trait_data_dir, 'trait_data'), exist_ok=True)
    os.makedirs(os.path.join(out_trait_data_dir, 'gene_data'), exist_ok=True)
    notebook_dir = os.path.join(output_base_dir, trait_name)
    if os.path.isdir(notebook_dir):
        for notebook_file in glob.glob(os.path.join(notebook_dir, '*.ipynb')):
            execute_notebook(notebook_file, utils_path, working_dir)
