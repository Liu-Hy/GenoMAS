import os
import glob
import nbformat as nbf
from nbconvert import NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor

def execute_notebook(notebook_path):
    with open(notebook_path, 'r') as nb_file:
        nb_content = nb_file.read()

    notebook = nbf.reads(nb_content, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        ep.preprocess(notebook, {'metadata': {'path': os.path.dirname(notebook_path)}})
    except Exception as e:
        print(f"Error executing the notebook {notebook_path}: {str(e)}")
        return False

    with open(notebook_path, 'w') as nb_file:
        nbf.write(notebook, nb_file)

    print(f"Executed notebook: {notebook_path}")
    return True

# # Just an experiment. Keep the real notebook output dir intact.
output_base_dir = 'code1'

# Process all Jupyter notebooks in the output directory
for trait_name in os.listdir(output_base_dir):
    notebook_dir = os.path.join(output_base_dir, trait_name)
    if os.path.isdir(notebook_dir):
        for notebook_file in glob.glob(os.path.join(notebook_dir, '*.ipynb')):
            execute_notebook(notebook_file)
