import ast
import json
import os
import traceback
from typing import List, Optional


def normalize_trait(trait):
    trait = '_'.join(trait.split())
    normalized_trait = ''.join(trait.split("'"))
    return normalized_trait

def normalize_gene_symbols(gene_symbols: List[str]) -> List[Optional[str]]:
    """Use gene synonym information extracted from the NCBI Gene database to normalize gene symbols in a list, and
    return a list of normalized symbols. Unmatched symbols are converted to None.
    """
    with open("./metadata/gene_synonym.json", "r") as f:
        synonym_dict = json.load(f)
    return [synonym_dict.get(g) for g in gene_symbols]

def get_question_pairs(file_path):
    """
    Reads a JSON metadata file and returns a list of trait-condition pairs as questions.
    """
    with open(file_path, 'r') as f:
        task_info = json.load(f)
        all_traits = sorted(list(task_info.keys()))
        all_pairs = []
        for trait in all_traits:
            all_pairs.append((trait, None))
            for condition in task_info[trait]['conditions']:
                all_pairs.append((trait, condition))
        return all_pairs

def extract_function_code(file_path, function_names):
    """
    Extracts the code of specific functions from a Python file.

    Args:
        file_path (str): Path to the Python file.
        function_names (list): List of function names to extract.

    Returns:
        dict: A dictionary where keys are function names, and values are their code as strings.
    """
    with open(file_path, 'r') as file:
        source_code = file.read()
    tree = ast.parse(source_code)
    extracted_codes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            function_code = ast.get_source_segment(source_code, node)
            extracted_codes.append(function_code)

    return '\n\n'.join(extracted_codes)


def load_last_cohort_info(version_dir):
    try:
        with open(os.path.join(version_dir, "last_cohort_info.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_last_cohort_info(version_dir, cohort_info):
    with open(os.path.join(version_dir, "last_cohort_info.json"), "w") as f:
        json.dump(cohort_info, f)


def delete_corrupted_files(output_dir, cohort):
    out_gene_dir = os.path.join(output_dir, 'gene_data')
    out_clinical_dir = os.path.join(output_dir, 'clinical_data')
    out_code_dir = os.path.join(output_dir, 'code')
    for this_dir in [output_dir, out_gene_dir, out_clinical_dir, out_code_dir]:
        ext = "py" if this_dir == out_code_dir else "csv"
        file_path = os.path.join(this_dir, f"{cohort}.{ext}")
        if os.path.exists(file_path):
            os.remove(file_path)


def load_completed_tasks(version_dir):
    """
    Load the set of completed tasks from a JSON file.
    If the file doesn't exist, return an empty set.
    """
    file_path = os.path.join(version_dir, "completed_tasks.json")
    if not os.path.exists(file_path):
        return set()
    try:
        with open(file_path, "r") as file:
            return {tuple(task) for task in json.load(file)}
    except json.JSONDecodeError:
        traceback.print_exc()
        return set()


def add_completed_task(task, version_dir):
    """
    Add a single completed task to the JSON file.
    """
    completed_tasks = load_completed_tasks(version_dir)
    completed_tasks.add(task)

    os.makedirs(version_dir, exist_ok=True)
    file_path = os.path.join(version_dir, "completed_tasks.json")
    with open(file_path, "w") as file:
        json.dump([list(task) for task in completed_tasks], file)
