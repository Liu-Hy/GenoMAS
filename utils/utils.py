import ast
import json
import os
import re
import traceback
from typing import List, Optional, Dict


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

def check_slow_inference(model: str) -> bool:
    """
    Checks if the model is a slow inference model by parsing the model name.
    """
    # Convert to lowercase first, then split into alphanumeric substrings
    substrings = re.findall(r'[a-z0-9]+', model.lower())
    
    # Check conditions
    has_slow_marker = any(s in ['o1', 'o3', 'r1'] for s in substrings)
    has_mini = 'mini' in substrings
    
    return has_slow_marker and not has_mini

def check_recent_openai_model(model: str) -> bool:
    """
    Checks if the model is a recent OpenAI model (with updated system prompt role) by parsing the model name.
    """
    substrings = re.findall(r'[a-z0-9]+', model.lower())
    has_recent_marker = any(s in ['o1', 'o3'] for s in substrings)
    
    return has_recent_marker

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


def gene_precision(pred: List[str], ref: List[str]) -> float:
    """
    Calculate precision of predicted genes against reference set.
    """
    if len(pred):
        precision = sum([p in ref for p in pred]) / len(pred)
    else:
        if len(ref):
            precision = 0
        else:
            precision = 1
    return precision


def gene_recall(pred: List[str], ref: List[str]) -> float:
    """
    Calculate recall of predicted genes against reference set.
    """
    if len(ref):
        recall = sum([p in pred for p in ref]) / len(ref)
    else:
        if len(pred):
            recall = 0
        else:
            recall = 1
    return recall


def gene_f1(pred: List[str], ref: List[str]) -> float:
    """
    Calculate F1 score between predicted and reference gene sets.
    """
    prec = gene_precision(pred, ref)
    rec = gene_recall(pred, ref)
    if prec + rec == 0:  # Prevent division by zero
        return 0
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def evaluate_gene_selection(pred: List[str], ref: List[str]) -> Dict[str, float]:
    """
    Evaluate the performance of predicted gene selection against a reference set.

    Args:
        pred (List[str]): List of predicted gene symbols.
        ref (List[str]): List of reference (ground truth) gene symbols.

    Returns:
        Dict[str, float]: Dictionary containing precision, recall, F1 score, and Jaccard similarity.
    """
    return {
        'precision': gene_precision(pred, ref) * 100,
        'recall': gene_recall(pred, ref) * 100,
        'f1': gene_f1(pred, ref) * 100,
    }