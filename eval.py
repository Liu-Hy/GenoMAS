import argparse
import json
import os
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.utils import get_question_pairs
from utils.metrics import evaluate_gene_selection
from tools.statistics import get_gene_regressors

def average_metrics(metrics_list):
    """Average a list of metric dictionaries."""
    if not metrics_list:
        return {}

    avg_metrics = {}
    for metric in metrics_list[0]:
        if isinstance(metrics_list[0][metric], (int, float)):
            avg_metrics[metric] = float(np.round(np.nanmean([p[metric] for p in metrics_list]), 2))

    return avg_metrics


def evaluate_dataset_selection(pred_dir, ref_dir):
    """
    Evaluate dataset filtering and selection by comparing predicted and reference cohort info files.
    
    This function evaluates two aspects:
    1. Dataset Filtering (DF): Binary classification of dataset availability (is_available)
    2. Dataset Selection (DS): Accuracy in selecting the best dataset(s) for each problem
    
    Args:
        pred_dir: Path to prediction directory
        ref_dir: Path to reference directory
        
    Returns:
        Dictionary of evaluation metrics for dataset filtering and selection
    """
    # Initialize lists to store per-trait metrics
    filtering_metrics_list = []
    selection_metrics_list = []
    
    # Track traits we've already evaluated for dataset filtering
    seen_traits = set()
    
    # Get all trait-condition pairs from the metadata directory
    task_info_file = './metadata/task_info.json'
    all_pairs = get_question_pairs(task_info_file)
    
    # Process each trait-condition pair
    with tqdm(total=len(all_pairs), desc="Evaluating dataset filtering and selection") as pbar:
        for i, (trait, condition) in enumerate(all_pairs):
            # Initialize metrics
            trait_filtering_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
            problem_selection_metrics = {'accuracy': 0.0}
            
            # Get trait cohort info paths
            ref_trait_dir = os.path.join(ref_dir, 'preprocess', trait)
            pred_trait_dir = os.path.join(pred_dir, 'preprocess', trait)
            ref_trait_info_path = os.path.join(ref_trait_dir, 'cohort_info.json')
            pred_trait_info_path = os.path.join(pred_trait_dir, 'cohort_info.json')
            
            if not os.path.exists(ref_trait_info_path):
                print(f"Warning: Reference cohort info not found at '{ref_trait_info_path}'")
                pbar.update(1)
                continue
                
            # Penalise missing artefacts (FN counts will increase; selection accuracy will be 0).
            if not os.path.exists(pred_trait_info_path):
                print(f"Warning: Prediction cohort info not found at '{pred_trait_info_path}'. Treating as empty.")
                pred_trait_info = {}
            else:
                with open(pred_trait_info_path, 'r') as f:
                    pred_trait_info = json.load(f)
            
            try:
                # Load reference and prediction trait cohort info
                with open(ref_trait_info_path, 'r') as f:
                    ref_trait_info = json.load(f)
                
                # Only evaluate trait filtering metrics if we haven't seen this trait before
                if trait not in seen_traits:
                    # Evaluate dataset filtering based on is_available attribute
                    for cohort_id in set(ref_trait_info.keys()).union(set(pred_trait_info.keys())):
                        ref_available = ref_trait_info.get(cohort_id, {}).get('is_available', False)
                        pred_available = pred_trait_info.get(cohort_id, {}).get('is_available', False)
                        
                        if ref_available and pred_available:
                            trait_filtering_metrics['tp'] += 1
                        elif ref_available and not pred_available:
                            trait_filtering_metrics['fn'] += 1
                        elif not ref_available and pred_available:
                            trait_filtering_metrics['fp'] += 1
                        else:  # not ref_available and not pred_available
                            trait_filtering_metrics['tn'] += 1
                    
                    # Calculate metrics for this trait
                    filtering_result = calculate_metrics_from_confusion(
                        trait_filtering_metrics['tp'],
                        trait_filtering_metrics['fp'],
                        trait_filtering_metrics['tn'],
                        trait_filtering_metrics['fn']
                    )
                    
                    # Store trait name as part of the metrics
                    filtering_result['trait'] = trait
                    
                    # Add to the filtering metrics list
                    filtering_metrics_list.append(filtering_result)
                    
                    # Mark this trait as seen
                    seen_traits.add(trait)
                
                # Select best dataset(s) using the refactored function
                ref_selection = select_cohorts(
                    root_dir=ref_dir,
                    trait=trait,
                    condition=condition
                )
                
                pred_selection = select_cohorts(
                    root_dir=pred_dir,
                    trait=trait,
                    condition=condition
                )
                
                # Check if selections match
                if ref_selection == pred_selection:
                    problem_selection_metrics['accuracy'] = 100.0
                
                # Store trait and condition names as part of the metrics
                problem_selection_metrics['trait'] = trait
                problem_selection_metrics['condition'] = condition
                
                selection_metrics_list.append(problem_selection_metrics)
                
                # Update running average more frequently - every 5 iterations or at start/end
                if (i + 1) % 5 == 0 or i == 0 or i == len(all_pairs) - 1:
                    # Display both filtering and selection metrics in a single progress bar update
                    display_running_average(
                        pbar, 
                        filtering_metrics_list, 
                        "Dataset filtering", 
                        ['precision', 'recall', 'f1', 'accuracy'],
                        selection_metrics_list,
                        "Dataset selection",
                        ['accuracy']
                    )
                
            except Exception as e:
                print(f"Error evaluating {trait}-{condition}: {str(e)}")
                print(traceback.format_exc())
            
            pbar.update(1)
    
    # Calculate average metrics across all traits
    avg_filtering_metrics = average_metrics(filtering_metrics_list)
    avg_selection_metrics = average_metrics(selection_metrics_list)
    
    return {
        'filtering_metrics': {
            'per_trait': filtering_metrics_list,
            'average': avg_filtering_metrics
        },
        'selection_metrics': {
            'per_problem': selection_metrics_list,
            'average': avg_selection_metrics
        }
    }


def select_cohorts(root_dir, trait, condition=None, gene_info_path='./metadata/task_info.json'):
    """
    Select the best cohort or cohort pair for analysis.
    Unified function that handles both one-step and two-step dataset selection.
    
    Args:
        root_dir: Base directory containing output data
        trait: Name of the trait
        condition: Name of the condition (optional)
        gene_info_path: Path to gene info metadata file (default: './metadata/task_info.json')
    
    Returns:
        For one-step: Selected cohort ID or None if no suitable cohort found
        For two-step: Tuple of (trait_cohort_id, condition_cohort_id) or None if no suitable pair found
    """
    # Set up necessary paths
    trait_dir = os.path.join(root_dir, 'preprocess', trait)
    trait_info_path = os.path.join(trait_dir, 'cohort_info.json')
    
    # Check if trait directory and info exist
    if not os.path.exists(trait_info_path):
        print(f"Warning: Trait cohort info not found for '{trait}'")
        return None
    
    # Load trait info
    with open(trait_info_path, 'r') as f:
        trait_info = json.load(f)
        
    # One-step problem (only trait, or trait with Age/Gender condition)
    if condition is None or condition.lower() in ['age', 'gender', 'none']:
        # Filter usable cohorts
        usable_cohorts = {}
        for cohort_id, info in trait_info.items():
            if info.get('is_usable', False):
                # For Age/Gender conditions, filter cohorts with that info
                if condition == 'Age' and not info.get('has_age', False):
                    continue
                elif condition == 'Gender' and not info.get('has_gender', False):
                    continue
                usable_cohorts[cohort_id] = info
        
        if not usable_cohorts:
            return None
        
        # Select cohort with largest sample size
        return max(usable_cohorts.items(), key=lambda x: x[1].get('sample_size', 0))[0]
    
    # Two-step problem (trait with another non-basic condition)
    else:
        # Set up condition paths
        condition_dir = os.path.join(root_dir, 'preprocess', condition)
        condition_info_path = os.path.join(condition_dir, 'cohort_info.json')
        
        # Check if condition directory and info exist
        if not os.path.exists(condition_info_path):
            print(f"Warning: Condition cohort info not found for '{condition}'")
            return None
            
        # Load condition info
        with open(condition_info_path, 'r') as f:
            condition_info = json.load(f)
            
        # Filter usable cohorts
        usable_trait_cohorts = {k: v for k, v in trait_info.items() if v.get('is_usable', False)}
        usable_condition_cohorts = {k: v for k, v in condition_info.items() if v.get('is_usable', False)}
        
        if not usable_trait_cohorts or not usable_condition_cohorts:
            return None
        
        # Create all possible pairs with their product of sample sizes
        pairs = []
        for trait_id, trait_info_item in usable_trait_cohorts.items():
            for cond_id, cond_info_item in usable_condition_cohorts.items():
                trait_size = trait_info_item.get('sample_size', 0)
                cond_size = cond_info_item.get('sample_size', 0)
                pairs.append((trait_id, cond_id, trait_size * cond_size))
        
        # Sort by product of sample sizes (largest first)
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Find first pair with common gene regressors
        for trait_id, cond_id, _ in pairs:
            trait_data_path = os.path.join(trait_dir, f"{trait_id}.csv")
            condition_data_path = os.path.join(condition_dir, f"{cond_id}.csv")
            
            if os.path.exists(trait_data_path) and os.path.exists(condition_data_path):
                # Load the data to check for common gene regressors
                try:
                    trait_data = pd.read_csv(trait_data_path, index_col=0).astype('float')
                    condition_data = pd.read_csv(condition_data_path, index_col=0).astype('float')
                    
                    # Check for common gene regressors
                    gene_regressors = get_gene_regressors(trait, condition, trait_data, condition_data, gene_info_path)
                    
                    if gene_regressors:
                        return trait_id, cond_id
                except Exception as e:
                    print(f"Error processing pair ({trait_id}, {cond_id}): {str(e)}")
                    # If there's an error, try the next pair
                    continue
        
        # No valid pair found
        return None


def calculate_metrics_from_confusion(tp, fp, tn, fn):
    """
    Calculate precision, recall, F1, and accuracy from confusion matrix values.
    
    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives
        
    Returns:
        Dictionary of metrics
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'accuracy': accuracy * 100
    }


def calculate_jaccard(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 0.0 if union == 0 else intersection / union


def calculate_pearson_correlation(df1, df2):
    """Calculate Pearson correlation between common features in two dataframes.
    Optimized for large datasets using numpy vectorization."""
    common_samples = df1.index.intersection(df2.index)
    common_features = df1.columns.intersection(df2.columns)
    
    if len(common_samples) == 0 or len(common_features) == 0:
        return 0.0
    
    # Extract only common samples and features
    aligned_df1 = df1.loc[common_samples, common_features]
    aligned_df2 = df2.loc[common_samples, common_features]
    
    # Fill missing values with column means (more efficient than column-by-column)
    aligned_df1 = aligned_df1.fillna(aligned_df1.mean())
    aligned_df2 = aligned_df2.fillna(aligned_df2.mean())
    
    # Handle any remaining NaNs (e.g., columns that are all NaN)
    aligned_df1 = aligned_df1.fillna(0.0)
    aligned_df2 = aligned_df2.fillna(0.0)
    
    # Vectorized Pearson correlation calculation
    try:
        # Convert to numpy arrays for faster computation
        X = aligned_df1.values
        Y = aligned_df2.values
        n_samples = X.shape[0]
        
        # Center the data (subtract column means)
        X_centered = X - np.mean(X, axis=0)
        Y_centered = Y - np.mean(Y, axis=0)
        
        # Calculate standard deviations for each column
        X_std = np.std(X, axis=0)
        Y_std = np.std(Y, axis=0)
        
        # Create mask for valid columns (non-zero std dev in both datasets)
        valid_cols = (X_std != 0) & (Y_std != 0)
        
        if not np.any(valid_cols):
            return 0.0  # No valid columns to correlate
        
        # Calculate correlation only for valid columns
        # Use the formula: corr = sum(X_centered * Y_centered) / (n * std_X * std_Y)
        numerator = np.sum(X_centered[:, valid_cols] * Y_centered[:, valid_cols], axis=0)
        denominator = n_samples * X_std[valid_cols] * Y_std[valid_cols]
        correlations = numerator / denominator
        
        # Handle any NaN values that might have slipped through
        correlations = np.nan_to_num(correlations, nan=0.0)
        
        # Return the mean correlation
        return float(np.mean(correlations))
    except Exception as e:
        print(f"Error calculating Pearson correlation: {str(e)}")
        return 0.0


def evaluate_csv(pred_file_path, ref_file_path, subtask="linked"):
    """
    Evaluate preprocessing by comparing prediction and reference CSV files.
    
    Args:
        pred_file_path: Path to the prediction CSV file
        ref_file_path: Path to the reference CSV file
        subtask: The preprocessing subtask ('gene', 'clinical', 'linked')
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Default metrics if file doesn't exist
    default_metrics = {
        'attributes_jaccard': 0.0,
        'samples_jaccard': 0.0,
        'feature_correlation': 0.0,
        'composite_similarity_correlation': 0.0
    }
    
    # Check if prediction file exists
    if not os.path.isfile(pred_file_path):
        return default_metrics
    
    try:
        # Read CSV files
        df1 = pd.read_csv(pred_file_path, index_col=0)
        df2 = pd.read_csv(ref_file_path, index_col=0)
        
        # Reset index and column names to avoid possible errors and confusion
        df1.index.name = None
        df1.columns.name = None
        df2.index.name = None
        df2.columns.name = None
        
        # Make sure rows represent samples and columns represent features
        if subtask != "linked":
            # Transpose the DataFrames
            df1 = df1.T
            df2 = df2.T
        
        # Return default metrics if any dataframe is empty
        if df1.empty or df2.empty:
            return default_metrics

        # Calculate metrics
        attributes_jaccard = calculate_jaccard(set(df1.columns), set(df2.columns))
        samples_jaccard = calculate_jaccard(set(df1.index), set(df2.index))
        feature_correlation = calculate_pearson_correlation(df1, df2)
        composite_similarity_correlation = attributes_jaccard * samples_jaccard * feature_correlation

        return {
            'attributes_jaccard': attributes_jaccard,
            'samples_jaccard': samples_jaccard,
            'feature_correlation': feature_correlation,
            'composite_similarity_correlation': composite_similarity_correlation
        }
    except Exception as e:
        print(f"Error processing {pred_file_path} and {ref_file_path}")
        print(f"Error details: {str(e)}")
        print(traceback.format_exc())
        return default_metrics


def display_running_average(pbar, metrics_list, task_name, metrics_to_show=None, second_metrics_list=None, second_task_name=None, second_metrics_to_show=None):
    """
    Display running average of metrics in the progress bar.
    
    Args:
        pbar: tqdm progress bar
        metrics_list: List of metric dictionaries
        task_name: Name of the task for display
        metrics_to_show: List of metrics to display (if None, show all numeric metrics)
        second_metrics_list: Optional second list of metrics to display (e.g., selection metrics)
        second_task_name: Name for the second task
        second_metrics_to_show: Metrics to show for the second task
    """
    # Skip if there are no metrics
    if not metrics_list:
        pbar.set_description(f"{task_name}: No metrics yet")
        return
    
    # Calculate average metrics
    avg_metrics = average_metrics(metrics_list)
    
    # Determine which metrics to show
    if metrics_to_show is None:
        metrics_to_show = [k for k, v in avg_metrics.items() if isinstance(v, (int, float))]
    
    # Filter out metadata keys that aren't metrics
    metrics_to_show = [m for m in metrics_to_show if m not in ['trait', 'file', 'condition', 'category']]
    
    # Create compact description for progress bar
    desc_parts = []
    for metric in metrics_to_show:
        if metric in avg_metrics:
            desc_parts.append(f"{metric[:3]}={avg_metrics[metric]:.2f}")
    
    # Process second metrics list if provided
    second_desc_parts = []
    if second_metrics_list and second_task_name:
        second_avg_metrics = average_metrics(second_metrics_list)
        
        if second_metrics_to_show is None:
            second_metrics_to_show = [k for k, v in second_avg_metrics.items() 
                                     if isinstance(v, (int, float))]
        
        # Filter out metadata keys that aren't metrics
        second_metrics_to_show = [m for m in second_metrics_to_show 
                                 if m not in ['trait', 'file', 'condition', 'category']]
        
        for metric in second_metrics_to_show:
            if metric in second_avg_metrics:
                second_desc_parts.append(f"{metric[:3]}={second_avg_metrics[metric]:.2f}")
    
    # Build the description with both primary and secondary metrics
    description = f"{task_name}: " + " ".join(desc_parts) if desc_parts else f"{task_name}: No metrics yet"
    
    if second_desc_parts and second_task_name:
        description += f" | {second_task_name}: " + " ".join(second_desc_parts)
    
    # Set the progress bar description
    pbar.set_description(description)


def evaluate_dataset_preprocessing(pred_dir, ref_dir, subtasks=None):
    """
    Evaluate preprocessing by comparing predicted and reference datasets.
    
    Args:
        pred_dir: Path to prediction directory
        ref_dir: Path to reference directory
        subtasks: List of subtasks to evaluate ('gene', 'clinical', 'linked')
                 or None to evaluate all
        
    Returns:
        Dictionary of evaluation metrics for each subtask
    """
    results = {}
    if subtasks is None:
        subtasks = ["gene", "clinical", "linked"]
    
    pred_preprocess_dir = os.path.join(pred_dir, "preprocess")
    ref_preprocess_dir = os.path.join(ref_dir, "preprocess")
    
    if not os.path.exists(pred_preprocess_dir):
        print(f"Warning: Preprocessing prediction directory '{pred_preprocess_dir}' does not exist.")
        return {subtask: {} for subtask in subtasks}
    
    for subtask in subtasks:
        metrics_list = []
        processed_count = 0
        
        # Get list of trait directories
        trait_dirs = []
        for t in os.listdir(ref_preprocess_dir):
            ref_trait_dir = os.path.join(ref_preprocess_dir, t)
            if os.path.isdir(ref_trait_dir):
                trait_dirs.append(t)
        
        # Count total files to process for better progress tracking
        total_files = 0
        for trait in trait_dirs:
            ref_trait_dir = os.path.join(ref_preprocess_dir, trait)
            # Determine the subdirectory path based on subtask
            if subtask in ["gene", "clinical"]:
                sub_dir = os.path.join(ref_trait_dir, f"{subtask}_data")
            else:  # linked
                sub_dir = ref_trait_dir
            
            if os.path.isdir(sub_dir):
                csv_files = [f for f in os.listdir(sub_dir) if f.endswith(".csv")]
                total_files += len(csv_files)
        
        # Process each trait directory with progress bar
        with tqdm(total=len(trait_dirs), desc=f"Evaluating {subtask} data preprocessing") as pbar:
            for trait_idx, trait in enumerate(trait_dirs):
                ref_trait_dir = os.path.join(ref_preprocess_dir, trait)
                
                # Determine the subdirectory path based on subtask
                if subtask in ["gene", "clinical"]:
                    sub_dir = os.path.join(ref_trait_dir, f"{subtask}_data")
                else:  # linked
                    sub_dir = ref_trait_dir
                
                if not os.path.isdir(sub_dir):
                    pbar.update(1)
                    continue
                
                # Process each CSV file
                csv_files = [f for f in sorted(os.listdir(sub_dir)) if f.endswith(".csv")]
                for file_idx, file in enumerate(csv_files):
                    ref_file_path = os.path.join(sub_dir, file)
                    
                    # Get corresponding prediction file path
                    if subtask in ["gene", "clinical"]:
                        pred_file_path = os.path.join(pred_preprocess_dir, trait, f"{subtask}_data", file)
                    else:  # linked
                        pred_file_path = os.path.join(pred_preprocess_dir, trait, file)
                    
                    try:
                        # Evaluate the file pair
                        file_metrics = evaluate_csv(pred_file_path, ref_file_path, subtask)
                        
                        # Add trait and file information
                        file_metrics['trait'] = trait
                        file_metrics['file'] = file
                        
                        metrics_list.append(file_metrics)
                        processed_count += 1
                        
                        # Update running average more frequently:
                        # - At first file
                        # - Every 5 files
                        # - At last file per trait
                        # - At last trait
                        if (processed_count % 5 == 0 or 
                            processed_count == 1 or 
                            file_idx == len(csv_files) - 1 or 
                            trait_idx == len(trait_dirs) - 1):
                            
                            # Show progress
                            pbar.write(f"\nProcessed {processed_count}/{total_files} files")
                            
                            # Display metrics
                            display_running_average(
                                pbar, 
                                metrics_list, 
                                f"{subtask.capitalize()} preprocessing", 
                                ['feature_correlation', 'composite_similarity_correlation']
                            )
                        
                    except Exception as e:
                        print(f"Error evaluating {trait}/{file}: {str(e)}")
                
                pbar.update(1)
        
        # Store both per-file metrics and averages
        results[subtask] = {
            'per_file': metrics_list,
            'average': average_metrics(metrics_list)
        }
    
    return results


def evaluate_statistical_analysis(pred_dir, ref_dir):
    """Evaluate statistical analysis (gene selection) task."""
    results = {}
    pred_regress_dir = os.path.join(pred_dir, 'regress')
    ref_regress_dir = os.path.join(ref_dir, 'regress')
    
    if not os.path.exists(pred_regress_dir):
        print(f"Warning: Statistical analysis prediction directory '{pred_regress_dir}' does not exist.")
        return {}, {}
    
    # Get all trait directories at once to prepare for processing
    trait_dirs = [t for t in sorted(os.listdir(ref_regress_dir)) 
                 if os.path.isdir(os.path.join(ref_regress_dir, t))]
    
    # Count and prepare all files for processing
    all_files = []
    for trait in trait_dirs:
        ref_trait_path = os.path.join(ref_regress_dir, trait)
        json_files = [f for f in sorted(os.listdir(ref_trait_path)) 
                     if f.startswith('significant_genes') and f.endswith('.json')]
        
        for filename in json_files:
            parts = filename.split('_')
            condition = '_'.join(parts[3:])[:-5]
            ref_file = os.path.join(ref_trait_path, filename)
            pred_file = os.path.join(pred_regress_dir, trait, filename)
            all_files.append((trait, condition, ref_file, pred_file))
    
    metrics_for_display = []
    with tqdm(total=len(all_files), desc="Evaluating statistical analysis") as pbar:
        for i, (trait, condition, ref_file, pred_file) in enumerate(all_files):
            try:
                metrics = evaluate_problem_result(ref_file, pred_file)
                results[(trait, condition)] = metrics
                
                # Add trait and condition for display purposes
                metrics_copy = metrics.copy()
                metrics_copy['trait'] = trait
                metrics_copy['condition'] = condition
                metrics_for_display.append(metrics_copy)
                
                # Update the progress bar display at regular intervals
                # Display on 1st, every 5th, and last file
                if i == 0 or (i + 1) % 5 == 0 or i == len(all_files) - 1:
                    display_running_average(
                        pbar, 
                        metrics_for_display, 
                        "Statistical analysis", 
                        ['precision', 'recall', 'f1', 'jaccard'] 
                    )
            except Exception as e:
                print(f"Error evaluating {pred_file}: {str(e)}")
            
            # Update the progress
            pbar.update(1)
    
    # Categorize and aggregate the results
    categorized_avg_metrics = categorize_and_aggregate(results)
    return results, categorized_avg_metrics


def evaluate_problem_result(ref_file, pred_file):
    """Calculate metrics for gene selection evaluation."""
    assert os.path.exists(ref_file), "Reference file does not exist"
    with open(ref_file, 'r') as rfile:
        ref = json.load(rfile)
    ref_genes = ref["significant_genes"]["Variable"]

    # If the 'pred_file' does not exist, it indicates the agent's regression code fails to run on this question
    metrics = {'success': 0.0,
               'precision': 0.0,
               'recall': 0.0,
               'f1': 0.0,
               'auroc': 0.0,
               'gsea_es': 0.0,
               'trait_pred_accuracy': 0.0,
               'trait_pred_f1': 0.0}

    if os.path.exists(pred_file):
        with open(pred_file, 'r') as file:
            result = json.load(file)
        pred_genes = result["significant_genes"]["Variable"]
        metrics.update(evaluate_gene_selection(pred_genes, ref_genes))

        # Optionally, record performance on trait prediction.
        try:
            metrics['trait_pred_accuracy'] = result["cv_performance"]["prediction"]["accuracy"]
        except KeyError:
            pass
        try:
            metrics['trait_pred_f1'] = result["cv_performance"]["prediction"]["f1"]
        except KeyError:
            pass

        metrics['success'] = 100.0

    return metrics


def categorize_and_aggregate(results):
    """Categorize and aggregate metrics by condition type."""
    categorized_results = {'Unconditional one-step': [], 'Conditional one-step': [], 'Two-step': []}
    for pair, metrics in results.items():
        condition = pair[1]
        if condition is None or condition.lower() == "none":
            category = 'Unconditional one-step'
        elif condition.lower() in ["age", "gender"]:
            category = 'Conditional one-step'
        else:
            category = 'Two-step'
        categorized_results[category].append(metrics)

    aggregated_metrics = {}
    for category, metrics_list in categorized_results.items():
        aggregated_metrics[category] = average_metrics(metrics_list)
    aggregated_metrics['Overall'] = average_metrics(
        [metric for sublist in categorized_results.values() for metric in sublist])
    return aggregated_metrics


def main(pred_dir, ref_dir, tasks=None, preprocess_subtasks=None):
    """
    Main evaluation function that can evaluate different tasks.
    
    Args:
        pred_dir: Path to prediction directory
        ref_dir: Path to reference directory
        tasks: List of tasks to evaluate ('selection', 'preprocessing', 'analysis')
               or None to evaluate all
        preprocess_subtasks: List of preprocessing subtasks to evaluate
                           ('gene', 'clinical', 'linked') or None to evaluate all
    
    Returns:
        Dictionary of evaluation results for each task
    """
    if tasks is None:
        tasks = ["selection", "preprocessing", "analysis"]
    
    results = {}
    
    # Evaluate dataset selection
    if "selection" in tasks:
        print("\n=== Evaluating Dataset Selection ===")
        results["selection"] = evaluate_dataset_selection(pred_dir, ref_dir)
        
        # Print selection results immediately
        print("\nDataset Selection Results:")
        if "filtering_metrics" in results["selection"]:
            filtering_avg = results["selection"]["filtering_metrics"]["average"]
            print("\nFiltering Average Metrics:")
            for metric, value in filtering_avg.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
        
        if "selection_metrics" in results["selection"]:
            selection_avg = results["selection"]["selection_metrics"]["average"]
            print("\nSelection Average Metrics:")
            for metric, value in selection_avg.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
    
    # Evaluate preprocessing
    if "preprocessing" in tasks:
        print("\n=== Evaluating Dataset Preprocessing ===")
        results["preprocessing"] = evaluate_dataset_preprocessing(pred_dir, ref_dir, preprocess_subtasks)
        
        # Print preprocessing results immediately
        print("\nDataset Preprocessing Results:")
        for subtask, subtask_results in results["preprocessing"].items():
            if "average" in subtask_results:
                avg_metrics = subtask_results["average"]
                print(f"\n{subtask.capitalize()} Average Metrics:")
                for metric, value in avg_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
            else:
                print(f"  No results available for {subtask}")
    
    # Evaluate statistical analysis
    if "analysis" in tasks:
        print("\n=== Evaluating Statistical Analysis ===")
        problem_results, categorized_metrics = evaluate_statistical_analysis(pred_dir, ref_dir)
        results["analysis"] = {
            "problem_results": problem_results,
            "categorized": categorized_metrics
        }
        
        # Print analysis results immediately
        print("\nStatistical Analysis Results:")
        for category, metrics in categorized_metrics.items():
            print(f"\n{category} Metrics:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for GeneTex")
    parser.add_argument("-p", "--pred-dir", type=str, default="./pred", 
                      help="Path to the prediction directory")
    parser.add_argument("-r", "--ref-dir", type=str, default="./output", 
                      help="Path to the reference directory")
    parser.add_argument("-t", "--tasks", type=str, nargs="+", 
                      choices=["selection", "preprocessing", "analysis"], default=None,
                      help="Tasks to evaluate (default: all)")
    parser.add_argument("-s", "--preprocess-subtasks", type=str, nargs="+", 
                      choices=["gene", "clinical", "linked"], default=None,
                      help="Preprocessing subtasks to evaluate (default: all)")
    
    args = parser.parse_args()
    
    try:
        # Run main evaluation - results are printed in the main function
        results = main(args.pred_dir, args.ref_dir, args.tasks, args.preprocess_subtasks)
    except Exception as e:
        print(f"Error in evaluation process: {str(e)}")
        print(traceback.format_exc())
