import argparse
import json
import os

import numpy as np

from utils.utils import evaluate_gene_selection


def calculate_metrics(ref_file, pred_file):
    assert os.path.exists(ref_file), "Reference file does not exist"
    with open(ref_file, 'r') as rfile:
        ref = json.load(rfile)
    ref_genes = ref["significant_genes"]["Variable"]

    # Initialize all metrics with 0
    # If the 'pred_file' does not exist, it indicates the agent's regression code fails to run on this question
    metrics = {'success': 0.0,
               'precision': 0.0,
               'recall': 0.0,
               'f1': 0.0,
               'trait_pred_accuracy': 0.0,
               'trait_pred_f1': 0.0, }

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


def average_metrics(metrics_list):
    if not metrics_list:
        return {}

    avg_metrics = {}
    for metric in metrics_list[0]:
        avg_metrics[metric] = np.round(np.mean([p[metric] for p in metrics_list]), 2)

    return avg_metrics


def main(pred_dir, ref_dir):
    results = {}
    pred_dir_path = os.path.join('output', 'regress', pred_dir)
    ref_dir_path = os.path.join('output', 'regress', ref_dir)
    for trait in sorted(os.listdir(ref_dir_path)):
        ref_trait_path = os.path.join(ref_dir_path, trait)
        if not os.path.isdir(ref_trait_path):
            continue
        for filename in sorted(os.listdir(ref_trait_path)):
            if filename.startswith('significant_genes') and filename.endswith('.json'):
                parts = filename.split('_')
                condition = '_'.join(parts[3:])[:-5]
                ref_file = os.path.join(ref_trait_path, filename)
                pred_file = os.path.join(pred_dir_path, trait, filename)
                metrics = calculate_metrics(ref_file, pred_file)

                results[(trait, condition)] = metrics
    categorized_avg_metrics = categorize_and_aggregate(results)
    return results, categorized_avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance in variable selection")
    parser.add_argument("-p", "--pred-dir", type=str, required=True, help="Path to the prediction directory")
    parser.add_argument("-r", "--ref-dir", type=str, required=True, help="Path to the reference directory")
    args = parser.parse_args()

    results, categorized_avg_metrics = main(args.pred_dir, args.ref_dir)

    print("\nAggregated Metrics by Category and Overall:")
    for category, metrics in categorized_avg_metrics.items():
        print(f"{category}: \n{metrics}")
