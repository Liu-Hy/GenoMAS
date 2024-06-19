import argparse
import os
import pandas as pd
import json
import numpy as np
from utils.statistics import evaluate_gene_selection

def calculate_metrics(ref_file, pred_file):
    assert os.path.exists(ref_file), "Reference file does not exist"
    with open(ref_file, 'r') as rfile:
        ref = json.load(rfile)
    ref_genes = ref["significant_genes"]["Variable"]

    # Initialize all metrics with 0
    # If the 'pred_file' does not exist, it indicates the agent's regression code fails to run on this question
    metrics = {'precision': 0, 'precision_at_50': 0, 'recall': 0, 'f1': 0, 'jaccard': 0, 'jaccard2': 0,
               'success': 0}
    metrics["cv_performance"] = ref["cv_performance"]
    for section in metrics["cv_performance"]:
        for m in metrics["cv_performance"][section]:
            metrics["cv_performance"][section][m] = 0

    if os.path.exists(pred_file):
        with open(pred_file, 'r') as file:
            result = json.load(file)
        metrics["cv_performance"] = result["cv_performance"]
        pred_genes = result["significant_genes"]["Variable"]
        metrics.update(evaluate_gene_selection(pred_genes, ref_genes))
        metrics['success'] = 1

    return metrics

def categorize_and_aggregate(results):
    categorized_results = {'unconditional one-step': [], 'conditional one-step': [], 'two-step': []}
    for pair, metrics in results.items():
        condition = pair[1]
        if condition is None or condition.lower() == "none":
            category = 'unconditional one-step'
        elif condition.lower() in ["age", "gender"]:
            category = 'conditional one-step'
        else:
            category = 'two-step'
        categorized_results[category].append(metrics)

    aggregated_metrics = {}
    for category, metrics_list in categorized_results.items():
        aggregated_metrics[category] = average_metrics(metrics_list)
    aggregated_metrics['overall'] = average_metrics(
        [metric for sublist in categorized_results.values() for metric in sublist])
    return aggregated_metrics

def average_metrics(metrics_list):
    if not metrics_list:
        return {}

    avg_metrics = {}
    for metric in metrics_list[0]:
        if isinstance(metrics_list[0][metric], dict):  # metric == "cv_performance"
            avg_metrics[metric] = {}
            for submetric in metrics_list[0][metric]:  # submetric in ["selection", "prediction"]
                avg_metrics[metric][submetric] = {}
                for subsubmetric in metrics_list[0][metric][submetric]:
                    avg_metrics[metric][submetric][subsubmetric] = np.round(np.mean(
                        [p[metric][submetric][subsubmetric] for p in metrics_list if
                         subsubmetric in p[metric][submetric]]), 2)
        else:
            avg_metrics[metric] = np.round(np.mean([p[metric] for p in metrics_list]), 2)
    return avg_metrics


def main(pred_dir, ref_dir):
    results = {}
    for trait in os.listdir(ref_dir):
        ref_trait_path = os.path.join(ref_dir, trait)
        if not os.path.isdir(ref_trait_path):
            continue
        for filename in os.listdir(ref_trait_path):
            if filename.startswith('significant_genes') and filename.endswith('.json'):
                parts = filename.split('_')
                condition = '_'.join(parts[3:])[:-5]
                ref_file = os.path.join(ref_trait_path, filename)
                pred_file = os.path.join(pred_dir, trait, filename)
                metrics = calculate_metrics(ref_file, pred_file)

                available_traits = os.listdir('/home/techt/Desktop/AI_for_Science/preprocessed/ours')
                #if trait in available_traits and (condition in available_traits or condition.lower() == 'None'):
                results[(trait, condition)] = metrics
                    #print(metrics)
                # print(len(results))
    categorized_avg_metrics = categorize_and_aggregate(results)
    return results, categorized_avg_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance in variable selection")
    parser.add_argument("-p", "--pred_dir", type=str, required=True, help="Path to the prediction directory")
    parser.add_argument("-r", "--ref_dir", type=str, required=True, help="Path to the reference directory")
    args = parser.parse_args()

    results, aggregated_results = main(args.pred_dir, args.ref_dir)
    # print("Individual Metrics:")
    # for path, metrics in results.items():
    #     print(f"{path}: {metrics}")

    print("\nAggregated Metrics by Category and Overall:")
    for category, metrics in aggregated_results.items():
        print(f"{category}: \n{metrics}")
