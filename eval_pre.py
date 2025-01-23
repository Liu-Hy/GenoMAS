import os
import pandas as pd
import numpy as np
import argparse


def calculate_jaccard_similarity(set1, set2):
    try:
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        if union == 0:
            return 0.0
        return intersection / union
    except Exception as e:
        print(f"Error calculating Jaccard similarity: {e}")
        return 0.0


def calculate_pearson_correlation(df1, df2):
    try:
        common_samples = df1.index.intersection(df2.index)
        common_features = df1.columns.intersection(df2.columns)

        if len(common_samples) == 0 or len(common_features) == 0:
            return 0.0

        aligned_df1 = df1.loc[common_samples, common_features]
        aligned_df2 = df2.loc[common_samples, common_features]

        correlations = []
        for col in common_features:
            series1 = aligned_df1[col].dropna()
            series2 = aligned_df2[col].dropna()
            if len(series1) == 0 or len(series2) == 0:
                correlations.append(0)
            else:
                correlations.append(np.corrcoef(series1, series2)[0, 1])

        if len(correlations) == 0:
            return 0.0

        return np.nanmean(correlations)
    except Exception as e:
        print(f"Error calculating Pearson correlation: {e}")
        return 0.0


def evaluate_preprocessing(df1, df2):
    try:
        if df1.empty or df2.empty:
            return {
                'attributes_jaccard': 0.0,
                'samples_jaccard': 0.0,
                'feature_correlation': 0.0,
                'composite_similarity_correlation': 0.0
            }

        attributes_jaccard = calculate_jaccard_similarity(set(df1.columns), set(df2.columns))
        samples_jaccard = calculate_jaccard_similarity(set(df1.index), set(df2.index))
        feature_correlation = calculate_pearson_correlation(df1, df2)
        composite_similarity_correlation = attributes_jaccard * samples_jaccard * feature_correlation

        return {
            'attributes_jaccard': attributes_jaccard,
            'samples_jaccard': samples_jaccard,
            'feature_correlation': feature_correlation,
            'composite_similarity_correlation': composite_similarity_correlation
        }
    except Exception as e:
        print(f"Error in preprocessing evaluation: {e}")
        return {
            'attributes_jaccard': 0.0,
            'samples_jaccard': 0.0,
            'feature_correlation': 0.0,
            'composite_similarity_correlation': 0.0
        }


def evaluate_preprocess(version1, version2, subtask="merged"):
    metrics = []
    dir1 = os.path.join("output/preprocess", version1)
    dir2 = os.path.join("output/preprocess", version2)

    for t in os.listdir(dir2):
        trait_dir = os.path.join(dir2, t)
        if not os.path.isdir(trait_dir):
            continue
        sub_dir = os.path.join(trait_dir, subtask) if subtask in ["gene", "trait"] else trait_dir
        if not os.path.isdir(sub_dir):
            continue
        for file in os.listdir(sub_dir):
            if file.endswith(".csv"):
                fpath2 = os.path.join(sub_dir, file)
                fpath1 = os.path.join(dir1, t, subtask, file) if subtask in ["gene", "trait"] else os.path.join(dir1, t,
                                                                                                                file)
                if not os.path.isfile(fpath1):
                    continue
                try:
                    df1 = pd.read_csv(fpath1)
                    df2 = pd.read_csv(fpath2)
                    print(f"Evaluating {file}")
                    rs = evaluate_preprocessing(df1, df2)
                    print(rs)
                    metrics.append(rs)
                    if len(metrics) % 10 == 0:
                        print_running_average(metrics)
                except Exception as e:
                    print(f"Error processing files {fpath1} and {fpath2}: {e}")

    # Calculate mean for each metric
    try:
        mean_metrics = {
            'attributes_jaccard': np.nanmean([m['attributes_jaccard'] for m in metrics]),
            'samples_jaccard': np.nanmean([m['samples_jaccard'] for m in metrics]),
            'feature_correlation': np.nanmean([m['feature_correlation'] for m in metrics]),
            'composite_similarity_correlation': np.nanmean([m['composite_similarity_correlation'] for m in metrics])
        }
    except Exception as e:
        print(f"Error calculating mean metrics: {e}")
        mean_metrics = {
            'attributes_jaccard': 0.0,
            'samples_jaccard': 0.0,
            'feature_correlation': 0.0,
            'composite_similarity_correlation': 0.0
        }

    return mean_metrics


def print_running_average(metrics):
    try:
        running_mean_metrics = {
            'attributes_jaccard': np.mean([m['attributes_jaccard'] for m in metrics]),
            'samples_jaccard': np.mean([m['samples_jaccard'] for m in metrics]),
            'feature_correlation': np.mean([m['feature_correlation'] for m in metrics]),
            'composite_similarity_correlation': np.mean([m['composite_similarity_correlation'] for m in metrics])
        }
        print("Running Average after {} records:".format(len(metrics)))
        print(running_mean_metrics)
    except Exception as e:
        print(f"Error calculating running average: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate preprocessing.')
    parser.add_argument('--version1', type=str, help='The first version to compare.')
    parser.add_argument('--version2', type=str, help='The second version to compare.')
    parser.add_argument('--subtask', type=str, default='merged', help='The subtask to evaluate (default: merged).')

    args = parser.parse_args()

    version1 = args.version1
    version2 = args.version2
    subtask = args.subtask

    try:
        results = evaluate_preprocess(version1, version2, subtask)
        print("Final Evaluation Results:")
        print(results)
    except Exception as e:
        print(f"Error in evaluation process: {e}")
