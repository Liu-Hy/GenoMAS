import os
import argparse
import pandas as pd
import numpy as np

def calculate_jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union

def calculate_pearson_correlation(df1, df2):
    common_samples = df1.index.intersection(df2.index)
    common_features = df1.columns.intersection(df2.columns)

    if len(common_samples) == 0 or len(common_features) == 0:
        return 0.0

    aligned_df1 = df1.loc[common_samples, common_features]
    aligned_df2 = df2.loc[common_samples, common_features]

    correlations = [np.corrcoef(aligned_df1[col], aligned_df2[col])[0, 1] for col in common_features]
    return np.nanmean(correlations)

def evaluate_preprocessing(df1, df2):
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
            try:
                if file.endswith(".csv"):
                    fpath2 = os.path.join(sub_dir, file)
                    fpath1 = os.path.join(dir1, t, file) if subtask == "merged" else os.path.join(dir1, t, subtask, file)
                    if not os.path.isfile(fpath1):
                        continue
                    df1 = pd.read_csv(fpath1)
                    df2 = pd.read_csv(fpath2)
                    print(f"Evaluation {file}")
                    rs = evaluate_preprocessing(df1, df2)
                    print(rs)
                    metrics.append(rs)
            except Exception as e:
                print(e)
                continue

    # Calculate mean for each metric
    mean_metrics = {
        'attributes_jaccard': np.mean([m['attributes_jaccard'] for m in metrics]),
        'samples_jaccard': np.mean([m['samples_jaccard'] for m in metrics]),
        'feature_correlation': np.mean([m['feature_correlation'] for m in metrics]),
        'composite_similarity_correlation': np.mean([m['composite_similarity_correlation'] for m in metrics])
    }

    return mean_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate preprocessing methods.')
    parser.add_argument('--version1', type=str, help='First version to compare')
    parser.add_argument('--version2', type=str, help='Second version to compare')
    parser.add_argument('--subtask', type=str, default='merged', choices=['merged', 'gene', 'trait'],
                        help='Subtask type: merged (default), gene, or trait')

    args = parser.parse_args()

    mean_metrics = evaluate_preprocess(args.version1, args.version2, args.subtask)
    print(mean_metrics)
