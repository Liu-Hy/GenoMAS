import os
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

def evaluate_preprocess(version1, version2):
    metrics = []
    dir1 = os.path.join("output/preprocess", version1)
    dir2 = os.path.join("output/preprocess", version2)

    for t in os.listdir(dir2):
        trait_dir = os.path.join(dir2, t)
        if not os.path.isdir(trait_dir):
            continue
        for file in os.listdir(trait_dir):
            if file.endswith(".csv"):
                fpath2 = os.path.join(trait_dir, file)
                fpath1 = os.path.join(dir1, t, file)
                if not os.path.isfile(fpath1):
                    continue
                df1 = pd.read_csv(fpath1)
                df2 = pd.read_csv(fpath2)
                print(f"Evaluation {file}")
                rs = evaluate_preprocessing(df1, df2)
                print(rs)
                metrics.append(rs)

    # Calculate mean for each metric
    mean_metrics = {
        'attributes_jaccard': np.mean([m['attributes_jaccard'] for m in metrics]),
        'samples_jaccard': np.mean([m['samples_jaccard'] for m in metrics]),
        'feature_correlation': np.mean([m['feature_correlation'] for m in metrics]),
        'composite_similarity_correlation': np.mean([m['composite_similarity_correlation'] for m in metrics])
    }

    return mean_metrics