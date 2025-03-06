import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics import roc_auc_score, average_precision_score

def gene_precision(pred: List[str], ref: List[str]) -> float:
    if len(pred) == 0:
        return 1.0 if len(ref) == 0 else 0.0
    return sum([p in ref for p in pred]) / len(pred)

def gene_recall(pred: List[str], ref: List[str]) -> float:
    if len(ref) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    return sum([p in pred for p in ref]) / len(ref)

def gene_f1(pred: List[str], ref: List[str]) -> float:
    prec = gene_precision(pred, ref)
    rec = gene_recall(pred, ref)
    if prec + rec == 0:
        return 0.0
    return 2.0 * (prec * rec) / (prec + rec)

def gsea_enrichment_score(ranked_genes: List[str], ref_genes: List[str]) -> float:
    """
    Compute a simple GSEA-like Enrichment Score (ES) using an unweighted running sum.
    """
    ref_set = set(ref_genes)
    N = len(ranked_genes)
    Nh = len(ref_set)
    Nm = N - Nh

    if Nh == 0 or Nh == N:
        return float('nan')

    hit_inc = 1.0 / float(Nh)
    miss_inc = 1.0 / float(Nm)
    running_sum = 0.0
    max_enrichment = -999999.0

    for g in ranked_genes:
        if g in ref_set:
            running_sum += hit_inc
        else:
            running_sum -= miss_inc
        if running_sum > max_enrichment:
            max_enrichment = running_sum

    return max_enrichment

def compute_auroc_auprc_full(pred_genes: List[str],
                                    ref_genes: List[str],
                                    all_genes: List[str]) -> (float, float):
    """
    Compute AUROC/AUPRC by assigning a strictly descending score
    to each predicted gene (based on its index in pred_genes) and
    zero to all unselected genes.
    """
    pred_set = set(pred_genes)
    ref_set = set(ref_genes)

    top_len = len(pred_genes)
    # Build a dictionary: gene -> score
    # If you already have actual coefficients, you could fill them in here instead.
    gene_scores = {}
    for i, g in enumerate(pred_genes):
        # Highest score for the 1st gene in pred_genes, second-highest for the 2nd, etc.
        # i=0 => rank = top_len; i=1 => rank = top_len-1, etc.
        gene_scores[g] = float(top_len - i)

    # Now assign score=0 to all other genes in all_genes
    for g in all_genes:
        if g not in gene_scores:
            gene_scores[g] = 0.0

    # Build arrays for scikit-learn
    y_true = np.array([1 if g in ref_set else 0 for g in all_genes], dtype=int)
    y_score = np.array([gene_scores[g] for g in all_genes], dtype=float)

    # Handle edge cases where all positives or all negatives
    if np.all(y_true == 0) or np.all(y_true == 1):
        return float('nan'), float('nan')

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    return auroc, auprc

def compute_auroc_auprc_approx(pred_genes: List[str], ref_genes: List[str],
                               n_genes: int = 20000):
    """
    Approximate AUROC/AUPRC if we only know which genes are predicted vs. not,
    and assume all unselected genes share the same (zero) score
    """
    pred_set = set(pred_genes)
    ref_set = set(ref_genes)

    top_len = len(pred_genes)
    if top_len > n_genes:
        print(f"Warning: predicted {top_len} genes, but only {n_genes} possible. Truncating.")
        pred_genes = pred_genes[:n_genes]
        pred_set = set(pred_genes)
        top_len = n_genes

    scores = np.zeros(n_genes, dtype=float)
    labels = np.zeros(n_genes, dtype=int)

    # Give descending scores to the predicted genes
    for i, g in enumerate(pred_genes):
        scores[i] = float(top_len - i)
        labels[i] = 1 if g in ref_set else 0

    # All other genes are 0.  No random sprinkling of 'missed' positives.
    # This means if a true positive wasn't predicted, it simply won't appear
    # in these top-len indices as '1'.

    # Now the *number* of positive labels in labels might be smaller
    # than the real total # of positives, but we can't fix that without
    # full knowledge or a custom approach.

    if len(ref_set) == 0 or len(ref_set) == n_genes:
        return float('nan'), float('nan')

    # Edge cases
    if np.all(labels == 0) or np.all(labels == 1):
        return float('nan'), float('nan')

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    return auroc, auprc

def evaluate_gene_selection(pred: List[str],
                            ref: List[str],
                            all_genes: Optional[List[str]] = None,
                            n_genes: int = 20000,
                            do_gsea: bool = True) -> Dict[str, float]:
    """
    Evaluate the performance of predicted gene selection against a reference set,
    with:
      - Precision, Recall, F1
      - AUROC, AUPRC (via a ranking-based approach)
      - Simple GSEA-ES (if do_gsea=True)
    """

    # Basic set-based metrics
    precision_val = gene_precision(pred, ref)
    recall_val = gene_recall(pred, ref)
    f1_val = gene_f1(pred, ref)

    # Compute AUROC / AUPRC: use our new ranking-based function if all_genes is given
    if all_genes is not None:
        auroc_val, auprc_val = compute_auroc_auprc_full(pred, ref, all_genes)
    else:
        # If you have NO list of all genes, you can still do your approximate approach,
        # but you might want to remove the random assignment from that function:
        auroc_val, auprc_val = compute_auroc_auprc_approx(pred, ref, n_genes)

    # Compute a GSEA-like enrichment score, if requested
    if do_gsea:
        # If we have all_genes, let's create a ranking by sorting them by the assigned score.
        es_val = float('nan')
        if all_genes is not None and len(all_genes) > 0:
            # We'll re-use the same scoring logic to create a fully ordered list
            # from highest to lowest rank.
            # Build the same gene_scores dict used in compute_auroc_auprc_full:

            pred_set = set(pred)
            top_len = len(pred)
            gene_scores = {}
            for i, g in enumerate(pred):
                gene_scores[g] = float(top_len - i)
            for g in all_genes:
                if g not in gene_scores:
                    gene_scores[g] = 0.0

            # Sort all_genes by descending score
            ranked_list = sorted(all_genes, key=lambda x: gene_scores[x], reverse=True)

            es_val = gsea_enrichment_score(ranked_list, ref)
        else:
            # If we don't have all_genes, revert to the old logic
            tail_count = n_genes - len(pred)
            if tail_count >= 0:
                dummy_genes = [f"UNK_{i}" for i in range(tail_count)]
                ranked_list = list(pred) + dummy_genes
                es_val = gsea_enrichment_score(ranked_list, ref)
            else:
                es_val = float('nan')
    else:
        es_val = float('nan')

    # print("AUROC:", auroc_val, "AUPRC:", auprc_val, "GSEA-ES:", es_val)

    return {
        'precision': precision_val * 100.0,
        'recall':    recall_val * 100.0,
        'f1':        f1_val * 100.0,
        'auroc':     auroc_val,
        #'auprc':     auprc_val,
        'gsea_es':   es_val
    }
