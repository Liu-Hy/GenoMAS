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

def compute_auroc_auprc_full(pred_genes: List[str], ref_genes: List[str], all_genes: List[str]):
    pred_set = set(pred_genes)
    ref_set = set(ref_genes)

    y_true = np.array([1 if g in ref_set else 0 for g in all_genes], dtype=int)
    y_score = np.array([1 if g in pred_set else 0 for g in all_genes], dtype=float)

    if np.all(y_true == 0) or np.all(y_true == 1):
        return float('nan'), float('nan')

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    return auroc, auprc

def compute_auroc_auprc_approx(pred_genes: List[str], ref_genes: List[str], n_genes: int = 20000):
    """
    Approximate AUROC/AUPRC if we only know which genes are predicted vs. not,
    and assume all unselected genes share the same (zero) score.
    """
    pred_set = set(pred_genes)
    ref_set = set(ref_genes)

    top_len = len(pred_genes)
    if top_len > n_genes:
        print(f"Warning: predicted {top_len} genes, but only {n_genes} possible. Truncating.")
        pred_genes = pred_genes[:n_genes]
        pred_set = set(pred_genes)
        top_len = n_genes

    missed = ref_set - pred_set
    n_missed = len(missed)

    scores = np.zeros(n_genes, dtype=float)
    labels = np.zeros(n_genes, dtype=int)

    # Assign descending scores for predicted genes
    for i, g in enumerate(pred_genes):
        scores[i] = float(top_len - i)  # or simply 1.0
        labels[i] = 1 if g in ref_set else 0

    bottom_len = n_genes - top_len
    if bottom_len < 0:
        return float('nan'), float('nan')

    if bottom_len > 0 and n_missed > 0:
        import random
        idx_missed = random.sample(range(top_len, top_len + bottom_len),
                                   k=min(n_missed, bottom_len))
        for j in idx_missed:
            labels[j] = 1

    if len(ref_set) == 0 or len(ref_set) == n_genes:
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
    now with:
      - Precision, Recall, F1
      - AUROC, AUPRC
      - Simple GSEA-ES (if do_gsea=True)
    """

    # Basic set-based metrics
    precision_val = gene_precision(pred, ref)
    recall_val = gene_recall(pred, ref)
    f1_val = gene_f1(pred, ref)

    # Compute AUROC / AUPRC
    if all_genes is not None:
        auroc_val, auprc_val = compute_auroc_auprc_full(pred, ref, all_genes)
    else:
        auroc_val, auprc_val = compute_auroc_auprc_approx(pred, ref, n_genes)

    # Compute a GSEA-like enrichment score, if requested
    if do_gsea:
        ranked_list = []
        if all_genes is not None:
            pred_set = set(pred)
            tail = [g for g in all_genes if g not in pred_set]
            ranked_list = list(pred) + tail
        else:
            tail_count = n_genes - len(pred)
            if tail_count >= 0:
                dummy_genes = [f"UNK_{i}" for i in range(tail_count)]
                ranked_list = list(pred) + dummy_genes
            else:
                # Inconsistent edge case: predicted more genes than total
                # We'll just store NaN and skip the GSEA
                pass

        if len(ranked_list) == 0:
            es_val = float('nan')
        else:
            es_val = gsea_enrichment_score(ranked_list, ref)

    else:
        es_val = float('nan')
    
    print("AUROC:", auroc_val, "AUPRC:", auprc_val, "GSEA-ES:", es_val)

    return {
        'precision': precision_val * 100.0,
        'recall':    recall_val * 100.0,
        'f1':        f1_val * 100.0,
        'auroc':     auroc_val,
        'auprc':     auprc_val,
        'gsea_es':   es_val
    }
