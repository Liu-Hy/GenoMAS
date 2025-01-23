import disgenet2r
from disgenet2r import disgenet2r
import pandas as pd
from collections import defaultdict


def get_disgenet_genes(trait_gene_dict, api_key=None):
    """
    Query DisGeNET for known gene-disease associations

    Parameters:
    trait_gene_dict (dict): Dictionary where keys are traits/diseases and values are
                           lists of predicted genes (gene symbols)
    api_key (str): DisGeNET API key. Register at https://www.disgenet.org/sign-up/

    Returns:
    dict: Dictionary where keys are traits and values are ordered lists of
          validated genes from DisGeNET
    """
    # Initialize DisGeNET client
    if api_key is None:
        raise ValueError("DisGeNET API key is required")

    dg = disgenet2r(api_key=api_key)

    # Store results
    validated_genes = defaultdict(list)

    for trait in trait_gene_dict.keys():
        try:
            # Search for disease/trait
            disease_results = dg.disease_search(trait)

            if len(disease_results) == 0:
                print(f"No results found for trait: {trait}")
                continue

            # Get disease ID (using first match)
            disease_id = disease_results.iloc[0]['diseaseId']

            # Get gene associations
            gene_associations = dg.disease2gene(disease_id)

            if len(gene_associations) == 0:
                print(f"No gene associations found for trait: {trait}")
                continue

            # Sort by score (higher score = stronger evidence)
            gene_associations = gene_associations.sort_values('score', ascending=False)

            # Extract gene symbols
            validated_genes[trait] = gene_associations['geneSymbol'].tolist()

        except Exception as e:
            print(f"Error processing trait {trait}: {str(e)}")
            continue

    return dict(validated_genes)


# Example usage:
"""
# First register and get API key from DisGeNET
api_key = "your_api_key_here"

# Example input dictionary
trait_gene_dict = {
    "Type 2 Diabetes": ["TCF7L2", "KCNJ11", "PPARG", "IRS1"],
    "Breast Cancer": ["BRCA1", "BRCA2", "TP53", "PIK3CA"]
}

# Get validated genes
validated_genes = get_disgenet_genes(trait_gene_dict, api_key)

# Example output format:
# {
#     "Type 2 Diabetes": ["TCF7L2", "KCNJ11", "SLC30A8", ...],
#     "Breast Cancer": ["BRCA1", "TP53", "BRCA2", ...]
# }
"""