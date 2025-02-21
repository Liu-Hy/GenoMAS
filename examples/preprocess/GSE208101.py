# Path Configuration
from tools.preprocess import *

# Processing context
trait = "Breast_Cancer"
cohort = "GSE208101"

# Input paths
in_trait_dir = "../DATA/GEO/Breast_Cancer"
in_cohort_dir = "../DATA/GEO/Breast_Cancer/GSE208101"

# Output paths
out_data_file = "./output/preprocess/1/Breast_Cancer/GSE208101.csv"
out_gene_data_file = "./output/preprocess/1/Breast_Cancer/gene_data/GSE208101.csv"
out_clinical_data_file = "./output/preprocess/1/Breast_Cancer/clinical_data/GSE208101.csv"
json_path = "./output/preprocess/1/Breast_Cancer/cohort_info.json"

# STEP1
from tools.preprocess import *
# 1. Identify the paths to the SOFT file and the matrix file
soft_file, matrix_file = geo_get_relevant_filepaths(in_cohort_dir)

# 2. Read the matrix file to obtain background information and sample characteristics data
background_prefixes = ['!Series_title', '!Series_summary', '!Series_overall_design']
clinical_prefixes = ['!Sample_geo_accession', '!Sample_characteristics_ch1']
background_info, clinical_data = get_background_and_clinical_data(matrix_file, background_prefixes, clinical_prefixes)

# 3. Obtain the sample characteristics dictionary from the clinical dataframe
sample_characteristics_dict = get_unique_values_by_row(clinical_data)

# 4. Explicitly print out all the background information and the sample characteristics dictionary
print("Background Information:")
print(background_info)
print("Sample Characteristics Dictionary:")
print(sample_characteristics_dict)
# 1) Gene Expression Data Availability
# From the background info, it is clear this dataset uses a gene expression profiling platform (Clariom D),
# so we consider gene expression data to be available.
is_gene_available = True

# 2) Variable Availability and Data Type Conversion
# Based on the sample characteristics dictionary, all samples have "gender: female" (only one unique value),
# "tissue: primary luminal breast cancer" (only one unique value), "disease state: luminal breast cancer" (one unique value),
# and "loco-regional recurrence" with three categories (EARLY, INTERMEDIATE, LATE), which does not reflect the trait
# "Breast_Cancer" vs. "Non-Cancer", but rather time-to-recurrence categories. Therefore, no key actually
# provides a varying "Breast_Cancer" trait, and there is also no key for age. Thus, each variable is effectively unavailable.

trait_row = None
age_row = None
gender_row = None

# Define data conversion functions (they won't be used since all variables are unavailable),
# but we provide them as placeholders to follow instructions.

def convert_trait(val: str) -> int:
    # Placeholder: Not used, but implemented for completeness.
    # Suppose we parse after the colon, but since data is unavailable, return None.
    return None

def convert_age(val: str) -> float:
    # Placeholder: Not used, but implemented for completeness.
    return None

def convert_gender(val: str) -> int:
    # Placeholder: Not used, but implemented for completeness.
    return None

# 3) Save metadata using initial filtering
# Trait data availability depends on whether trait_row is None.
is_trait_available = (trait_row is not None)

is_usable = validate_and_save_cohort_info(
    is_final=False,
    cohort=cohort,
    info_path=json_path,
    is_gene_available=is_gene_available,
    is_trait_available=is_trait_available
)

# 4) Clinical Feature Extraction
# We only proceed if trait_row is not None. Otherwise, skip.
if trait_row is not None:
    selected_clinical_df = geo_select_clinical_features(
        clinical_data,
        trait="Breast_Cancer",
        trait_row=trait_row,
        convert_trait=convert_trait,
        age_row=age_row,
        convert_age=convert_age,
        gender_row=gender_row,
        convert_gender=convert_gender
    )
    print("Preview of clinical data:", preview_df(selected_clinical_df))
    selected_clinical_df.to_csv(out_clinical_data_file, index=False)
# STEP3
# 1. Use the get_genetic_data function from the library to get the gene_data from the matrix_file previously defined.
gene_data = get_genetic_data(matrix_file)

# 2. Print the first 20 row IDs (gene or probe identifiers) for future observation.
print(gene_data.index[:20])
# These identifiers (e.g., "TC0100006432.hg.1") are not standard HUGO gene symbols.
# Therefore, gene mapping is required.
print("requires_gene_mapping = True")
# STEP5
# 1. Use the 'get_gene_annotation' function from the library to get gene annotation data from the SOFT file.
gene_annotation = get_gene_annotation(soft_file)

# 2. Use the 'preview_df' function from the library to preview the data and print out the results.
print("Gene annotation preview:")
print(preview_df(gene_annotation))
# STEP6: Gene Identifier Mapping

# 1) Decide which columns match the gene expression data ID and the gene symbol.
#    From our inspection, the "ID" column contains the probe IDs matching the gene_data index.
#    The "gene_assignment" column contains text from which we can extract gene symbols.

# 2) Get a gene mapping dataframe using the library's get_gene_mapping function.
gene_mapping_df = get_gene_mapping(
    annotation=gene_annotation,
    prob_col='ID',
    gene_col='gene_assignment'
)

# 3) Apply the mapping to convert the probe-level expression data into gene-level data.
gene_data = apply_gene_mapping(gene_data, gene_mapping_df)

# To observe some information about the resulting gene_data, let's print its shape and a quick head.
print("Mapped gene_data shape:", gene_data.shape)
print("Head of the mapped gene_data:")
print(gene_data.head())
# STEP7

import pandas as pd

# 1. Normalize the obtained gene data using the NCBI Gene synonym database
normalized_gene_data = normalize_gene_symbols_in_index(gene_data)
normalized_gene_data.to_csv(out_gene_data_file)

# Since we determined earlier that trait data is not available (trait_row = None),
# "selected_clinical_df" was never created. We therefore have no clinical data to link,
# and the dataset is not usable for trait analysis. We'll handle final validation accordingly.

# 2-4. Skip linking, missing-value handling, and bias checking because trait data is unavailable
# Prepare a minimal placeholder DataFrame for final validation.
placeholder_df = pd.DataFrame()

# 5. Conduct final quality validation and save relevant information about the linked cohort data
# Since trait data is unavailable, is_trait_available=False, the dataset won't be usable.
# However, validate_and_save_cohort_info requires a boolean for is_biased when is_final=True.
# We'll set is_biased=True (forcing the dataset to be considered not usable).
is_usable = validate_and_save_cohort_info(
    is_final=True,
    cohort=cohort,
    info_path=json_path,
    is_gene_available=True,
    is_trait_available=False,
    is_biased=True,
    df=placeholder_df,
    note="Trait data not available; cannot link clinical and genetic data."
)

# 6. If the dataset is usable (which, given no trait, it won't be), save the final linked data
if is_usable:
    # Normally we would save the linked data, but here it will remain unavailable.
    pass