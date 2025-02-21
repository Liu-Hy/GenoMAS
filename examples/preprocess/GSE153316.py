# Path Configuration
from tools.preprocess import *

# Processing context
trait = "Breast_Cancer"
cohort = "GSE153316"

# Input paths
in_trait_dir = "../DATA/GEO/Breast_Cancer"
in_cohort_dir = "../DATA/GEO/Breast_Cancer/GSE153316"

# Output paths
out_data_file = "./output/preprocess/1/Breast_Cancer/GSE153316.csv"
out_gene_data_file = "./output/preprocess/1/Breast_Cancer/gene_data/GSE153316.csv"
out_clinical_data_file = "./output/preprocess/1/Breast_Cancer/clinical_data/GSE153316.csv"
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
is_gene_available = True  # "Gene expression profiles" suggests it is indeed gene expression data.

# 2) Variable Availability and Data Type Conversion

# Based on the sample characteristics, the 'trait' variable ("Breast_Cancer") is constant for all samples
# (they're all mastectomy patients); hence it's not useful for association (only one unique value).
trait_row = None

# For age, row 2 has multiple distinct values like "age: 39", "age: 36", etc.
age_row = 2

# For gender, there is no relevant row in the dictionary.
gender_row = None

# 2.2 Define the data conversion functions.
# Even if the variable is not used (trait_row = None, gender_row = None), we still define them per instructions.

def convert_trait(x: str):
    # The trait "Breast_Cancer" is constant, so we skip detailed parsing.
    # Return None to indicate no meaningful variation.
    return None

def convert_age(x: str):
    # Example format: "age: 39"
    # Extract the part after the colon and convert to float if possible.
    try:
        val_str = x.split(":", 1)[1].strip()
        return float(val_str)
    except:
        return None

def convert_gender(x: str):
    # No actual data available, but define a stub for completeness.
    # Return None always in this dataset.
    return None

# 3) Save Metadata using initial filtering
is_trait_available = (trait_row is not None)
validate_and_save_cohort_info(
    is_final=False,
    cohort=cohort,
    info_path=json_path,
    is_gene_available=is_gene_available,
    is_trait_available=is_trait_available
)

# 4) Clinical Feature Extraction: Skip because trait_row is None (no trait data available)
# STEP3
# 1. Use the get_genetic_data function from the library to get the gene_data from the matrix_file previously defined.
gene_data = get_genetic_data(matrix_file)

# 2. Print the first 20 row IDs (gene or probe identifiers) for future observation.
print(gene_data.index[:20])
# Based on visual inspection, these 'AFFX' prefixes are typically Affymetrix probe/control IDs rather than standard human gene symbols.
# Therefore, they require mapping to standard gene symbols.
print("requires_gene_mapping = True")
# STEP5
# 1. Use the 'get_gene_annotation' function from the library to get gene annotation data from the SOFT file.
gene_annotation = get_gene_annotation(soft_file)

# 2. Use the 'preview_df' function from the library to preview the data and print out the results.
print("Gene annotation preview:")
print(preview_df(gene_annotation))
# STEP: Gene Identifier Mapping

# 1) Identify the columns for probe IDs and gene symbols based on the annotation preview.
#    From inspection, "ID" in the annotation matches the probe ID in the expression data,
#    and "SPOT_ID.1" contains the textual gene symbol information.

# 2) Build the mapping dataframe.
mapping_df = get_gene_mapping(gene_annotation, prob_col='ID', gene_col='SPOT_ID.1')

# 3) Convert the probe-level data to gene-level data using the mapping.
gene_data = apply_gene_mapping(gene_data, mapping_df)

# (Optional) Print a small preview to confirm structure.
print("Gene expression data (mapped) preview:")
print(preview_df(gene_data))
# STEP7

# 1. Normalize the obtained gene data using the NCBI Gene synonym database
normalized_gene_data = normalize_gene_symbols_in_index(gene_data)
normalized_gene_data.to_csv(out_gene_data_file)

# Since trait_row was None, there is no usable trait data.
# Hence, it's not possible to perform final linking or bias checking for association studies.

# 2. We record dataset metadata indicating that it lacks trait data (so it's not usable).
validate_and_save_cohort_info(
    is_final=False,  # We only do the initial validation because trait is unavailable
    cohort=cohort,
    info_path=json_path,
    is_gene_available=True,
    is_trait_available=False
)

# 3. Because there's no trait data, we skip linking, bias checking, and final saving.