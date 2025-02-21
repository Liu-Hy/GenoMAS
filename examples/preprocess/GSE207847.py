# Path Configuration
from tools.preprocess import *

# Processing context
trait = "Breast_Cancer"
cohort = "GSE207847"

# Input paths
in_trait_dir = "../DATA/GEO/Breast_Cancer"
in_cohort_dir = "../DATA/GEO/Breast_Cancer/GSE207847"

# Output paths
out_data_file = "./output/preprocess/1/Breast_Cancer/GSE207847.csv"
out_gene_data_file = "./output/preprocess/1/Breast_Cancer/gene_data/GSE207847.csv"
out_clinical_data_file = "./output/preprocess/1/Breast_Cancer/clinical_data/GSE207847.csv"
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
# 1. Determine if the dataset contains suitable gene expression data
#    (based on the background: "We performed gene expression profile...")
is_gene_available = True

# 2. Variable Availability (trait, age, gender) and Data Type Conversions
#    From the sample characteristics dictionary, we observe:
#    - Row 0 has "disease state: luminal breast cancer" (only 1 unique value)
#    - Row 1 has "gender: female" (only 1 unique value)
#    - Row 2 has "tissue: primary luminal breast cancer" (only 1 unique value)
#    - Row 3 has "loco-regional recurrence: LATE/EARLY/INTERMEDIATE" (not our target trait)
#    => All data relevant to "Breast_Cancer" or "age" or "gender" shows no variation or is not provided.
#    Hence, we conclude all three variables are unavailable for this dataset.

trait_row = None
age_row = None
gender_row = None

# Define the data type conversion functions (though they won't be used here since all rows are None)

def convert_trait(value: str):
    """
    Convert trait data to binary or continuous.
    Since 'trait_row' is None, this function will not be used.
    In other contexts, we'd parse the value after the colon and map
    known variants to desired data type. Here, return None as placeholder.
    """
    return None

def convert_age(value: str):
    """
    Convert age data to a continuous variable.
    Since 'age_row' is None, this function will not be used.
    In other contexts, we'd parse the value after the colon,
    convert to float, handle unknown as None, etc.
    """
    return None

def convert_gender(value: str):
    """
    Convert gender data to binary (female=0, male=1).
    Since 'gender_row' is None, this function will not be used.
    """
    return None

# 3. Conduct initial filtering and save metadata
#    Trait data availability depends on whether trait_row is None
is_trait_available = (trait_row is not None)

is_usable = validate_and_save_cohort_info(
    is_final=False,
    cohort=cohort,
    info_path=json_path,
    is_gene_available=is_gene_available,
    is_trait_available=is_trait_available
)

# 4. Clinical Feature Extraction - Skip because trait_row is None
if is_trait_available:
    # Would extract and save clinical features if available.
    pass
# STEP3
# 1. Use the get_genetic_data function from the library to get the gene_data from the matrix_file previously defined.
gene_data = get_genetic_data(matrix_file)

# 2. Print the first 20 row IDs (gene or probe identifiers) for future observation.
print(gene_data.index[:20])
print("requires_gene_mapping = True")
# STEP5
# 1. Use the 'get_gene_annotation' function from the library to get gene annotation data from the SOFT file.
gene_annotation = get_gene_annotation(soft_file)

# 2. Use the 'preview_df' function from the library to preview the data and print out the results.
print("Gene annotation preview:")
print(preview_df(gene_annotation))
# STEP: Gene Identifier Mapping

# 1. Identify the columns in the gene annotation dataframe
#    - 'ID' matches the probe identifiers in the gene expression data
#    - 'gene_assignment' contains the gene symbols (within the text)
probe_col = 'ID'
gene_symbol_col = 'gene_assignment'

# 2. Get a gene mapping dataframe from the annotation
mapping_df = get_gene_mapping(gene_annotation, probe_col, gene_symbol_col)

# 3. Convert probe-level expression to gene-level expression
gene_data = apply_gene_mapping(gene_data, mapping_df)

# For validation, print the shape of the mapped gene_data
print("Mapped gene_data shape:", gene_data.shape)
import pandas as pd

# STEP7

# 1. Normalize the obtained gene data using the NCBI Gene synonym database
normalized_gene_data = normalize_gene_symbols_in_index(gene_data)
normalized_gene_data.to_csv(out_gene_data_file)

# 2. Since no clinical data was extracted (trait_row=None in previous steps),
#    define a placeholder clinical DataFrame so the code won't fail.
selected_clinical_df = pd.DataFrame()

# Attempt to link clinical and genetic data
linked_data = geo_link_clinical_genetic_data(selected_clinical_df, normalized_gene_data)

# If the trait column isn't present after linking, it implies no trait data is available.
if trait not in linked_data.columns:
    # Provide a dummy DataFrame and set is_biased=False so is_final=True doesn't error out
    empty_df = pd.DataFrame()
    validate_and_save_cohort_info(
        is_final=True,
        cohort=cohort,
        info_path=json_path,
        is_gene_available=True,
        is_trait_available=False,
        is_biased=False,      # Dummy value to satisfy function requirements
        df=empty_df,          # Dummy DataFrame
        note="No trait data available for this dataset."
    )
else:
    # 3. Handle missing values systematically
    linked_data_processed = handle_missing_values(linked_data, trait_col=trait)

    # 4. Check for biased trait and remove any biased demographic features
    trait_biased, linked_data_final = judge_and_remove_biased_features(linked_data_processed, trait)

    # 5. Final quality validation and metadata saving
    is_usable = validate_and_save_cohort_info(
        is_final=True,
        cohort=cohort,
        info_path=json_path,
        is_gene_available=True,
        is_trait_available=True,
        is_biased=trait_biased,
        df=linked_data_final,
        note="Dataset processed with GEO pipeline. Checked for missing values and bias."
    )

    # 6. If dataset is usable, save the final linked data
    if is_usable:
        linked_data_final.to_csv(out_data_file)