# Path Configuration
from tools.preprocess import *

# Processing context
trait = "Breast_Cancer"
cohort = "GSE248830"

# Input paths
in_trait_dir = "../DATA/GEO/Breast_Cancer"
in_cohort_dir = "../DATA/GEO/Breast_Cancer/GSE248830"

# Output paths
out_data_file = "./output/preprocess/1/Breast_Cancer/GSE248830.csv"
out_gene_data_file = "./output/preprocess/1/Breast_Cancer/gene_data/GSE248830.csv"
out_clinical_data_file = "./output/preprocess/1/Breast_Cancer/clinical_data/GSE248830.csv"
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
# 1) Determine if the dataset likely contains gene expression data.
#    From the background information, this dataset has "Targeted gene expression profiles ... using nCounter PanCancer IO 360™ Panel".
#    Hence, set is_gene_available to True.
is_gene_available = True

# 2) Check availability of variables: trait, age, gender
#    From the sample characteristics dictionary, we see:
#    - Row 0: 'age at diagnosis: ...'
#    - Row 1: 'Sex: female/male'
#    - Row 2: 'histology: ...', which helps distinguish "adenocaricnoma" (lung) vs. "TNBC"/"ER"/"HER2"/"PR" (breast).
trait_row = 2
age_row = 0
gender_row = 1

# 2.2) Define data conversion functions.

def convert_trait(x: str):
    """
    Convert histology to a binary indicator for 'Breast_Cancer': 
      - 1 if the histology suggests breast cancer 
      - 0 if it suggests lung adenocarcinoma
      - None for unknown or unrecognized patterns
    """
    parts = x.split(':', 1)
    if len(parts) < 2:
        return None
    val = parts[1].strip().lower()
    if 'adenocaricnoma' in val:
        return 0
    if 'tnbc' in val or 'her2' in val or 'er' in val or 'pr' in val:
        return 1
    if 'unknown' in val:
        return None
    return None

def convert_age(x: str):
    """
    Convert age at diagnosis to a continuous float. Return None if 'n.a.' or not a valid number.
    """
    parts = x.split(':', 1)
    if len(parts) < 2:
        return None
    val = parts[1].strip().lower()
    if val == 'n.a.':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def convert_gender(x: str):
    """
    Convert gender to a binary indicator: female -> 0, male -> 1, None otherwise.
    """
    parts = x.split(':', 1)
    if len(parts) < 2:
        return None
    val = parts[1].strip().lower()
    if val == 'female':
        return 0
    if val == 'male':
        return 1
    return None

# 3) Save metadata with initial filtering.
#    Trait data is available if trait_row is not None.
is_trait_available = (trait_row is not None)
usable_initial = validate_and_save_cohort_info(
    is_final=False,
    cohort=cohort,
    info_path=json_path,
    is_gene_available=is_gene_available,
    is_trait_available=is_trait_available
)

# 4) If trait data is available, extract and preview clinical features, then save to CSV.
if is_trait_available:
    selected_clinical_data = geo_select_clinical_features(
        clinical_df=clinical_data,  # Assume 'clinical_data' is a DataFrame already loaded
        trait=trait,                # "Breast_Cancer"
        trait_row=trait_row,
        convert_trait=convert_trait,
        age_row=age_row,
        convert_age=convert_age,
        gender_row=gender_row,
        convert_gender=convert_gender
    )

    # Preview extracted clinical data
    clinical_preview = preview_df(selected_clinical_data)
    print("Clinical Data Preview:", clinical_preview)

    # Save the clinical features to CSV
    selected_clinical_data.to_csv(out_clinical_data_file, index=False)
# STEP3
# 1. Use the get_genetic_data function from the library to get the gene_data from the matrix_file previously defined.
gene_data = get_genetic_data(matrix_file)

# 2. Print the first 20 row IDs (gene or probe identifiers) for future observation.
print(gene_data.index[:20])
# Based on the provided gene identifiers (A2M, ACVR1C, ADAM12, ADGRE1, ADM, ADORA2A, AKT1, etc.),
# they appear to be valid human gene symbols and do not require additional mapping.
# Concluding answer:
print("requires_gene_mapping = False")
# STEP5

# 1. Normalize the obtained gene data using the NCBI Gene synonym database
normalized_gene_data = normalize_gene_symbols_in_index(gene_data)
normalized_gene_data.to_csv(out_gene_data_file)

# 2. Link the clinical and genetic data
linked_data = geo_link_clinical_genetic_data(selected_clinical_data, normalized_gene_data)

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