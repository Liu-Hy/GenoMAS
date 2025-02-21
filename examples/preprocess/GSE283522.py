# Path Configuration
from tools.preprocess import *

# Processing context
trait = "Breast_Cancer"
cohort = "GSE283522"

# Input paths
in_trait_dir = "../DATA/GEO/Breast_Cancer"
in_cohort_dir = "../DATA/GEO/Breast_Cancer/GSE283522"

# Output paths
out_data_file = "./output/preprocess/1/Breast_Cancer/GSE283522.csv"
out_gene_data_file = "./output/preprocess/1/Breast_Cancer/gene_data/GSE283522.csv"
out_clinical_data_file = "./output/preprocess/1/Breast_Cancer/clinical_data/GSE283522.csv"
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
import re

# 1. Gene Expression Data Availability
# Based on the background describing RNA-sequencing (mFISHseq), this dataset likely contains gene expression data.
is_gene_available = True

# 2. Variable Availability and Conversions

# 2.1 Identify rows in the Sample Characteristics Dictionary
#    Trait: row 1 (contains "isolate: breast cancer patient", "isolate: healthy individual", etc.)
trait_row = 1

#    Age: row 2 (contains "age: 55 - 59", "age: 70 - 74", etc.)
age_row = 2

#    Gender: row 5 (contains "Sex: female", "Sex: male", etc.)
gender_row = 5

# 2.2 Define data type conversions
def convert_trait(value: str):
    """
    Convert the value in row 1 into a binary indicator for breast cancer.
    'isolate: breast cancer patient' -> 1
    'isolate: healthy individual' -> 0
    otherwise -> None
    """
    parts = value.split(':', 1)
    if len(parts) < 2:
        return None
    v = parts[1].strip().lower()
    if 'breast cancer patient' in v:
        return 1
    elif 'healthy individual' in v:
        return 0
    else:
        return None

def convert_age(value: str):
    """
    Convert the value in row 2 into a continuous numeric age.
    Example: 'age: 55 - 59' -> 57 (midpoint), 'age: not applicable' -> None
    """
    parts = value.split(':', 1)
    if len(parts) < 2:
        return None
    range_str = parts[1].strip().lower()
    if 'not applicable' in range_str:
        return None
    # Attempt to extract numeric values:
    digits = re.findall(r'\d+', range_str)
    if len(digits) == 2:
        low, high = map(int, digits)
        return (low + high) / 2
    elif len(digits) == 1:
        return int(digits[0])
    else:
        return None

def convert_gender(value: str):
    """
    Convert the value in row 5 into a binary indicator for gender.
    'Sex: female' -> 0
    'Sex: male' -> 1
    otherwise -> None
    """
    parts = value.split(':', 1)
    if len(parts) < 2:
        return None
    v = parts[1].strip().lower()
    if v == 'female':
        return 0
    elif v == 'male':
        return 1
    else:
        return None

# 3. Save Metadata with initial filtering
is_trait_available = (trait_row is not None)
is_usable = validate_and_save_cohort_info(
    is_final=False,
    cohort=cohort,
    info_path=json_path,
    is_gene_available=is_gene_available,
    is_trait_available=is_trait_available
)

# 4. Clinical Feature Extraction (only if trait_row is available)
if trait_row is not None:
    selected_clinical = geo_select_clinical_features(
        clinical_data,
        trait=trait,
        trait_row=trait_row,
        convert_trait=convert_trait,
        age_row=age_row,
        convert_age=convert_age,
        gender_row=gender_row,
        convert_gender=convert_gender
    )
    # Observe the extracted clinical dataframe
    preview = preview_df(selected_clinical)
    print("Preview of selected clinical features:", preview)

    # Save clinical data to CSV
    selected_clinical.to_csv(out_clinical_data_file, index=False)
# STEP3
# 1. Use the get_genetic_data function from the library to get the gene_data from the matrix_file previously defined.
gene_data = get_genetic_data(matrix_file)

# 2. Print the first 20 row IDs (gene or probe identifiers) for future observation.
print(gene_data.index[:20])