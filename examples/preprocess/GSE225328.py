# Path Configuration
from tools.preprocess import *

# Processing context
trait = "Breast_Cancer"
cohort = "GSE225328"

# Input paths
in_trait_dir = "../DATA/GEO/Breast_Cancer"
in_cohort_dir = "../DATA/GEO/Breast_Cancer/GSE225328"

# Output paths
out_data_file = "./output/preprocess/1/Breast_Cancer/GSE225328.csv"
out_gene_data_file = "./output/preprocess/1/Breast_Cancer/gene_data/GSE225328.csv"
out_clinical_data_file = "./output/preprocess/1/Breast_Cancer/clinical_data/GSE225328.csv"
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
# 1. Determine if gene expression data is available
#    From the background ("Transcriptome profiling"), we consider this dataset as containing gene expression data
is_gene_available = True

# 2. Identify the corresponding rows for each variable in the sample characteristics dictionary
#    Here, both 'disease' and 'Sex' have only one unique value ("early-stage luminal breast cancer" and "female"),
#    so they offer no variability for association studies. Hence, we consider them unavailable.
trait_row = None
age_row = None
gender_row = None

# Define data conversion functions.
# Since our identified rows are None, we won't actually use these functions,
# but we still define them as requested.
def convert_trait(value: str):
    return None

def convert_age(value: str):
    return None

def convert_gender(value: str):
    return None

# 3. Conduct initial filtering of usability
is_trait_available = (trait_row is not None)
is_usable = validate_and_save_cohort_info(
    is_final=False,
    cohort=cohort,
    info_path=json_path,
    is_gene_available=is_gene_available,
    is_trait_available=is_trait_available
)

# 4. Because trait_row is None, we skip clinical feature extraction and saving.