# Path Configuration
from tools.preprocess import *

# Processing context
trait = "Breast_Cancer"
cohort = "GSE249377"

# Input paths
in_trait_dir = "../DATA/GEO/Breast_Cancer"
in_cohort_dir = "../DATA/GEO/Breast_Cancer/GSE249377"

# Output paths
out_data_file = "./output/preprocess/1/Breast_Cancer/GSE249377.csv"
out_gene_data_file = "./output/preprocess/1/Breast_Cancer/gene_data/GSE249377.csv"
out_clinical_data_file = "./output/preprocess/1/Breast_Cancer/clinical_data/GSE249377.csv"
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
# 1. Gene Expression Data Availability
is_gene_available = True  # From background info, this dataset provides transcriptomic (gene expression) data.

# 2. Variable Availability and Data Type Conversion
#    After reviewing the sample characteristics, none of the rows provide distinct "Breast_Cancer" statuses,
#    nor do they provide "age" or "gender" information. The experiment uses only MCF7 (a breast cancer cell line),
#    which does not vary among samples in a way that is useful for association studies.
trait_row = None
age_row = None
gender_row = None

# Define conversion functions (they won't be used here, but we must still define them):
def convert_trait(value: str) -> Optional[Union[float, int]]:
    return None  # No trait row available, so always return None

def convert_age(value: str) -> Optional[float]:
    return None  # No age row available, so always return None

def convert_gender(value: str) -> Optional[int]:
    return None  # No gender row available, so always return None

# 3. Save Metadata (initial filtering)
#    If trait_row is None, is_trait_available should be False
is_usable = validate_and_save_cohort_info(
    is_final=False,
    cohort=cohort,
    info_path=json_path,
    is_gene_available=is_gene_available,
    is_trait_available=(trait_row is not None)
)

# 4. Clinical Feature Extraction
#    Since trait_row is None, we do not perform clinical feature extraction and skip this step.
# STEP3
# 1. Use the get_genetic_data function from the library to get the gene_data from the matrix_file previously defined.
gene_data = get_genetic_data(matrix_file)

# 2. Print the first 20 row IDs (gene or probe identifiers) for future observation.
print(gene_data.index[:20])