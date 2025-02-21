# Path Configuration
from tools.preprocess import *

# Processing context
trait = "Breast_Cancer"
cohort = "GSE270721"

# Input paths
in_trait_dir = "../DATA/GEO/Breast_Cancer"
in_cohort_dir = "../DATA/GEO/Breast_Cancer/GSE270721"

# Output paths
out_data_file = "./output/preprocess/1/Breast_Cancer/GSE270721.csv"
out_gene_data_file = "./output/preprocess/1/Breast_Cancer/gene_data/GSE270721.csv"
out_clinical_data_file = "./output/preprocess/1/Breast_Cancer/clinical_data/GSE270721.csv"
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
is_gene_available = True  # HTA 2.0 microarrays indicate gene expression data

# 2. Variable Availability and Data Type Conversion
#    Based on the sample characteristics, only 'age' has multiple non-constant values.
trait_row = None   # Not found or constant (all are breast cancer patients)
age_row = 2        # Key with age information
gender_row = None  # No gender information found

def convert_trait(value: str):
    # No trait data row, so not applicable in this cohort
    return None

def convert_age(value: str):
    # The format seems to be "age: 78.00" or "age: not available"
    # Extract the substring after ':'
    parts = value.split(':', 1)
    if len(parts) < 2:
        return None
    val_str = parts[1].strip().lower()
    if val_str == "not available":
        return None
    try:
        return float(val_str)
    except ValueError:
        return None

def convert_gender(value: str):
    # No gender data row, so not applicable
    return None

# 3. Save Metadata (initial filtering)
#    Trait is considered unavailable since trait_row is None
is_trait_available = (trait_row is not None)
is_usable = validate_and_save_cohort_info(
    is_final=False,
    cohort=cohort,
    info_path=json_path,
    is_gene_available=is_gene_available,
    is_trait_available=is_trait_available
)

# 4. Clinical Feature Extraction
#    Skip if trait_row is None
if trait_row is not None:
    # We would perform clinical data extraction here, but trait_row is None in this case.
    pass
# STEP3
# 1. Use the get_genetic_data function from the library to get the gene_data from the matrix_file previously defined.
gene_data = get_genetic_data(matrix_file)

# 2. Print the first 20 row IDs (gene or probe identifiers) for future observation.
print(gene_data.index[:20])
# The given identifiers (e.g., TC01000001.hg.1) are not recognizable standard human gene symbols.
# They likely need mapping to official gene symbols.
print("requires_gene_mapping = True")
# STEP5
# 1. Use the 'get_gene_annotation' function from the library to get gene annotation data from the SOFT file.
gene_annotation = get_gene_annotation(soft_file)

# 2. Use the 'preview_df' function from the library to preview the data and print out the results.
print("Gene annotation preview:")
print(preview_df(gene_annotation))
# STEP: Gene Identifier Mapping

# 1. Identify the relevant columns in gene_annotation for probe IDs and gene symbols.
#    From the preview, "ID" matches the row IDs in our gene_data, and "gene_assignment" holds gene symbol info.

mapping_df = get_gene_mapping(gene_annotation, prob_col="ID", gene_col="gene_assignment")

# 2. Convert probe-level measurements to gene-level by applying the mapping.
gene_data = apply_gene_mapping(gene_data, mapping_df)
# STEP8

# 1. Normalize the obtained gene data with the 'normalize_gene_symbols_in_index' function
normalized_gene_data = normalize_gene_symbols_in_index(gene_data)
normalized_gene_data.to_csv(out_gene_data_file)

# Because trait data is not available (trait_row was None), we skip linking clinical data and trait-based analyses.

# 2. Perform final validation and save cohort info.
#    Per the library requirements, we must provide 'df' and 'is_biased' even though trait is unavailable.
#    Setting 'is_biased=False' does not indicate the trait is balanced; rather, we are forced to supply a boolean.
#    The function will mark the dataset as unusable because is_trait_available=False.
validate_and_save_cohort_info(
    is_final=True,
    cohort=cohort,
    info_path=json_path,
    is_gene_available=True,
    is_trait_available=False,
    df=normalized_gene_data,
    is_biased=False,
    note="No trait or demographic data is available for association analysis."
)

# 3. Since the dataset is not usable for trait-based analysis, we do not save any final linked data.