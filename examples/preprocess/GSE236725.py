# Path Configuration
from tools.preprocess import *

# Processing context
trait = "Breast_Cancer"
cohort = "GSE236725"

# Input paths
in_trait_dir = "../DATA/GEO/Breast_Cancer"
in_cohort_dir = "../DATA/GEO/Breast_Cancer/GSE236725"

# Output paths
out_data_file = "./output/preprocess/1/Breast_Cancer/GSE236725.csv"
out_gene_data_file = "./output/preprocess/1/Breast_Cancer/gene_data/GSE236725.csv"
out_clinical_data_file = "./output/preprocess/1/Breast_Cancer/clinical_data/GSE236725.csv"
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
# 1. Determine if the dataset likely contains gene expression data
is_gene_available = True  # The study used Affymetrix microarrays, so it's gene expression data

# 2. Check variable availability
# The "disease state: breast cancer" field is constant (i.e., identical for all samples),
# so it does not provide variability for association analysis. Age and gender are not present.
trait_row = None
age_row = None
gender_row = None

# 3. Save metadata (initial filtering)
is_trait_available = (trait_row is not None)
is_usable = validate_and_save_cohort_info(
    is_final=False,
    cohort=cohort,
    info_path=json_path,
    is_gene_available=is_gene_available,
    is_trait_available=is_trait_available
)

# 4. If trait_row were available, we would extract clinical features here, but it's None, so we skip.
# STEP3
# 1. Use the get_genetic_data function from the library to get the gene_data from the matrix_file previously defined.
gene_data = get_genetic_data(matrix_file)

# 2. Print the first 20 row IDs (gene or probe identifiers) for future observation.
print(gene_data.index[:20])
# These identifiers (e.g., "1007_s_at", "1053_at") are Affymetrix probe IDs, not standard gene symbols.
requires_gene_mapping = True
# STEP5
# 1. Use the 'get_gene_annotation' function from the library to get gene annotation data from the SOFT file.
gene_annotation = get_gene_annotation(soft_file)

# 2. Use the 'preview_df' function from the library to preview the data and print out the results.
print("Gene annotation preview:")
print(preview_df(gene_annotation))
# STEP: Gene Identifier Mapping

# 1. Identify the columns for probe IDs and gene symbols 
#    ('ID' for probes, 'Gene Symbol' for gene symbols).

# 2. Extract the mapping between probe IDs and gene symbols into a DataFrame.
mapping_df = get_gene_mapping(gene_annotation, prob_col='ID', gene_col='Gene Symbol')

# 3. Convert probe-level measurements to gene expression data using the mapping information.
gene_data = apply_gene_mapping(gene_data, mapping_df)
# STEP7: Data Normalization and Partial Validation (No Trait Data)

# 1. Normalize gene symbols in the obtained gene expression data using synonym information from the NCBI Gene database.
#    Remove data of unrecognized gene symbols, and average the expression values of gene symbols that are mapped to the
#    same standard symbol. Save the normalized data as a CSV file to out_gene_data_file.
normalized_gene_data = normalize_gene_symbols_in_index(gene_data)
normalized_gene_data.to_csv(out_gene_data_file)

# Since we do not have a trait (trait_row was None in previous steps), we cannot perform a final trait-based analysis.
# Therefore, we record partial validation with is_final=False, so we do not need to provide df or is_biased.
is_trait_available = False
is_gene_available = True

is_usable = validate_and_save_cohort_info(
    is_final=False,
    cohort=cohort,
    info_path=json_path,
    is_gene_available=is_gene_available,
    is_trait_available=is_trait_available
)

# There is no trait data to link or validate further, so we do not perform additional steps here.
# is_usable is expected to be False, indicating we cannot proceed with final usage.