# Path Configuration
from tools.preprocess import *

# Processing context
trait = "Breast_Cancer"
cohort = "GSE234017"

# Input paths
in_trait_dir = "../DATA/GEO/Breast_Cancer"
in_cohort_dir = "../DATA/GEO/Breast_Cancer/GSE234017"

# Output paths
out_data_file = "./output/preprocess/1/Breast_Cancer/GSE234017.csv"
out_gene_data_file = "./output/preprocess/1/Breast_Cancer/gene_data/GSE234017.csv"
out_clinical_data_file = "./output/preprocess/1/Breast_Cancer/clinical_data/GSE234017.csv"
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
# 1. Decide if gene expression data is likely available
is_gene_available = True  # Spatial transcriptomics indicates gene expression data

# 2. Identify variable availability
# From the sample characteristics dictionary, row 2 ("genotype: WT/BRCA1/BRCA2") 
# best reflects the trait "Breast_Cancer" in a binary manner (WT vs BRCA-mutated)
trait_row = 2
age_row = None
gender_row = None

# 2.2 Data Type Conversions
def convert_trait(value: str):
    parts = value.split(':')
    if len(parts) < 2:
        return None
    val = parts[1].strip()
    # WT => 0, BRCA1 => 1, BRCA2 => 1
    if val == "WT":
        return 0
    elif val in ["BRCA1", "BRCA2"]:
        return 1
    return None

def convert_age(value: str):
    # No age data is provided
    return None

def convert_gender(value: str):
    # No gender data is provided
    return None

# 3. Save metadata with initial filtering
is_trait_available = (trait_row is not None)
_ = validate_and_save_cohort_info(
    is_final=False,
    cohort=cohort,
    info_path=json_path,
    is_gene_available=is_gene_available,
    is_trait_available=is_trait_available
)

# 4. Clinical Feature Extraction (only if trait_row is not None)
if trait_row is not None:
    selected_clinical_df = geo_select_clinical_features(
        clinical_df=clinical_data,
        trait=trait,
        trait_row=trait_row,
        convert_trait=convert_trait,
        age_row=age_row,
        convert_age=convert_age,
        gender_row=gender_row,
        convert_gender=convert_gender
    )

    preview_result = preview_df(selected_clinical_df, n=5, max_items=200)
    print("Preview of selected clinical features:", preview_result)
    selected_clinical_df.to_csv(out_clinical_data_file, index=False)
# STEP3
# 1. Use the get_genetic_data function from the library to get the gene_data from the matrix_file previously defined.
gene_data = get_genetic_data(matrix_file)

# 2. Print the first 20 row IDs (gene or probe identifiers) for future observation.
print(gene_data.index[:20])
# Based on the observed identifiers, they do not appear to be standard human gene symbols.
# Thus, they likely require mapping to official gene symbols.
print("They appear to be some form of platform-based IDs.")
print("requires_gene_mapping = True")
# STEP5
# 1. Use the 'get_gene_annotation' function from the library to get gene annotation data from the SOFT file.
gene_annotation = get_gene_annotation(soft_file)

# 2. Use the 'preview_df' function from the library to preview the data and print out the results.
print("Gene annotation preview:")
print(preview_df(gene_annotation))
# STEP: Gene Identifier Mapping

# 1. Identify the columns in the gene annotation that contain the same IDs as in 'gene_data' 
#    and the column that contains the gene symbols ("ID" for probe identifiers, "ORF" for gene symbols).
mapping_df = get_gene_mapping(gene_annotation, prob_col='ID', gene_col='ORF')

# 2. Convert probe-level measurements to gene expression data using the mapping dataframe.
gene_data = apply_gene_mapping(gene_data, mapping_df)

# 3. Print the shape and a small preview of the resulting gene-level expression dataframe.
print("Gene data shape after mapping:", gene_data.shape)
print("Preview of mapped gene data:", preview_df(gene_data, n=5, max_items=20))
# STEP7

# 1. Normalize the obtained gene data using the NCBI Gene synonym database
normalized_gene_data = normalize_gene_symbols_in_index(gene_data)
normalized_gene_data.to_csv(out_gene_data_file)

# 2. Link the clinical and genetic data
linked_data = geo_link_clinical_genetic_data(selected_clinical_df, normalized_gene_data)

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