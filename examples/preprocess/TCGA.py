# Path Configuration
from tools.preprocess import *

# Processing context
trait = "Breast_Cancer"

# Input paths
tcga_root_dir = "../DATA/TCGA"

# Output paths
out_data_file = "./output/preprocess/1/Breast_Cancer/TCGA.csv"
out_gene_data_file = "./output/preprocess/1/Breast_Cancer/gene_data/TCGA.csv"
out_clinical_data_file = "./output/preprocess/1/Breast_Cancer/clinical_data/TCGA.csv"
json_path = "./output/preprocess/1/Breast_Cancer/cohort_info.json"

import os
import pandas as pd

# 1) Identify the subdirectory for "Breast_Cancer"
cohort_name = "TCGA_Breast_Cancer_(BRCA)"  # Found by matching "Breast_Cancer" with the list of directories
cohort_dir = os.path.join(tcga_root_dir, cohort_name)

# 2) Identify the paths to clinical and genetic files
clinical_file_path, genetic_file_path = tcga_get_relevant_filepaths(cohort_dir)

# 3) Load the files as Pandas DataFrames
clinical_df = pd.read_csv(clinical_file_path, index_col=0, sep='\t')
genetic_df = pd.read_csv(genetic_file_path, index_col=0, sep='\t')

# 4) Print the column names of the clinical DataFrame
print(clinical_df.columns.tolist())
# 1) Identify the candidate columns
candidate_age_cols = ['Age_at_Initial_Pathologic_Diagnosis_nature2012', 'age_at_initial_pathologic_diagnosis']
candidate_gender_cols = ['Gender_nature2012', 'gender']

# Print them in the specified format
print(f"candidate_age_cols = {candidate_age_cols}")
print(f"candidate_gender_cols = {candidate_gender_cols}")

# 2) Extract and preview the candidate columns from the clinical data
age_subset = clinical_df[candidate_age_cols] if candidate_age_cols else pd.DataFrame()
gender_subset = clinical_df[candidate_gender_cols] if candidate_gender_cols else pd.DataFrame()

if not age_subset.empty:
    print("Age subset preview:")
    print(preview_df(age_subset, n=5))

if not gender_subset.empty:
    print("Gender subset preview:")
    print(preview_df(gender_subset, n=5))
# Based on the previews, we see that the second candidate age column ('age_at_initial_pathologic_diagnosis')
# contains valid age values, while the first only has NaN. Similarly, the second candidate gender column ('gender')
# contains valid gender values, while the first only has NaN.

age_col = "age_at_initial_pathologic_diagnosis"
gender_col = "gender"

print("Selected age_col:", age_col)
print("Selected gender_col:", gender_col)
# 1) Extract and standardize clinical features (trait, age, gender) from the TCGA data
selected_clinical_df = tcga_select_clinical_features(
    clinical_df=clinical_df,
    trait=trait,
    age_col=age_col,
    gender_col=gender_col
)

# 2) Normalize gene symbols in the gene expression data
genetic_df_normalized = normalize_gene_symbols_in_index(genetic_df)
genetic_df_normalized.to_csv(out_gene_data_file)

# 3) Link clinical and genetic data on sample IDs
gene_expr_t = genetic_df_normalized.T
linked_data = selected_clinical_df.join(gene_expr_t, how='inner')

# 4) Handle missing values in the linked data
linked_data = handle_missing_values(linked_data, trait)

# 5) Determine whether the trait and some demographic features are severely biased
trait_biased, linked_data = judge_and_remove_biased_features(linked_data, trait)

# 6) Validate and save cohort information
is_usable = validate_and_save_cohort_info(
    is_final=True,
    cohort="TCGA",
    info_path=json_path,
    is_gene_available=True,
    is_trait_available=True,
    is_biased=trait_biased,
    df=linked_data,
    note="Prostate Cancer data from TCGA."
)

# 7) If usable, save the final linked data, including clinical and genetic features
if is_usable:
    linked_data.to_csv(out_data_file)
    # Save clinical subset if present
    clinical_cols = [col for col in [trait, "Age", "Gender"] if col in linked_data.columns]
    if clinical_cols:
        linked_data[clinical_cols].to_csv(out_clinical_data_file)