ROLE_PROMPT: str = \
        """You are a data engineer in a biomedical research team. Your goal is to write code for wrangling biomedical data. 
In this project, you will focus on wrangling Series data from the GEO database."""

GUIDELINES: str = \
        """
High-Level Guidelines for Data Wrangling Task:
First, import the necessary functions and classes from the utility library at the beginning of the code and define paths for the raw and processed data.
Next, identify and read relevant files to gather background and sample characteristics data. Print the extracted information for reference in subsequent steps.
Check for gene expression data availability. Assess the availability of key variables (trait, age, gender), identify their data keys, and choose appropriate data types. Write conversion functions for these variables to ensure proper data formatting.
Select relevant clinical features based on the identified variables and print a preview of the selected clinical data.
Extract genetic data from the matrix file and print the first 20 row IDs for reference.
Determine if gene identifiers need mapping to gene symbols. If mapping is needed, use gene annotation data to perform the mapping and print a preview.
Finally, normalize gene data and merge it with clinical data. Assess and remove any biased features from the merged dataset. Save cohort information and, if the dataset is not biased, save the processed data to a file.
"""

TOOLS: str = \
        """Tools:
"utils.preprocess" provides lots of well-developed helper functions for this project. Henceforth, it will be referred to as "the library." Please import and use functions from the library when possible. Below is the source code:
{utils_code}
        """

SETUPS: str = \
        """
Setups:
1. Path to the raw GEO dataset for preprocessing: {in_cohort_dir}
2. Directory to save the preprocessed GEO dataset: {output_dir}

NOTICE 1: The overall preprocessing requires multiple code snippets, each based on the execution results of the previous snippets. 
Consequently, the instructions will be divided into multiple STEPS. Each STEP requires you to write a code snippet, and then the execution result will be given to you for either revision of the current STEP or progression to the next STEP.

NOTICE 2: Please import all functions and classes in 'utils.preprocess' at the beginning of the code:
'from utils.preprocess import *'

Based on the context, write code to follow the instructions.
"""
# Also, the trait name has been assigned to the string variable "trait." To make the code more general, please use the variable instead of the string literal.
INSTRUCTION_STEP1: str = \
        """
        STEP1:
1. Identify the paths to the soft file and the matrix file, and assign them to the variables 'soft_file' and 'matrix_file'.
2. Read the matrix file to obtain background information about the dataset and sample characteristics data through the 'get_background_and_clinical_data' function from the library. For the input parameters to the function,
'background_prefixes' should be a list consisting of strings '!Series_title', '!Series_summary', and '!Series_overall_design'. 'clinical_prefixes' should be a list consisting of '!Sample_geo_accession' and '!Sample_characteristics_ch1'.
3. Obtain the sample characteristics dictionary from the clinical dataframe via the 'get_unique_values_by_row' function from the utils
4. Explicitly print out the all the background information and the sample characteristics dictionary. This information is for STEP2 to further write code.
        """

CODE_STEP1: str = \
        """# STEP1
from utils.preprocess import *
# 1. Identify the paths to the soft file and the matrix file
cohort_dir = '{in_cohort_dir}'
soft_file, matrix_file = geo_get_relevant_filepaths(cohort_dir)

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
    """

INSTRUCTION_STEP2: str = \
        """
STEP2: Dataset Analysis and Questions

As a biomedical research team, we are analyzing datasets to study the association between the human trait '{trait}' and genetic factors, considering the possible influence of age and gender. After searching the GEO database and parsing the matrix file of a series, STEP 1 has provided background information and sample characteristics data. Please review the output from STEP 1 and answer the following questions regarding this dataset:

1. Gene Expression Data Availability
   - Is this dataset likely to contain gene expression data? (Note: Pure miRNA data or methylation data are not suitable.)
     - If YES, set `is_gene_available` to `True`. Otherwise set it to `False`.

2. Variable Availability and Data Type Conversion
   For each of the variables '{trait}', 'age', and 'gender', address the following points:

   **2.1 Data Availability**
     - If human data for this variable is available, identify the key in the sample characteristics dictionary where unique values of this variable are recorded. The key is an integer. The variable information might be explicitly recorded or inferred from the field with biomedical knowledge or understanding of the dataset background.
     - If you can't find such a key, the data is not available. Constant features are useless in our associative studies. Therefore, if there is only one unique value under the key, consider the variable as not available. Similarly, if you infer the value from places other than the sample characteristics data, like "everyone has the same value for some variable", consider it as not available.
     - Name the keys `trait_row`, `age_row`, and `gender_row`, respectively. Use None to indicate the corresponding data is not available.

   **2.3 Data Type Conversion**
     - Choose an appropriate data type for each variable ('continuous' or 'binary').
     - If the data type is binary, convert values to 0 and 1. For gender data, convert female to 0 and male to 1.
     - Write a Python function to convert any given value of the variable to this data type. Typically, a colon (':') separates the header and the value in each cell, so ensure to extract the value after the colon in the function. Unknown values should be converted to None.
     - When data is not explicitly given but can be inferred, carefully observe the unique values in the sample characteristics dictionary, think about what they mean in this dataset, and design a heuristic rule to convert those values into the chosen type. If you are 90% sure that some cases should be mapped to a known value, please do that instead of giving `None`. 
     - Name the functions `convert_trait`, `convert_age`, and `convert_gender`, respectively.

3. Save Metadata
   Please save cohort information by following the function call format strictly:
   ```python
   save_cohort_info('{cohort}', '{json_path}', is_gene_available, trait_row is not None)
   ```

4. Clinical Feature Extraction
   If trait_row is not None, it means clinical data is available, then you MUST DO this substep. Otherwise, you should skip this subskip.
   Use the `geo_select_clinical_features` function to obtain the output `selected_clinical_data` from the input dataframe, save it to a csv file, and preview it. Follow the function call format strictly:
    ```python
    selected_clinical_data = geo_select_clinical_features(clinical_data, '{trait}', trait_row, convert_trait, age_row, convert_age, gender_row, convert_gender)
    csv_path = '{out_trait_data_file}'
    selected_clinical_data.to_csv(csv_path)
    print(preview_df(selected_clinical_data))
    ```
   - Note: `clinical_data` has been previously defined, and all functions have been imported from utils.preprocess in previous steps.
   - Do not comment out the function call.

[Output of STEP 1]
        """

INSTRUCTION_STEP3: str = \
        """
STEP3:
1. Use the get_genetic_data function from the library to get the gene_data from the matrix_file previously defined.
2. print the first 20 row ids for following step.
        """

CODE_STEP3: str = \
        """# STEP3
# 1. Use the get_genetic_data function from the library to get the gene_data from the matrix_file previously defined.
gene_data = get_genetic_data(matrix_file)

# 2. Print the first 20 row ids for the following step.
print(gene_data.index[:20])
    """

INSTRUCTION_STEP4: str = \
        """
STEP4:
Given the row headers (from STEP3) of a gene expression dataset in GEO. Based on your biomedical knowledge, are they human gene symbols, or are they some other identifiers that need to be mapped to gene symbols? Your answer should be concluded by starting a new line and strictly following this format:
requires_gene_mapping = (True or False)

[Output of STEP3]
        """

INSTRUCTION_STEP5: str = \
        """
STEP5:
If requires_gene_mapping is True, do the following substeps 1-2; otherwise, skip STEP5.
    1. Use the 'get_gene_annotation' function from the library to get gene annotation data from the soft file
    2. Use the 'preview_df' function from the library to preview the data and print out the results for the following step. Mark the printing with a string like "Gene annotation".
        """

CODE_STEP5: str = \
        """# STEP5
# 1. Use the 'get_gene_annotation' function from the library to get gene annotation data from the soft file.
gene_annotation = get_gene_annotation(soft_file)

# 2. Use the 'preview_df' function from the library to preview the data and print out the results.
print("Gene annotation preview:")
print(preview_df(gene_annotation))
    """

INSTRUCTION_STEP6: str = \
        """
STEP6:
If requires_gene_mapping is True, do the following substeps; otherwise, MUST SKIP them.
    1. When analyzing a gene expression dataset, we need to map some identifiers of genes to actual gene symbols. STEP3 prints out some of those identifiers, 
    and STEP5 prints out part of the gene annotation data converted to a Python dictionary. 
    Please read the dictionary and decide which key stores the same kind of identifiers as in STEP3, and which key stores the gene symbols, 
    or a new type of identifier that's common enough to be mapped to gene symbols. 
    The new type should be one of 'symbol', 'entrezgene', 'ensembl.gene', or other strings that are valid fields in the mygene library.
    Please strictly follow this format in your answer:
    org_identifier_key = 'key_name1'
    new_identifier_key = 'key_name2'
    new_identifier_type = 'symbol'  # or another field
    2. Get the dataframe storing the mapping between the original and new gene IDs using the 'get_gene_mapping' function from the library. 
    3. Apply the mapping with the 'apply_gene_mapping' function from the library, and name the resulting gene expression dataframe "gene_data". 
       """

INSTRUCTION_STEP7: str = \
        """
STEP7:
1. Normalize the obtained gene data with the 'normalize_gene_identifiers_in_index' function from the library. Save the normalized genetic data to a csv file in the path `{out_gene_data_file}`
2. Merge the clinical and genetic data with the 'geo_merge_clinical_genetic_data' function from the library, and assign the merged data to a variable 'merged_data'.
3. Determine whether the trait '{trait}' and some demographic attributes in the data is severely biased, and remove biased attributes with the 'judge_and_remove_biased_features' function from the library.
4. Save the cohort information with the 'save_cohort_info' function from the library. Hint: set the 'json_path' variable to '{json_path}', and assuming 'is_trait_biased' indicates whether the trait is biased, follow the function call format strictly to save the cohort information: 
save_cohort_info(cohort, json_path, True, True, is_trait_biased, merged_data).
5. If the trait in the data is not severely biased (regardless of whether the other attributes are biased), save the merged data to a csv file, in the path '{out_data_file}'. Otherwise, you must not save it.
        """

CODE_STEP7: str = \
        """# STEP7
# 1. Normalize the obtained gene data with the 'normalize_gene_identifiers_in_index' function from the library.
normalized_gene_data = normalize_gene_identifiers_in_index(gene_data, new_identifier_type)
gene_csv_path = '{out_gene_data_file}'
normalized_gene_data.to_csv(gene_csv_path)

# 2. Merge the clinical and genetic data with the 'geo_merge_clinical_genetic_data' function from the library.
merged_data = geo_merge_clinical_genetic_data(selected_clinical_data, normalized_gene_data)

# 3. Determine whether the trait and some demographic attributes in the data is severely biased, and remove biased attributes.
trait_biased, unbiased_merged_data = judge_and_remove_biased_features(merged_data, '{trait}')

# If the trait is not severely biased, save the cohort information and the merged data.

# 4. Save the cohort information.
save_cohort_info('{cohort}', '{json_path}', True, True, is_trait_biased, merged_data)

if not trait_biased:
    # 5. If the trait is not severely biased, save the merged data to a csv file.
    csv_path = '{out_data_file}'
    unbiased_merged_data.to_csv(csv_path)
    """