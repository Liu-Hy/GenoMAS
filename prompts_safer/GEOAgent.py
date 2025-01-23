ROLE_PROMPT: str = \
"""You are an expert data engineer specializing in biomedical data analysis. Your task is to preprocess and wrangle gene
expression data from the GEO (Gene Expression Omnibus) database, ensuring it's suitable for downstream analysis."""

GUIDELINES: str = \
"""Guidelines for Preprocessing Gene Expression Data from GEO Series:

Gene expression datasets from GEO often require careful preprocessing to ensure reliable downstream analysis. This 
pipeline standardizes the preprocessing steps while maintaining data quality and biological relevance.

1. Initial Data Acquisition and Organization
   GEO series typically contain two key files: a SOFT file with detailed annotations and a matrix file with expression 
   values.
   - Identify and locate both files in the dataset
   - Extract essential metadata including series description and clinical annotations
   - Observe sample characteristics to understand the dataset's demographic and clinical composition

2. Clinical Feature Assessment
   Human studies require careful consideration of both the trait of interest and potential confounding factors.
   - Examine if the dataset contains gene expression measurements (not solely miRNA or methylation data)
   - Assess availability of the target trait in clinical annotations
   - Identify age and gender information, which are important covariates that often confound gene expression
   - Convert clinical variables to appropriate data types:
     * Binary traits should be coded as 0/1 with consistent rules (e.g., control=0, case=1)
     * Continuous traits should be converted to numerical values
   - Extract and standardize clinical features when trait data is present

3. Gene Expression Matrix Processing
   Microarray and RNA-seq data often come with different types of gene identifiers, requiring careful handling.
   - Extract the gene expression matrix while preserving sample identifiers
   - Observe the format of gene identifiers (e.g., gene symbols, probe IDs, RefSeq)

4. Gene Identifier Review
   Modern analyses require standardized gene symbols, but many datasets use platform-specific identifiers.
   - Analyze whether the expression data uses standardized human gene symbols
   - If non-standard identifiers are used, which means gene mapping is needed, then proceed with gene annotation and 
     mapping steps; otherwise, jump directly to data integration

5. Gene Annotation Extraction
   When mapping is needed, we extract probe-gene relationships from the platform annotation.
   - Extract the mapping information from the SOFT file
   - Identify the appropriate columns containing probe IDs and corresponding gene symbols
   - Observe the annotation data to verify its completeness and quality.

6. Gene Symbol Mapping
   The relationship between probes and genes is often many-to-many, requiring careful handling:
   - For one probe mapping to multiple genes:
     * Split the probe's expression value equally among all target genes
     * This maintains the total expression signal while avoiding bias
   - For multiple probes mapping to one gene:
     * Sum the contributions from all probes
     * This captures the total expression while accounting for split values
   - Example: If probe P1 maps to genes G1 and G2, and probe P2 maps to G2:
     * G1 receives 0.5 × P1
     * G2 receives (0.5 × P1) + P2

7. Data Integration and Quality Control
   The final step ensures data quality while maximizing usable samples and features.
   - Normalize gene symbols to ensure consistency across the dataset
   - Integrate clinical and genetic data, ensuring proper sample alignment
   - Apply systematic missing value handling:
     * Remove samples lacking trait information as they cannot contribute to analysis
     * Remove genes with excessive missing values (>20%) to maintain data reliability
     * Filter out samples with too many missing gene measurements (>5%)
     * Carefully impute remaining missing values: use mode imputation for gender and mean for other features.
   - Evaluate potential biases in trait and demographic features
   - Proceed with saving the processed dataset only if it passes quality checks
"""

TOOLS: str = \
"""Tools:
"tools.preprocess" provides well-developed helper functions for this project. Henceforth, it will be referred to as 
"the library". Please import and use functions from the library when possible. But if none of these function satisfy
your needs, feel free to adapt the implementation or write your own. Below is the source code:
{tools_code}
"""

SETUPS: str = \
"""
Programming Environment Setup:
{path_setup}

NOTE: The overall preprocessing requires multiple steps, where each step often depends on the execution results of 
previous steps. In each step, you perform an Action Unit by writing a code snippet following instructions, and then the 
execution result will be given to you for either revision of the current step or progression to the next step.

Based on the context, write code to follow the instructions.
"""

DATA_LOADING_PROMPT: str = \
"""
1. Identify the paths to the SOFT file and the matrix file, and assign them to the variables 'soft_file' and 
   'matrix_file'.
2. Read the matrix file to obtain dataset background information and the clinical dataframe 'clinical_data' through the 
   'get_background_and_clinical_data' function from the library. Extract relevant text lines by prefix matching. For 
   background information, use prefixes '!Series_title', '!Series_summary', and '!Series_overall_design'; for clinical 
   data, use prefixes '!Sample_geo_accession' and '!Sample_characteristics_ch1'.
3. Obtain the sample characteristics dictionary from 'clinical_data' via the 'get_unique_values_by_row' function from 
   the library
4. Explicitly print out all the background information and the sample characteristics dictionary.
"""

CODE_STEP1: str = \
"""# STEP1
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
"""

FEATURE_ANALYSIS_EXTRACTION_PROMPT: str = \
"""
As a biomedical research team, we are analyzing datasets to study the association between the human trait and genetic 
factors, considering the possible influence of age and gender. (The trait is defined above in Setups and assigned to the
Python variable 'trait'.) After searching the GEO database and parsing the matrix file of a series, we have obtained
background information and sample characteristics data. Please review the output from the previous step and answer the 
following questions regarding this dataset:

1. Gene Expression Data Availability
   - Is this dataset likely to contain gene expression data? (Note: Pure miRNA data or methylation data are not 
     suitable.)
     - If YES, set `is_gene_available` to `True`. Otherwise set it to `False`.

2. Variable Availability and Data Type Conversion
   For each of the variables '{trait}', 'age', and 'gender', address the following points:

   **2.1 Data Availability**
     - If human data for this variable is available, identify the key in the sample characteristics dictionary where 
       unique values of this variable are recorded. The key is an integer. The variable data might be explicitly 
       recorded or can be inferred from the field with biomedical knowledge or understanding of the dataset background. 
       If you can't find such a key, this means the data is not available. 
     - Constant features are useless in our associative studies. Therefore, if there is only one unique value under the
       key in the sample characteristics dictionary, or if the background information suggests that "everyone has the 
       same value for some variable", consider it as not available.
     - Name the keys `trait_row`, `age_row`, and `gender_row`, respectively. Use None to indicate the corresponding data
       is not available.

   **2.2 Data Type Conversion**
     - Choose an appropriate data type for each variable ('continuous' or 'binary').
     - If the data type is binary, convert values to 0 and 1. For gender data, convert female to 0 and male to 1.
     - Write a Python function to convert any given value of the variable to this data type. Typically, a colon (':') 
       separates the header and the value in each cell, so ensure to extract the value after the colon in the function. 
       Unknown values should be converted to None.
     - When data is not explicitly given but can be inferred, carefully observe the unique values in the sample 
       characteristics dictionary, think about what they mean in this dataset, and design a heuristic rule to convert 
       those values into the chosen type. If you are 90% sure that some cases should be mapped to a known value, please 
       do that instead of giving `None`. 
     - Name the functions `convert_trait`, `convert_age`, and `convert_gender`, respectively.

3. Save Metadata
   Please conduct initial filtering on the usability of the dataset and save relevant information using the 
   `validate_and_save_cohort_info` function from the library. Trait data availability can be determined by whether 
   `trait_row` is None.

4. Clinical Feature Extraction
   If trait_row is not None, it means clinical data is available, then you MUST DO this substep. Otherwise, you should 
   skip this substep.
   Use the `geo_select_clinical_features` function to obtain the output `selected_clinical_data` from the input 
   dataframe `clinical_data` that we previously obtained, observe it with the `preview_df` function, and save it  
   as a CSV file to `out_clinical_data_file`. 

[Output of the previous step]
"""

GENE_DATA_EXTRACTION_PROMPT: str = \
"""
1. Use the get_genetic_data function from the library to get the gene_data from the matrix_file previously defined.
2. print the first 20 row ids for future observation.
"""

CODE_STEP3: str = \
"""# STEP3
# 1. Use the get_genetic_data function from the library to get the gene_data from the matrix_file previously defined.
gene_data = get_genetic_data(matrix_file)

# 2. Print the first 20 row IDs (gene identifiers) for future observation.
print(gene_data.index[:20])
"""

GENE_IDENTIFIER_REVIEW_PROMPT: str = \
"""
Observe the gene identifiers in the gene expression data given in the previous step. Based on your biomedical knowledge,
are they human gene symbols, or are they some other identifiers that need to be mapped to gene symbols? Your answer 
should be concluded by starting a new line and strictly following this format:
requires_gene_mapping = (True or False)

[Output of the previous step]
"""

GENE_ANNOTATION_PROMPT: str = \
"""
1. Use the 'get_gene_annotation' function from the library to get gene annotation data from the SOFT file
2. Use the 'preview_df' function from the library to preview the data and print out the results for future 
   observation. Mark the printing with a string like "Gene annotation".
"""

CODE_STEP5: str = \
"""# STEP5
# 1. Use the 'get_gene_annotation' function from the library to get gene annotation data from the SOFT file.
gene_annotation = get_gene_annotation(soft_file)

# 2. Use the 'preview_df' function from the library to preview the data and print out the results.
print("Gene annotation preview:")
print(preview_df(gene_annotation))
"""

GENE_IDENTIFIER_MAPPING_PROMPT: str = \
"""
1. When analyzing a gene expression dataset, we need to map some identifiers of genes to actual gene symbols. Please 
   observe the gene identifiers in the gene expression data given in a previous step, and also the dictionary preview 
   of the gene annotation data given in another step. 
   Please decide which key in the dictionary stores the same kind of gene identifiers as in the gene expression data, 
   and which key stores the gene symbols. Please strictly follow this format in your answer:
   identifier_key = 'key_name1'
   gene_symbol_key = 'key_name2'
2. Get the dataframe storing the mapping between probe IDs and genes using the 'get_gene_mapping' function from the 
   library. 
3. Apply the mapping with the 'apply_gene_mapping' function from the library, and name the resulting gene expression
   dataframe as 'gene_data'. We handle the many-to-many relation between probes and genes in this way: for each probe 
   that maps to multiple genes, we divide its expression value equally among those genes, then sum up all probe values 
   for each gene.
"""

DATA_NORMALIZATION_LINKING_PROMPT: str = \
"""
1. Normalize the obtained gene data with the 'normalize_gene_symbols_in_index' function from the library. Save the 
   normalized genetic data as a CSV file to 'out_gene_data_file'
2. Link the clinical and genetic data with the 'geo_link_clinical_genetic_data' function from the library, and assign 
   the linked data to a variable 'linked_data'.
3. Handle missing values in the linked data with the 'handle_missing_values' function from the library. We remove 
   samples with missing trait values, remove genes features with >20% missing values, remove samples with >5% missing 
   genes, and then impute remaining missing values. We impute gender with the mode, and impute other features with the 
   mean.
4. Determine whether the trait and some demographic features in the data are severely biased, and remove biased 
   features with the 'judge_and_remove_biased_features' function from the library.
5. Conduct final quality validation and save relevant information about the linked cohort data with the 
   'validate_and_save_cohort_info' function from the library. You may optionally take notes about anything that
   is worthy of attention about the dataset.
6. If the linked data is usable, save it as a CSV file to 'out_data_file'. Otherwise, you must not save it.
"""

CODE_STEP7: str = \
"""# STEP7
# 1. Normalize the obtained gene data with the 'normalize_gene_symbols_in_index' function from the library.
normalized_gene_data = normalize_gene_symbols_in_index(gene_data)
normalized_gene_data.to_csv(out_gene_data_file)

# 2. Link the clinical and genetic data with the 'geo_link_clinical_genetic_data' function from the library.
linked_data = geo_link_clinical_genetic_data(selected_clinical_data, normalized_gene_data)

# 3. Handle missing values in the linked data
linked_data = handle_missing_values(linked_data, trait)

# 4. Determine whether the trait and some demographic features are severely biased, and remove biased features.
is_trait_biased, unbiased_linked_data = judge_and_remove_biased_features(linked_data, trait)

# 5. Conduct quality check and save the cohort information.
is_usable = validate_and_save_cohort_info(True, cohort, json_path, True, True, is_trait_biased, linked_data)

# 6. If the linked data is usable, save it as a CSV file to 'out_data_file'.
if is_usable:
    unbiased_linked_data.to_csv(out_data_file)
"""

TASK_COMPLETED_PROMPT: str = \
"""
Mark the task as completed when you have finished processing a dataset or you find you should stop early, either because 
there is no input data available, or that the gene expression or clinical data is missing from the dataset.
"""