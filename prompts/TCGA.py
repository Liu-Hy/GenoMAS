TCGA_ROLE_PROMPT: str = \
"""You are an expert data engineer specializing in biomedical data analysis. Your task is to preprocess and wrangle gene
expression data from the TCGA (The Cancer Genome Atlas) database, ensuring it's suitable for downstream analysis."""

TCGA_GUIDELINES: str = \
"""Guidelines for Preprocessing Gene Expression Data from TCGA Series:

Gene expression datasets from TCGA often require careful preprocessing to ensure reliable downstream analysis. This 
pipeline standardizes the preprocessing steps while maintaining data quality and biological relevance.

1. Data Loading
   - Find the most appropriate cancer cohort for our trait of interest
   - Since TCGA organizes data by cancer types, choosing the right cohort is crucial
   - Each cohort contains both clinical information about patients and their gene expression profiles. We need both 
     types of data to understand the relationship between patient characteristics and gene activity
   - If we can't find suitable data, it's better to skip this trait than proceed with an inappropriate dataset

2. Patient Demographics
   - Cancer progression and treatment responses often vary with age and gender
   - Look for these important demographic factors in the clinical data
   - Create a list of possible columns that might contain this information
   - Sometimes this information might be recorded under different names or formats
   - Understanding the patient population helps interpret gene expression patterns

3. Demographic Data Quality
   - Choose the most reliable feature for age and gender information
   - Patient demographics are crucial for understanding disease contexts, since different cancer 
     types may affect age groups or genders differently
   - While missing demographic data isn't ideal, it doesn't prevent analysis. Other clinical factors 
     may still provide valuable insights

4. Data Integration
   - Combine patient information with their gene expression data
   - Cancer studies require both clinical context and molecular profiles
   - Clean up the data to ensure accuracy:
     * Remove unreliable or sparse measurements
     * Handle missing information appropriately
   - Check if the patient group is representative and unbiased
   - Only save data for future analysis if the data quality meets research standards
"""


TCGA_DATA_LOADING_PROMPT: str = \
"""
1. Review the following subdirectories from the TCGA Xena dataset root directory (`tcga_root_dir`). Select the 
   subdirectory whose name contains either synonymous or highly overlapping phenotypic terms to our target trait, as 
   this subdirectory likely contains relevant trait data. If multiple options exist, choose the most specific match. 
   If no suitable directory is found, we need to skip this trait and mark the task as completed. Subdirectories are 
   listed below:
   {list_subdirs}

2. For the selected directory, identify the paths to two key files: a clinical data file, which contains 
   'clinicalMatrix' in filename, and a genetic data file, which contains 'PANCAN' in filename. 

3. Load both files as Pandas dataframes. Use index_col=0, sep='\t', and be careful with the file format.

4. Print the column names of the clinical data for further analysis.
"""

TCGA_FIND_CANDIDATE_DEMOGRAPHIC_PROMPT: str = \
"""
1. In a previous step, we obtained a list of column names from a biomedical dataset. Please examine the list and 
   identify all the columns that are likely to contain information about patients' age. Similarly, please also identify 
   all columns that may hold data on patients' gender. Please provide your answer by strictly following this format, 
   without redundant words:
   candidate_age_cols = [col_name1, col_name2, ...]
   candidate_gender_cols = [col_name1, col_name2, ...]

   If no columns match a criterion, please provide an empty list for the corresponding variable.

2. For both age and gender, if applicable, please extract the candidate columns from the clinical data, and preview the
   extracted data by displaying the column names and their first 5 values as a Python dictionary.
   
[Output of a previous step]
"""

TCGA_SELECT_DEMOGRAPHIC_PROMPT: str = \
"""
1. In a previous step, we identified candidate columns for demographic information. We then created two separate Python 
   dictionaries: one for age values and one for gender values, each storing the first 5 values from their respective 
   candidate columns. Please select a single column from the candidate columns that most accurately record age and 
   gender information, respectively. It should contain meaningful values and have no large proportion of missing values. 
   Assign the chosen column name for age information to the variable `age_col`, and for gender information to the 
   variable `gender_col`. In case the input dictionary for either age or gender is empty, or if no suitable column is 
   found for the demographic attribute after thorough inspection, set the corresponding variable to None.

2. Explicitly print out the information for the chosen `age_col` and chosen `gender_col`.

[Output of a previous step]
"""

TCGA_FEATURE_ENGINEERING_PROMPT: str = \
"""
1. Extract and standardize the clinical features to get a dataframe with the trait and optional age and gender 
   information.

2. Normalize gene symbols in the obtained gene expression data using synonym information from the NCBI Gene database. 
   Remove data of unrecognized gene symbols, and average the expression values of gene symbols that are mapped to the 
   same standard symbol. Save the normalized data as a CSV file to `out_gene_data_file`

3. Link the clinical and genetic data on sample IDs, and assign the linked data to a variable `linked_data`.

4. Handle missing values in the linked data systematically. We remove samples with missing trait values, remove genes 
   features with >20% missing values, remove samples with >5% missing genes, and then impute remaining missing values. 
   We impute gender with the mode, and impute other features with the mean.

5. Determine whether the trait and some demographic features in the dataset are severely biased. We determine whether 
   the trait is severely biased to validate the usability of the dataset later. Biased demographic features are 
   tolerable, and we simply remove them.

6. Conduct final quality validation and save relevant information about the linked cohort data using the
   `validate_and_save_cohort_info` function from the library. You may optionally take notes about anything that is 
   worthy of attention about the dataset.

7. If the linked data is usable, save it as a CSV file to `out_data_file`. Otherwise, you must not save it.
"""