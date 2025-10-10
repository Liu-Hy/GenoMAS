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
