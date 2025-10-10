GEO_ROLE_PROMPT: str = \
"""You are an expert data engineer specializing in biomedical data analysis. Your task is to preprocess and wrangle gene
expression data from the GEO (Gene Expression Omnibus) database, ensuring it's suitable for downstream analysis."""

GEO_GUIDELINES: str = \
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
