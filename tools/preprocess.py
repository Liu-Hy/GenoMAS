import gzip
import io
import json
import os
import re
from typing import Callable, Optional, List, Tuple, Dict, Union, Any

import mygene
import pandas as pd


def geo_get_relevant_filepaths(cohort_dir: str) -> Tuple[str, str]:
    """Find the file paths of a SOFT file and a matrix file from the given data directory of a cohort.
    If there are multiple SOFT files or matrix files, simply choose the first one. Used for the GEO dataset.
    """
    files = os.listdir(cohort_dir)
    soft_files = [f for f in files if 'soft' in f.lower()]
    matrix_files = [f for f in files if 'matrix' in f.lower()]
    assert len(soft_files) > 0 and len(matrix_files) > 0
    soft_file_path = os.path.join(cohort_dir, soft_files[0])
    matrix_file_path = os.path.join(cohort_dir, matrix_files[0])

    return soft_file_path, matrix_file_path


def tcga_get_relevant_filepaths(cohort_dir: str) -> Tuple[str, str]:
    """Find the file paths of a clinical file and a genetic file from the given data directory of a cohort.
    If there are multiple clinical or genetic data files, simply choose the first one. Used for the TCGA Xena dataset.
    """
    files = os.listdir(cohort_dir)
    clinical_files = [f for f in files if 'clinicalmatrix' in f.lower()]
    genetic_files = [f for f in files if 'pancan' in f.lower()]
    clinical_file_path = os.path.join(cohort_dir, clinical_files[0])
    genetic_file_path = os.path.join(cohort_dir, genetic_files[0])
    return clinical_file_path, genetic_file_path


def line_generator(source: str, source_type: str) -> str:
    """Generator that yields lines from a file or a string.

    Parameters:
    - source: File path or string content.
    - source_type: 'file' or 'string'.
    """
    if source_type == 'file':
        with gzip.open(source, 'rt') as f:
            for line in f:
                yield line.strip()
    elif source_type == 'string':
        for line in source.split('\n'):
            yield line.strip()
    else:
        raise ValueError("source_type must be 'file' or 'string'")


def filter_content_by_prefix(
        source: str,
        prefixes_a: List[str],
        prefixes_b: Optional[List[str]] = None,
        unselect: bool = False,
        source_type: str = 'file',
        return_df_a: bool = True,
        return_df_b: bool = True
) -> Tuple[Union[str, pd.DataFrame], Optional[Union[str, pd.DataFrame]]]:
    """
    Filters rows from a file or a list of strings based on specified prefixes.

    Parameters:
    - source (str): File path or string content to filter.
    - prefixes_a (List[str]): Primary list of prefixes to filter by.
    - prefixes_b (Optional[List[str]]): Optional secondary list of prefixes to filter by.
    - unselect (bool): If True, selects rows that do not start with the specified prefixes.
    - source_type (str): 'file' if source is a file path, 'string' if source is a string of text.
    - return_df_a (bool): If True, returns filtered content for prefixes_a as a pandas DataFrame.
    - return_df_b (bool): If True, and if prefixes_b is provided, returns filtered content for prefixes_b as a pandas DataFrame.

    Returns:
    - Tuple: A tuple where the first element is the filtered content for prefixes_a, and the second element is the filtered content for prefixes_b.
    """
    filtered_lines_a = []
    filtered_lines_b = []
    prefix_set_a = set(prefixes_a)
    if prefixes_b is not None:
        prefix_set_b = set(prefixes_b)

    # Use generator to get lines
    for line in line_generator(source, source_type):
        matched_a = any(line.startswith(prefix) for prefix in prefix_set_a)
        if matched_a != unselect:
            filtered_lines_a.append(line)
        if prefixes_b is not None:
            matched_b = any(line.startswith(prefix) for prefix in prefix_set_b)
            if matched_b != unselect:
                filtered_lines_b.append(line)

    filtered_content_a = '\n'.join(filtered_lines_a)
    if return_df_a:
        filtered_content_a = pd.read_csv(io.StringIO(filtered_content_a), delimiter='\t', low_memory=False,
                                         on_bad_lines='skip')
    filtered_content_b = None
    if filtered_lines_b:
        filtered_content_b = '\n'.join(filtered_lines_b)
        if return_df_b:
            filtered_content_b = pd.read_csv(io.StringIO(filtered_content_b), delimiter='\t', low_memory=False,
                                             on_bad_lines='skip')

    return filtered_content_a, filtered_content_b


def get_background_and_clinical_data(file_path: str,
                                     prefixes_a: List[str] = ['!Series_title', '!Series_summary',
                                                              '!Series_overall_design'],
                                     prefixes_b: List[str] = ['!Sample_geo_accession', '!Sample_characteristics_ch1']
                                     ) -> Tuple[str, pd.DataFrame]:
    """Extract background information and clinical data from a matrix file."""
    background_info, clinical_data = filter_content_by_prefix(file_path, prefixes_a, prefixes_b, unselect=False,
                                                              source_type='file',
                                                              return_df_a=False, return_df_b=True)
    return background_info, clinical_data


def get_gene_annotation(file_path: str, prefixes: List[str] = ['^', '!', '#']) -> pd.DataFrame:
    """Extract gene annotation data from a SOFT file."""
    gene_metadata = filter_content_by_prefix(file_path, prefixes_a=prefixes, unselect=True, source_type='file',
                                             return_df_a=True)
    return gene_metadata[0]


def get_gene_mapping(annotation: pd.DataFrame, prob_col: str, gene_col: str) -> pd.DataFrame:
    """Process gene annotation to get mapping between gene names and probes."""
    mapping_data = annotation.loc[:, [prob_col, gene_col]]
    mapping_data = mapping_data.dropna()
    mapping_data = mapping_data.rename(columns={gene_col: 'Gene'}).astype({'ID': 'str'})

    return mapping_data


def get_genetic_data(file_path: str, marker: str = "!series_matrix_table_begin") -> pd.DataFrame:
    """Read the gene expression data into a dataframe, and adjust its format"""
    # Determine the number of rows to skip
    with gzip.open(file_path, 'rt') as file:
        for i, line in enumerate(file):
            if marker in line:
                skip_rows = i + 1  # +1 to skip the marker row itself
                break
        else:
            raise ValueError(f"Marker '{marker}' not found in the file.")

    # Read the genetic data into a dataframe
    genetic_data = pd.read_csv(file_path, compression='gzip', skiprows=skip_rows, comment='!', delimiter='\t',
                               on_bad_lines='skip')
    genetic_data = genetic_data.rename(columns={'ID_REF': 'ID'}).astype({'ID': 'str'})
    genetic_data.set_index('ID', inplace=True)

    return genetic_data


def extract_human_gene_symbols(text: str) -> List[str]:
    """
    Extract a list of likely human gene symbols from (often messy) GEO annotation text.
    Includes both canonical symbols and the C#orf# pattern (e.g., C10orf54).
    Excludes transcripts/predictions starting with NR_, XR_, LOC\d+, LINC\d+.
    Filters out a few trivial lab terms (DNA, RNA, PCR, EST, CHR).
    """

    # Explanation:
    # (?!NR_|XR_|LOC\d+|LINC\d+)     Negative lookahead to exclude undesired prefixes.
    # (?: ...|... )                 Group of alternatives:
    #   [A-Z][A-Z0-9-]{1,9}         The basic pattern: start uppercase + up to 9 more uppercase letters, digits or dashes.
    #   C\d+orf\d+                  The special orf pattern (case-insensitive for "orf"? 
    #                               Usually official symbol is 'orf' in lowercase, so let's keep it as is).
    #
    # \b Word boundaries to ensure we capture whole tokens.
    pattern = (
        r"\b"
        r"(?!NR_|XR_|LOC\d+|LINC\d+)"
        r"(?:[A-Z][A-Z0-9-]{1,9}|C\d+orf\d+)"
        r"\b"
    )

    if not isinstance(text, str):
        return []

    candidates = re.findall(pattern, text)

    # Exclude trivial lab terms
    exclude_simple = {"DNA", "RNA", "PCR", "EST", "CHR"}
    filtered = [c for c in candidates if c not in exclude_simple]

    # Deduplicate in the order found
    return list(dict.fromkeys(filtered))


def apply_gene_mapping(expression_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert measured data about gene probes into gene expression data.
    If a probe maps to n genes, each gene gets expression / n. Then we sum across all probes for each gene.

    Parameters:
    expression_df (DataFrame): A DataFrame with gene expression data, indexed by 'ID'.
    mapping_df (DataFrame): A DataFrame mapping 'ID' to 'Gene', with 'ID' as a column.

    Returns:
    DataFrame: A DataFrame with summed gene expression values, indexed by 'Gene'.
    """
    mapping_df = mapping_df[mapping_df['ID'].isin(expression_df.index)].copy()
    mapping_df['Gene'] = mapping_df['Gene'].apply(extract_human_gene_symbols)

    # Count genes per probe and expand to one gene per row
    mapping_df['num_genes'] = mapping_df['Gene'].apply(len)
    mapping_df = mapping_df.explode('Gene')
    # Empty list becomes NaN after explode, which should be dropped
    mapping_df = mapping_df.dropna(subset=['Gene'])
    mapping_df.set_index('ID', inplace=True)

    # Merge and distribute expression values
    merged_df = mapping_df.join(expression_df)
    expr_cols = [col for col in merged_df.columns if col not in ['Gene', 'num_genes']]
    merged_df[expr_cols] = merged_df[expr_cols].div(merged_df['num_genes'].replace(0, 1), axis=0)

    # Sum expression values for each gene
    gene_expression_df = merged_df.groupby('Gene')[expr_cols].sum()

    return gene_expression_df


def normalize_gene_symbols_in_index(gene_df: pd.DataFrame) -> pd.DataFrame:
    """Use gene synonym information extracted from the NCBI Gene database to normalize gene symbols in dataframe index,
    and aggregate rows with same normalized symbol.
    """
    with open("./metadata/gene_synonym.json", "r") as f:
        synonym_dict = json.load(f)
    gene_df.index = gene_df.index.str.upper().map(lambda x: synonym_dict.get(x))
    gene_df = gene_df[gene_df.index.notnull()]
    gene_df = gene_df.groupby(gene_df.index).mean()

    return gene_df


def get_feature_data(clinical_df: pd.DataFrame, row_id: int, feature: str, convert_fn: Callable) -> pd.DataFrame:
    """Extract and convert a feature row from clinical data."""
    # Get the row as a DataFrame using iloc[row_id:row_id + 1]
    df = clinical_df.iloc[row_id:row_id + 1].drop(columns=['!Sample_geo_accession'], errors='ignore')
    # Set the index name to the feature
    df.index = [feature]
    # Apply conversion function to all elements
    df = df.map(convert_fn).astype(float)
    return df


def judge_binary_variable_biased(dataframe: pd.DataFrame, col_name: str, min_proportion: float = 0.1,
                                 min_num: int = 5) -> bool:
    """Check if a binary variable's distribution is too biased for analysis."""
    label_counter = dataframe[col_name].value_counts()
    total_samples = len(dataframe)
    rare_label_num = label_counter.min()
    rare_label = label_counter.idxmin()
    rare_label_proportion = rare_label_num / total_samples

    print(
        f"For the feature \'{col_name}\', the least common label is '{rare_label}' with {rare_label_num} occurrences. This represents {rare_label_proportion:.2%} of the dataset.")

    biased = (len(label_counter) < 2) or ((rare_label_proportion < min_proportion) and (rare_label_num < min_num))
    return bool(biased)


def judge_continuous_variable_biased(dataframe: pd.DataFrame, col_name: str) -> bool:
    """Check if the distribution of a continuous variable in the dataset is too biased to be usable for analysis.
    As a starting point, we consider it biased if all values are the same. For the next step, maybe ask GPT to judge
    based on quartile statistics combined with its common sense knowledge about this feature.
    """
    quartiles = dataframe[col_name].quantile([0.25, 0.5, 0.75])
    min_value = dataframe[col_name].min()
    max_value = dataframe[col_name].max()

    # Printing quartile information
    print(f"Quartiles for '{col_name}':")
    print(f"  25%: {quartiles[0.25]}")
    print(f"  50% (Median): {quartiles[0.5]}")
    print(f"  75%: {quartiles[0.75]}")
    print(f"Min: {min_value}")
    print(f"Max: {max_value}")

    biased = min_value == max_value

    return bool(biased)


def tcga_convert_trait(row_index: str) -> int:
    """Convert TCGA sample IDs to binary trait labels based on last two digits.
    Tumor types range from 01 - 09, normal types from 10 - 19.
    """
    last_two_digits = int(row_index[-2:])

    if 1 <= last_two_digits <= 9:
        return 1
    elif 10 <= last_two_digits <= 19:
        return 0
    else:
        return -1


def tcga_convert_gender(cell: str) -> Optional[int]:
    """Convert gender strings to binary values (0: female, 1: male)."""
    if isinstance(cell, str):
        cell = cell.lower()

    if cell == "female":
        return 0
    elif cell == "male":
        return 1
    else:
        return None


def tcga_convert_age(cell: str) -> Optional[int]:
    """Extract age value from string using regex."""
    match = re.search(r'\d+', str(cell))
    if match:
        return int(match.group())
    else:
        return None


def get_unique_values_by_row(dataframe: pd.DataFrame, max_len: int = 30) -> Dict[str, List[Any]]:
    """Organize the unique values in each row of the given dataframe, to get a dictionary."""
    if '!Sample_geo_accession' in dataframe.columns:
        dataframe = dataframe.drop(columns=['!Sample_geo_accession'])
    unique_values_dict = {}
    for index, row in dataframe.iterrows():
        unique_values = list(row.unique())[:max_len]
        unique_values_dict[index] = unique_values
    return unique_values_dict


def tcga_select_clinical_features(clinical_df: pd.DataFrame, trait: str,
                                  age_col: Optional[str] = None,
                                  gender_col: Optional[str] = None) -> pd.DataFrame:
    """Select and process clinical features from TCGA Xena data."""
    feature_list = []
    trait_data = clinical_df.index.to_series().apply(tcga_convert_trait).rename(trait)
    feature_list.append(trait_data)
    if age_col:
        age_data = clinical_df[age_col].apply(tcga_convert_age).rename("Age")
        feature_list.append(age_data)
    if gender_col:
        gender_data = clinical_df[gender_col].apply(tcga_convert_gender).rename("Gender")
        feature_list.append(gender_data)
    selected_clinical_df = pd.concat(feature_list, axis=1)
    return selected_clinical_df


def geo_select_clinical_features(clinical_df: pd.DataFrame, trait: str, trait_row: int,
                                 convert_trait: Callable,
                                 age_row: Optional[int] = None,
                                 convert_age: Optional[Callable] = None,
                                 gender_row: Optional[int] = None,
                                 convert_gender: Optional[Callable] = None) -> pd.DataFrame:
    """
    Extracts and processes specific clinical features from a DataFrame representing
    sample characteristics in the GEO database series.

    Parameters:
    - clinical_df (pd.DataFrame): DataFrame containing clinical data.
    - trait (str): The trait of interest.
    - trait_row (int): Row identifier for the trait in the DataFrame.
    - convert_trait (Callable): Function to convert trait data into a desired format.
    - age_row (int, optional): Row identifier for age data. Default is None.
    - convert_age (Callable, optional): Function to convert age data. Default is None.
    - gender_row (int, optional): Row identifier for gender data. Default is None.
    - convert_gender (Callable, optional): Function to convert gender data. Default is None.

    Returns:
    pd.DataFrame: A DataFrame containing the selected and processed clinical features.
    """
    feature_list = []

    trait_data = get_feature_data(clinical_df, trait_row, trait, convert_trait)
    feature_list.append(trait_data)
    if age_row is not None:
        age_data = get_feature_data(clinical_df, age_row, 'Age', convert_age)
        feature_list.append(age_data)
    if gender_row is not None:
        gender_data = get_feature_data(clinical_df, gender_row, 'Gender', convert_gender)
        feature_list.append(gender_data)

    selected_clinical_df = pd.concat(feature_list, axis=0)
    return selected_clinical_df


def geo_link_clinical_genetic_data(clinical_df: pd.DataFrame, genetic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Link clinical and genetic data to get a dataframe for associational studies.
    """
    # Ensure proper column naming
    if 'ID' in genetic_df.columns:
        genetic_df = genetic_df.rename(columns={'ID': 'Gene'})
    if 'Gene' in genetic_df.columns:
        genetic_df = genetic_df.set_index('Gene')

    linked_data = pd.concat([clinical_df, genetic_df], axis=0).T

    return linked_data


def handle_missing_values(df: pd.DataFrame, trait_col: str) -> pd.DataFrame:
    """
    Handle missing values in gene expression dataset following best practices:
    1. Remove samples with missing trait values
    2. Remove genes with >20% missing values
    3. Remove samples with >5% missing genes
    4. Impute remaining missing values for genes and covariates
    
    Parameters:
    df : pd.DataFrame
        DataFrame containing trait, covariates, and gene expression data
    trait_col : str
        Name of the trait column

    Returns:
    --------
    pd.DataFrame
        Processed dataframe with missing values handled
    """

    # Identify gene columns
    covariate_cols = [trait_col, 'Age', 'Gender']
    gene_cols = [col for col in df.columns if col not in covariate_cols]

    # 1. Drop samples with missing trait
    df = df.dropna(subset=[trait_col])

    # 2. Filter genes with >20% missing values
    gene_missing_pct = df[gene_cols].isna().mean()
    genes_to_keep = gene_missing_pct[gene_missing_pct <= 0.2].index
    df = df[[col for col in df.columns if col in genes_to_keep or col in covariate_cols]]

    # 3. Filter samples with >5% missing genes
    gene_cols = [col for col in df.columns if col not in covariate_cols]
    sample_missing_pct = df[gene_cols].isna().mean(axis=1)
    samples_to_keep = sample_missing_pct[sample_missing_pct <= 0.05].index
    df = df.loc[samples_to_keep]

    # 4. Impute remaining missing values
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].mean())

    if 'Gender' in df.columns:
        mode_result = df['Gender'].mode()
        if len(mode_result) > 0:
            df['Gender'] = df['Gender'].fillna(mode_result[0])
        else:
            # If no mode exists (all NaN), drop the Gender column as it's not useful
            df = df.drop('Gender', axis=1)

    df[gene_cols] = df[gene_cols].fillna(df[gene_cols].mean())

    return df


def judge_and_remove_biased_features(df: pd.DataFrame, trait: str) -> Tuple[bool, pd.DataFrame]:
    """Evaluate and remove biased features from the dataset.
    Checks if trait, age, and gender distributions are biased. Removes age and gender features if they are biased.
    """
    trait_type = 'binary' if len(df[trait].unique()) == 2 else 'continuous'
    if trait_type == "binary":
        trait_biased = judge_binary_variable_biased(df, trait)
    else:
        trait_biased = judge_continuous_variable_biased(df, trait)
    if trait_biased:
        print(f"The distribution of the feature \'{trait}\' in this dataset is severely biased.\n")
    else:
        print(f"The distribution of the feature \'{trait}\' in this dataset is fine.\n")
    if "Age" in df.columns:
        age_biased = judge_continuous_variable_biased(df, 'Age')
        if age_biased:
            print(f"The distribution of the feature \'Age\' in this dataset is severely biased.\n")
            df = df.drop(columns='Age')
        else:
            print(f"The distribution of the feature \'Age\' in this dataset is fine.\n")
    if "Gender" in df.columns:
        gender_biased = judge_binary_variable_biased(df, 'Gender')
        if gender_biased:
            print(f"The distribution of the feature \'Gender\' in this dataset is severely biased.\n")
            df = df.drop(columns='Gender')
        else:
            print(f"The distribution of the feature \'Gender\' in this dataset is fine.\n")

    return trait_biased, df


def validate_and_save_cohort_info(is_final: bool, cohort: str, info_path: str, is_gene_available: bool,
                                  is_trait_available: bool,
                                  is_biased: Optional[bool] = None, df: Optional[pd.DataFrame] = None,
                                  note: str = '') -> bool:
    """
    Validate and save information about the usability and quality of a dataset for statistical analysis.

    Parameters:
    is_final (bool): If True, performs final validation with full data quality checks and records metadata.
                     If False, performs initial filtering on gene and trait data availability, and only records metadata for failed datasets.
    cohort (str): A unique identifier for the dataset.
    info_path (str): File path to the JSON file where records are stored.
    is_gene_available (bool): Indicates whether the dataset contains genetic data
    is_trait_available (bool): Indicates whether the dataset contains trait data

    The below parameters are only used when 'is_final' is True:
    is_biased (bool, optional): Indicates whether the dataset is too biased to be usable.
    df (pandas.DataFrame, optional): The preprocessed dataset.
    note (str, optional): Additional notes about the dataset.

    Returns:
    bool: True if the dataset was completely preprocessed and saved, ready for future statistical analysis. 
    """
    is_usable = False
    if not is_final:
        if is_gene_available and is_trait_available:
            # The dataset passes initial filtering. Return to continue preprocessing.
            return is_usable
        else:
            # The dataset will be filtered out. Record metadata.
            new_record = {"is_usable": False,
                          "is_gene_available": is_gene_available,
                          "is_trait_available": is_trait_available,
                          "is_available": False,
                          "is_biased": None,
                          "has_age": None,
                          "has_gender": None,
                          "sample_size": None,
                          "note": None}
    else:
        # Perform final validation
        if (df is None) or (is_biased is None):
            raise ValueError("For final data validation, 'df' and 'is_biased' must be provided.")
        # Detect abnormality in data and override input indicators
        if len(df) <= 0 or len(df.columns) <= 4:
            print(f"Abnormality detected in the cohort: {cohort}. Preprocessing failed.")
            is_gene_available = False
            if len(df) <= 0:
                is_trait_available = False
        is_available = is_gene_available and is_trait_available

        is_usable = is_available and (is_biased is False)
        new_record = {"is_usable": is_usable,
                      "is_gene_available": is_gene_available,
                      "is_trait_available": is_trait_available,
                      "is_available": is_available,
                      "is_biased": is_biased if is_available else None,
                      "has_age": "Age" in df.columns if is_available else None,
                      "has_gender": "Gender" in df.columns if is_available else None,
                      "sample_size": len(df) if is_available else None,
                      "note": note}

    trait_directory = os.path.dirname(info_path)
    os.makedirs(trait_directory, exist_ok=True)
    if not os.path.exists(info_path):
        with open(info_path, 'w') as file:
            json.dump({}, file)
        print(f"A new JSON file was created at: {info_path}")

    with open(info_path, "r") as file:
        records = json.load(file)
    records[cohort] = new_record

    temp_path = info_path + ".tmp"
    try:
        with open(temp_path, 'w') as file:
            json.dump(records, file)
        os.replace(temp_path, info_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return is_usable


def preview_df(df: pd.DataFrame, n: int = 5, max_items: int = 200) -> Dict[str, Any]:
    """Preview DataFrame contents with limited number of items."""
    # Get the dictionary of the first n rows
    data_dict = df.head(n).to_dict(orient='list')

    # If the dictionary has more than max_items, truncate it
    if len(data_dict) > max_items:
        truncated_dict = {k: data_dict[k] for k in list(data_dict.keys())[:max_items]}
        return truncated_dict
    else:
        return data_dict
