import os
import io
import pandas as pd
import gzip
import mygene
import re
import json
from typing import Callable, Optional, List, Tuple, Dict, Union, Any

def geo_get_relevant_filepaths(cohort_dir):
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


def xena_get_relevant_filepaths(cohort_dir):
    """Find the file paths of a clinical file and a genetic file from the given data directory of a cohort.
    If there are multiple clinical or genetic data files, simply choose the first one. Used for the TCGA Xena dataset.
    """
    files = os.listdir(cohort_dir)
    clinical_files = [f for f in files if 'clinicalmatrix' in f.lower()]
    genetic_files = [f for f in files if 'pancan' in f.lower()]
    clinical_file_path = os.path.join(cohort_dir, clinical_files[0])
    genetic_file_path = os.path.join(cohort_dir, genetic_files[0])
    return clinical_file_path, genetic_file_path

def line_generator(source, source_type):
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
        filtered_content_a = pd.read_csv(io.StringIO(filtered_content_a), delimiter='\t', low_memory=False, on_bad_lines='skip')
    filtered_content_b = None
    if filtered_lines_b:
        filtered_content_b = '\n'.join(filtered_lines_b)
        if return_df_b:
            filtered_content_b = pd.read_csv(io.StringIO(filtered_content_b), delimiter='\t', low_memory=False, on_bad_lines='skip')

    return filtered_content_a, filtered_content_b


def get_background_and_clinical_data(file_path,
                                     prefixes_a=['!Series_title', '!Series_summary', '!Series_overall_design'],
                                     prefixes_b=['!Sample_geo_accession', '!Sample_characteristics_ch1']):
    """Extract from a matrix file the background information about the dataset, and sample characteristics data"""
    background_info, clinical_data = filter_content_by_prefix(file_path, prefixes_a, prefixes_b, unselect=False,
                                                              source_type='file',
                                                              return_df_a=False, return_df_b=True)
    return background_info, clinical_data


def get_gene_annotation(file_path, prefixes=['^', '!', '#']):
    """Extract from a SOFT file the gene annotation data"""
    gene_metadata = filter_content_by_prefix(file_path, prefixes_a=prefixes, unselect=True, source_type='file',
                                             return_df_a=True)
    return gene_metadata[0]


def get_gene_mapping(annotation, prob_col, gene_col):
    """Process the gene annotation to get the mapping between gene names and gene probes.
    """
    mapping_data = annotation.loc[:, [prob_col, gene_col]]
    mapping_data = mapping_data.dropna()
    mapping_data = mapping_data.rename(columns={gene_col: 'Gene'}).astype({'ID': 'str'})

    return mapping_data


def get_genetic_data(file_path):
    """Read the gene expression data into a dataframe, and adjust its format"""
    genetic_data = pd.read_csv(file_path, compression='gzip', skiprows=52, comment='!', delimiter='\t')
    genetic_data = genetic_data.dropna()
    genetic_data = genetic_data.rename(columns={'ID_REF': 'ID'}).astype({'ID': 'str'})
    genetic_data.set_index('ID', inplace=True)

    return genetic_data


def apply_gene_mapping(expression_df, mapping_df):
    """
    Converts measured data about gene probes into gene expression data.
    Handles the potential many-to-many relationship between probes and genes.

    Parameters:
    expression_df (DataFrame): A DataFrame with gene expression data, indexed by 'ID'.
    mapping_df (DataFrame): A DataFrame mapping 'ID' to 'Gene', with 'ID' as a column.

    Returns:
    DataFrame: A DataFrame with mean gene expression values, indexed by 'Gene'.
    """

    # Define a regex pattern for splitting gene names
    split_pattern = r';|\|+|/{2,}|,|\[|\]|\(|\)'

    # Split the 'Gene' column in 'mapping_df' using the regex pattern
    mapping_df['Gene'] = mapping_df['Gene'].str.split(split_pattern)
    mapping_df = mapping_df.explode('Gene')

    # Set 'ID' as the index of 'mapping_df' for merging
    mapping_df.set_index('ID', inplace=True)

    # Merge 'mapping_df' with 'expression_df' by their indices
    merged_df = mapping_df.join(expression_df)

    # Group by 'Gene' and calculate the mean expression values
    gene_expression_df = merged_df.groupby('Gene').mean().dropna()

    return gene_expression_df


def normalize_gene_symbols(gene_symbols, batch_size=1000):
    """Normalize human gene symbols in batches using the 'mygenes' library"""
    mg = mygene.MyGeneInfo()
    normalized_genes = {}

    # Process in batches
    for i in range(0, len(gene_symbols), batch_size):
        batch = gene_symbols[i:i + batch_size]
        results = mg.querymany(batch, scopes='symbol', fields='symbol', species='human', verbose=False)

        # Update the normalized_genes dictionary with results from this batch
        for gene in results:
            normalized_genes[gene['query']] = gene.get('symbol', None)

    # Return the normalized symbols in the same order as the input
    return [normalized_genes.get(symbol) for symbol in gene_symbols]


def normalize_gene_symbols_in_index(gene_df):
    """Normalize the human gene symbols at the index of a dataframe, and replace the index with its normalized version.
    Remove the rows where the index failed to be normalized."""
    normalized_gene_list = normalize_gene_symbols(gene_df.index.tolist())
    assert len(normalized_gene_list) == len(gene_df.index)
    gene_df.index = normalized_gene_list
    gene_df = gene_df[gene_df.index.notnull()]
    return gene_df


def get_feature_data(clinical_df, row_id, feature, convert_fn):
    """select the row corresponding to a feature in the sample characteristics dataframe, and convert the feature into
    a binary or continuous variable"""
    clinical_df = clinical_df.iloc[row_id:row_id + 1].drop(columns=['!Sample_geo_accession'], errors='ignore')
    clinical_df.index = [feature]
    clinical_df = clinical_df.applymap(convert_fn)

    return clinical_df

def judge_binary_variable_biased(dataframe, col_name, min_proportion=0.1, min_num=5):
    """
    Check if the distribution of a binary variable in the dataset is too biased to be usable for analysis
    :param dataframe:
    :param col_name:
    :param min_proportion:
    :param min_num:
    :return:
    """
    label_counter = dataframe[col_name].value_counts()
    total_samples = len(dataframe)
    rare_label_num = label_counter.min()
    rare_label = label_counter.idxmin()
    rare_label_proportion = rare_label_num / total_samples

    print(
        f"For the feature \'{col_name}\', the least common label is '{rare_label}' with {rare_label_num} occurrences. This represents {rare_label_proportion:.2%} of the dataset.")

    biased = (len(label_counter) < 2) or ((rare_label_proportion < min_proportion) and (rare_label_num < min_num))
    return bool(biased)


def judge_continuous_variable_biased(dataframe, col_name):
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


def check_rows_and_columns(dataframe, display=False):
    """
    Get the lists of row names and column names of a dataset, and optionally observe them.
    :param dataframe:
    :param display:
    :return:
    """
    dataframe_rows = dataframe.index.tolist()
    if display:
        print(f"The dataset has {len(dataframe_rows)} rows, such as {dataframe_rows[:20]}")
    dataframe_cols = dataframe.columns.tolist()
    if display:
        print(f"\nThe dataset has {len(dataframe_cols)} columns, such as {dataframe_cols[:20]}")
    return dataframe_rows, dataframe_cols


def xena_convert_trait(row_index: str):
    """
    Convert the trait information from Sample IDs to labels depending on the last two digits.
    Tumor types range from 01 - 09, normal types from 10 - 19.
    :param row_index: the index value of a row
    :return: the converted value
    """
    last_two_digits = int(row_index[-2:])

    if 1 <= last_two_digits <= 9:
        return 1
    elif 10 <= last_two_digits <= 19:
        return 0
    else:
        return -1


def xena_convert_gender(cell: str):
    """Convert the cell content about gender to a binary value
    """
    if isinstance(cell, str):
        cell = cell.lower()

    if cell == "female":
        return 0
    elif cell == "male":
        return 1
    else:
        return None


def xena_convert_age(cell: str):
    """Convert the cell content about age to a numerical value using regular expression
    """
    match = re.search(r'\d+', str(cell))
    if match:
        return int(match.group())
    else:
        return None

def get_unique_values_by_row(dataframe, max_len=30):
    """
    Organize the unique values in each row of the given dataframe, to get a dictionary
    :param dataframe:
    :param max_len:
    :return:
    """
    if '!Sample_geo_accession' in dataframe.columns:
        dataframe = dataframe.drop(columns=['!Sample_geo_accession'])
    unique_values_dict = {}
    for index, row in dataframe.iterrows():
        unique_values = list(row.unique())[:max_len]
        unique_values_dict[index] = unique_values
    return unique_values_dict


def xena_select_clinical_features(clinical_df, trait, age_col=None, gender_col=None):
    feature_list = []
    trait_data = clinical_df.index.to_series().apply(xena_convert_trait).rename(trait)
    feature_list.append(trait_data)
    if age_col:
        age_data = clinical_df[age_col].apply(xena_convert_age).rename("Age")
        feature_list.append(age_data)
    if gender_col:
        gender_data = clinical_df[gender_col].apply(xena_convert_gender).rename("Gender")
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


def geo_merge_clinical_genetic_data(clinical_df, genetic_df):
    """
    Merge the clinical features and gene expression features from two dataframes into one dataframe
    """
    if 'ID' in genetic_df.columns:
        genetic_df = genetic_df.rename(columns={'ID': 'Gene'})
    if 'Gene' in genetic_df.columns:
        genetic_df = genetic_df.set_index('Gene')
    merged_data = pd.concat([clinical_df, genetic_df], axis=0).T.dropna()
    return merged_data


def judge_and_remove_biased_features(df, trait, trait_type):
    assert trait_type in ["binary", "continuous"], f"The trait must be either a binary or a continuous variable!"
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


def save_cohort_info(cohort: str, info_path: str, is_available: bool, is_biased: Optional[bool] = None,
                     df: Optional[pd.DataFrame] = None, note: str = '') -> None:
    """
    Add or update information about the usability and quality of a dataset for statistical analysis.

    Parameters:
    cohort (str): A unique identifier for the dataset.
    info_path (str): File path to the JSON file where records are stored.
    is_available (bool): Indicates whether both the genetic data and trait data are available in the dataset, and can be
     preprocessed into a dataframe.
    is_biased (bool, optional): Indicates whether the dataset is too biased to be usable.
        Required if `is_available` is True.
    df (pandas.DataFrame, optional): The preprocessed dataset. Required if `is_available` is True.
    note (str, optional): Additional notes about the dataset.

    Returns:
    None: The function does not return a value but updates or creates a record in the specified JSON file.
    """
    if is_available:
        assert (df is not None) and (is_biased is not None), "'df' and 'is_biased' should be provided if this cohort " \
                                                             "is relevant."
    is_usable = is_available and (not is_biased)
    new_record = {"is_usable": is_usable,
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

def preview_df(df, n=5):
    return df.head(n).to_dict(orient='list')
