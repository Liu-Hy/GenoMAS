import os
import json
import ast
import numpy as np
import pandas as pd
import itertools
from typing import Callable, Optional, List, Tuple, Dict, Union, Any
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sparse_lmm import LMM
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)
def read_json_to_dataframe(json_file: str) -> pd.DataFrame:
    """
    Reads a JSON file and converts it into a pandas DataFrame.

    Args:
    json_file (str): The path to the JSON file containing the data.

    Returns:
    DataFrame: A pandas DataFrame with the JSON data.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    return pd.DataFrame.from_dict(data, orient='index').reset_index().rename(columns={'index': 'cohort_id'})
def filter_and_rank_cohorts(json_file: str, condition: Union[str, None] = None) -> Tuple[
    Union[str, None], pd.DataFrame]:
    """
    Reads a JSON file, filters cohorts based on usability and an optional condition, then ranks them by sample size.

    Args:
    json_file (str): The path to the JSON file containing the data.
    condition (str, optional): An additional condition for filtering. If None, only 'is_usable' is considered.

    Returns:
    Tuple: A tuple containing the best cohort ID (str or None if no suitable cohort is found) and
           the filtered and ranked DataFrame.
    """
    df = read_json_to_dataframe(json_file)
    if df.empty:
        return None, df
    if condition:
        if condition.lower() in ['age', 'gender']:
            condition = 'has_' + condition.lower()
        assert condition in ['has_age', 'has_gender']
        filtered_df = df[(df['is_usable'] == True) & (df[condition] == True)]
    else:
        filtered_df = df[df['is_usable'] == True]
    ranked_df = filtered_df.sort_values(by='sample_size', ascending=False)
    best_cohort_id = ranked_df.iloc[0]['cohort_id'] if not ranked_df.empty else None

    return best_cohort_id, ranked_df

def select_and_load_cohort(data_root: str, trait: str, condition=None, is_two_step=True, gene_info_path=None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[List]]:
    """
    Selects the best cohort data for specified trait (and optionally condition) from a given data root directory, load
    the data and find the gene regressors if in two-step mode. This function supports data selection for both single-step
    and two-step regression based on the is_two_step flag.

    Args:
        data_root (str): The root directory containing cohort data.
        trait (str): The trait of interest.
        condition (str, optional): The condition of interest; required if is_two_step is True.
        is_two_step (bool, optional): If True, will be used in two-step regression using data from both trait and condition
                                      along with gene information. Requires condition and gene_info_path to be specified.
        gene_info_path (str, optional): Path to gene information file; required if is_two_step is True.

    Returns:
        tuple: A tuple containing:
               - trait_data (Optional[pd.DataFrame]): Data for the selected trait cohort.
               - condition_data (Optional[pd.DataFrame]): Data for the selected condition cohort if in two-step mode, otherwise None.
               - gene_regressors (Optional[pd.DataFrame]): Gene regression data if in two-step mode, otherwise None.

    """
    trait_dir = os.path.join(data_root, trait)
    if not condition:
        is_two_step = False
    if not is_two_step:
        trait_cohort_id, trait_info_df = filter_and_rank_cohorts(os.path.join(trait_dir, 'cohort_info.json'), condition)
        if trait_cohort_id is None:
            return None, None, None
        else:
            trait_data = pd.read_csv(os.path.join(trait_dir, trait_cohort_id + '.csv')).astype('float')
            return trait_data, None, None
    else:
        assert gene_info_path is not None, "A path to gene information file must be specified for two-step regression"
        condition_dir = os.path.join(data_root, condition)
        trait_cohort_id, trait_info_df = filter_and_rank_cohorts(os.path.join(trait_dir, 'cohort_info.json'), None)
        condition_cohort_id, condition_info_df = filter_and_rank_cohorts(os.path.join(condition_dir, 'cohort_info.json'), None)
        if trait_cohort_id is None or condition_cohort_id is None:
            print(f"No available data for either the trait or the condition, best cohorts being '{trait_cohort_id}' and '{condition_cohort_id}'")
            return None, None, None
        merged_df = pd.merge(trait_info_df.assign(key=1), condition_info_df.assign(key=1), on='key').drop(columns='key')
        merged_df['sample_product'] = merged_df['sample_size_x'] * merged_df['sample_size_y']
        merged_df = merged_df.sort_values(by='sample_product', ascending=False)
        for index, row in merged_df.iterrows():
            trait_data_path = os.path.join(trait_dir, row['cohort_id_x'] + '.csv')
            condition_data_path = os.path.join(condition_dir, row['cohort_id_y'] + '.csv')
            trait_data = pd.read_csv(trait_data_path).astype('float')
            condition_data = pd.read_csv(condition_data_path).astype('float')
            gene_regressors = get_gene_regressors(trait, condition, trait_data, condition_data, gene_info_path)
            if gene_regressors:
                return trait_data, condition_data, gene_regressors
        print(f"No available cohorts with common regressors for the trait '{trait}' and the condition '{condition}'")
        return None, None, None

def normalize_data(X_train, X_test=None):
    """This function computes the mean and standard deviation of the feature matrix (X_train) and uses these statistics
    to normalize X_train. If provided, it can also normalize a separate test feature matrix (X_test) using the same
    statistics. """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Handling columns with std = 0
    std_no_zero = np.where(std == 0, 1, std)

    X_train_normalized = (X_train - mean) / std_no_zero
    # Set normalized values to 0 where std was 0
    X_train_normalized[:, std == 0] = 0

    if X_test is not None:
        X_test_normalized = (X_test - mean) / std_no_zero
        X_test_normalized[:, std == 0] = 0
    else:
        X_test_normalized = None

    return X_train_normalized, X_test_normalized

def detect_batch_effect(X):
    """
    Detect potential batch effects in a dataset using eigenvalues of XX^T after centering the data.

    Args:
    X (numpy.ndarray): A feature matrix with shape (n_samples, n_features).

    Returns:
    bool: True if a potential batch effect is detected, False otherwise.
    """
    n_samples, n_features = X.shape
    X_centered = X - X.mean(axis=0)
    XXt = np.dot(X_centered, X_centered.T)

    # Compute the eigenvalues of XX^T
    eigen_values = np.linalg.eigvalsh(XXt)  # Using eigvalsh since XX^T is symmetric
    eigen_values = sorted(eigen_values, reverse=True)[:10]  # Focus on the largest 10 eigenvalues
    eigen_values = np.array(eigen_values)
    normalized_ev = eigen_values / eigen_values[0]

    # Check for large gaps in the eigenvalues
    for i in range(len(normalized_ev) - 1):
        gap = normalized_ev[i] - normalized_ev[i + 1]
        if gap > 200 / n_samples:  # You may need to adjust this threshold. 200
            return True

    return False

class ResidualizationRegressor:
    def __init__(self, regression_model_constructor, params=None):
        if params is None:
            params = {}
        self.regression_model = regression_model_constructor(**params)
        self.beta_Z = None  # Coefficients for regression of Y on Z
        self.beta_X = None  # Coefficients for regression of residual on X
        self.neg_log_p_values = None  # Negative logarithm of p-values
        self.p_values = None  # Actual p-values

    def _reshape_data(self, data):
        """
        Reshape the data to ensure it's in the correct format (2D array).
        :param data: The input data (can be 1D or 2D array).
        :return: Reshaped 2D array.
        """
        if data.ndim == 1:
            return data.reshape(-1, 1)
        return data

    def _reshape_output(self, data):
        """
        Reshape the output data to ensure it's in the correct format (1D array).
        :param data: The output data (can be 1D or 2D array).
        :return: Reshaped 1D array.
        """
        if data.ndim == 2 and data.shape[1] == 1:
            return data.ravel()
        return data

    def fit(self, X, Y, Z=None):
        X = self._reshape_data(X)
        Y = self._reshape_data(Y)

        if Z is not None:
            Z = self._reshape_data(Z)
            # Step 1: Linear regression of Y on Z
            Z_ones = np.column_stack((np.ones(Z.shape[0]), Z))
            self.beta_Z = np.linalg.pinv(Z_ones.T @ Z_ones) @ Z_ones.T @ Y
            Y_hat = Z_ones @ self.beta_Z
            e_Y = Y - Y_hat  # Residual of Y
        else:
            e_Y = Y
        self.regression_model.fit(X, e_Y)

        # Obtain coefficients from the regression model
        if hasattr(self.regression_model, 'coef_'):
            self.beta_X = self.regression_model.coef_
        elif hasattr(self.regression_model, 'getBeta'):
            beta_output = self.regression_model.getBeta()
            self.beta_X = self._reshape_output(beta_output)

        # Obtain negative logarithm of p-values, if available
        if hasattr(self.regression_model, 'getNegLogP'):
            neg_log_p_output = self.regression_model.getNegLogP()
            if neg_log_p_output is not None:
                self.neg_log_p_values = self._reshape_output(neg_log_p_output)
                self.p_values = np.exp(-self.neg_log_p_values)
                # Handling p-values depending on presence of Z
                if Z is not None:
                    p_values_Z = np.full(Z.shape[1], np.nan)
                    self.p_values = np.concatenate((p_values_Z, self.p_values))

    def predict(self, X, Z=None):
        X = self._reshape_data(X)
        e_Y = self.regression_model.predict(X)

        if Z is not None:
            Z = self._reshape_data(Z)
            Z_ones = np.column_stack((np.ones(Z.shape[0]), Z))
            Y = e_Y + Z_ones @ self.beta_Z.ravel()
        else:
            Y = e_Y
        return Y

    def get_coefficients(self):
        if self.beta_Z is not None:
            return np.concatenate((self.beta_Z[1:].ravel(), self.beta_X.ravel()))
        return self.beta_X.ravel()

    def get_p_values(self):
        return self.p_values


def gene_jaccard(pred, ref):
    p = set(pred)
    r = set(ref)
    if len(p.union(r)):
        iou = len(p.intersection(r)) / len(p.union(r))
    else:
        iou = 0
    return iou

def gene_jaccard2(pred, ref):
    min_len = min(len(pred), len(ref))
    p = set(pred[:min_len])
    r = set(ref[:min_len])
    if len(p.union(r)):
        iou = len(p.intersection(r)) / len(p.union(r))
    else:
        iou = 0
    return iou

def gene_precision_at_50(pred, ref):
    pred_top = pred[:50]
    if len(pred_top) > 0:
        precision = sum([p in ref for p in pred_top]) / len(pred_top)
    else:
        precision = 0
    return precision

def gene_precision(pred, ref):
    if len(pred):
        precision = sum([p in ref for p in pred]) / len(pred)
    else:
        precision = 0
    return precision

def gene_recall(pred, ref):
    if len(ref):
        recall = sum([p in pred for p in ref]) / len(ref)
    else:
        recall = 0
    return recall

def gene_f1(pred, ref):
    prec = gene_precision(pred, ref)
    rec = gene_recall(pred, ref)
    if prec + rec == 0:  # Prevent division by zero
        return 0
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1

def evaluate_gene_selection(pred: List[str], ref: List[str]):
    return {
        'precision': gene_precision(pred, ref) * 100,
        'recall': gene_recall(pred, ref) * 100,
        'f1': gene_f1(pred, ref) * 100,
        'jaccard': gene_jaccard(pred, ref) * 100
    }

def cross_validation(
    model_constructor: Callable,
    model_params: Dict[str, Any],
    X: np.ndarray,
    Y: np.ndarray,
    var_names: List[str],
    trait: str,
    gene_info_path: str,
    condition: Optional[str] = None,
    Z: Optional[np.ndarray] = None,
    k: int = 5
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation for either classification or regression models,
    assessing both prediction accuracy and variable selection precision.

    Parameters:
    - model_constructor: Callable that constructs a model instance.
    - model_params: Dictionary of parameters to pass to the model constructor.
    - X: Input features as a numpy array.
    - Y: Target variable as a numpy array.
    - var_names: List of names of all variables considered in the model.
    - trait: Name of the trait under analysis.
    - gene_info_path: Path to the file containing gene information.
    - condition: Optional; name of the condition considered in the model, if applicable.
    - Z: Optional; conditions as a numpy array, if applicable.
    - k: Number of folds for cross-validation.

    Returns:
    - A dictionary containing the averaged results from the cross-validation,
      including metrics like accuracy, precision, recall, F1 score, NMSE, and R-squared,
      along with variable selection metrics based on gene identification.
    """
    np.random.seed(42)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    fold_size = len(X) // k
    performances = []

    target_type = 'binary' if len(np.unique(Y)) == 2 else 'continuous'
    for i in range(k):
        # Split data into train and test based on the current fold
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, test_indices)

        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]
        normalized_X_train, normalized_X_test = normalize_data(X_train, X_test)

        if Z is not None:
            Z_train, Z_test = Z[train_indices], Z[test_indices]
            normalized_Z_train, normalized_Z_test = normalize_data(Z_train, Z_test)
        else:
            normalized_Z_train = normalized_Z_test = None

        model = ResidualizationRegressor(model_constructor, model_params)
        model.fit(normalized_X_train, Y_train, normalized_Z_train)
        predictions = model.predict(normalized_X_test, normalized_Z_test)

        performance = {}
        if target_type == 'binary':
            predictions = (predictions > 0.5).astype(int)
            Y_test = (Y_test > 0.5).astype(int)
            performance['prediction'] = {
                "accuracy": accuracy_score(Y_test, predictions) * 100,
                "precision": precision_score(Y_test, predictions, zero_division=0) * 100,
                "recall": recall_score(Y_test, predictions, zero_division=0) * 100,
                "f1": f1_score(Y_test, predictions, zero_division=0) * 100
            }
        elif target_type == 'continuous':
            nmse = np.sum((Y_test - predictions) ** 2) / np.sum((Y_test - np.mean(Y_test)) ** 2)
            rsq = r2_score(Y_test, predictions)
            performance['prediction'] = {
                "nmse": nmse,
                "r_squared": rsq
            }

        pred_genes = interpret_result(model, var_names, trait, condition)["Variable"]
        ref_genes = get_known_related_genes(gene_info_path, feature=trait)
        var_genes = [v for v in var_names if v not in [trait, condition]]
        ref_genes = [r for r in ref_genes if r in var_genes]
        performance["selection"] = evaluate_gene_selection(pred_genes, ref_genes)
        performances.append(performance)

    # Calculate average performance across all metrics
    cv_means = {}
    for metric in performances[0]:
        if isinstance(performances[0][metric], dict):
            cv_means[metric] = {}
            for submetric in performances[0][metric]:
                cv_means[metric][submetric] = np.mean([p[metric][submetric] for p in performances])
        else:
            cv_means[metric] = np.mean([p[metric] for p in performances])

    print(f'The cross-validation performance: {cv_means}')

    return cv_means

def tune_hyperparameters(
    model_constructor: Callable,
    param_values: List[float],
    X: np.ndarray,
    Y: np.ndarray,
    var_names: list,
    trait: str,
    gene_info_path: str,
    condition: Optional[str] = None,
    Z: Optional[np.ndarray] = None,
    fixed_params: Optional[Dict[str, Any]] = {},
    k: int = 5
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Tune hyperparameters for a given model by exploring combinations of parameter values.

    This function performs cross-validation to find the best hyperparameter settings based on the precision
    of gene identification. It returns the best configuration along with the top performance metrics for both
    prediction and gene identification.

    Parameters:
    - model_constructor: A callable that returns an instance of the model to be used.
    - param_values: List specifying the possible values of hyperparameter(s) to be tuned.
    - X: Input features as a numpy array.
    - Y: Target variable as a numpy array.
    - var_names: List of names of all variables considered in the model.
    - trait: Name of the trait under analysis.
    - gene_info_path: File path to the gene information data.
    - condition: Optional; name of the condition considered in the model, if applicable.
    - Z: Optional; conditions as a numpy array, if applicable.
    - fixed_params: Dictionary specifying hyperparameters and their values that are set different from default, but do
      not need to be tuned.
    - k: Number of folds for cross-validation.

    Returns:
    - Tuple containing:
        1. Dictionary of the best hyperparameters based on gene identification precision.
        2. Dictionary of the best performances for 'selection' and 'prediction'.
    """
    best_selection_score = -np.inf
    best_prediction_score = -np.inf
    best_config = {}
    best_performance = {}
    prediction_metric = "f1" if len(np.unique(Y)) == 2 else "r_squared"
    # Generate all combinations of parameters to be tuned
    if model_constructor == LMM:
        param = "lamda"
    elif model_constructor == Lasso:
        param = "alpha"
    tune_params = {param: param_values}
    keys, values = zip(*tune_params.items())
    for combination in itertools.product(*values):
        # Combine the fixed parameters with the current combination of tuning parameters
        current_params = dict(zip(keys, combination))
        current_params.update(fixed_params)

        results = cross_validation(model_constructor, current_params, X, Y, var_names, trait, gene_info_path, condition,
                                   Z, k)

        current_prediction_score = results["prediction"][prediction_metric]
        if current_prediction_score > best_prediction_score:
            best_prediction_score = current_prediction_score
            best_performance["prediction"] = results["prediction"]

        current_selection_score = results["selection"]["f1"]
        if current_selection_score > best_selection_score:
            best_selection_score = current_selection_score
            best_config = current_params
            best_performance["selection"] = results["selection"]

    return best_config, best_performance

def get_known_related_genes(file_path, feature):
    """Read a csv file into a dataframe about gene-trait association, and get the gene symbols related to a given
    trait"""
    related_gene_df = pd.read_csv(file_path)
    related_gene_df = related_gene_df.loc[:, ['Trait', 'Related_Genes']].set_index('Trait')
    if feature not in related_gene_df.index:
        print(f"The gene info file does not contain genes related to the feature '{feature}'.")
        return None
    feature_related_genes = ast.literal_eval(related_gene_df.loc[feature].tolist()[0])
    # feature_related_genes = [gn.strip() for gn in feature_related_genes if isinstance(gn, str)]

    return feature_related_genes


def get_gene_regressors(trait, condition, trait_df, condition_df, gene_info_path):
    """Find the appropriate genes for two-step regression. Compare the indices of two dataframes to find the genes
    in common, and select those that are known to be related to a trait"""
    gene_regressors = None
    related_genes = get_known_related_genes(gene_info_path, condition)
    if not related_genes:
        return None
    genes_in_trait_data = set(trait_df.columns) - {'Age', 'Gender', trait}
    genes_in_condition_data = set(condition_df.columns) - {'Age', 'Gender', trait}

    common_genes_across_data = genes_in_trait_data.intersection(genes_in_condition_data)
    if len(common_genes_across_data) == 0:
        #print("The trait and condition datasets have no genes in common. Please try other datasets")
        return None
    else:
        ##print(
            ## f"The trait and condition datasets have {len(common_genes_across_data)} genes in common, such as {list(common_genes_across_data)[:10]}.")
        common_genes = [g for g in related_genes if g in common_genes_across_data]
        if len(common_genes) > 0:
            gene_regressors = list(common_genes)[:10]
            #print(
                #f"Found {len(common_genes)} candidate genes that can be used in two-step regression analysis, such as {gene_regressors[:10]}.")
        else:
            #print(
                #f"The condition and trait datasets have common genes, but among them we didn't find indicator genes for the condition")
            return None

    return gene_regressors


def normalize_trait(trait):
    trait = '_'.join(trait.split())
    normalized_trait = ''.join(trait.split("'"))
    return normalized_trait

def interpret_result(model: ResidualizationRegressor, var_names: List[str], trait: str, condition = None,
                     threshold: float = 0.05, print_output = False) -> dict:
    """This function interprets and reports the result of a trained linear regression model, where the regressor
    consists of one variable about some biomedical condition and multiple variables about genetic factors.
    The function extracts coefficients and p-values from the model, identifies significant genes based on
    p-values or non-zero coefficients, depending on the availability of p-values, and optionally prints the output.

    Parameters:
        model (Any): The trained regression Model.
        var_names (List[str]): List of names of all the variables involved in the regression analysis.
        trait (str): The target trait of interest.
        condition (str): The specific condition to examine within the model.
        threshold (float): Significance level for p-value correction. Defaults to 0.05.
        print_output (bool): Flag to determine whether to print the output to the console. Defaults to False.

    Returns:
        dict: A dictionary containing the list of significant genes, sorted by their importance, and the corresponding
        coefficient magnitude or corrected p-value.
    """
    assert isinstance(model, ResidualizationRegressor), "The model must be an instance of the ResidualizationRegressor" \
                                                        "class."
    feature_names = [var for var in var_names if var != trait]

    # If a condition is specified, move it to the beginning of the list
    if condition:
        if condition in feature_names:
            feature_names.remove(condition)
        feature_names.insert(0, condition)

    coefficients = model.get_coefficients().reshape(-1).tolist()
    p_values = model.get_p_values()
    if p_values is None:
        regression_df = pd.DataFrame({
            'Variable': feature_names,
            'Coefficient': coefficients
        })
    else:
        regression_df = pd.DataFrame({
            'Variable': feature_names,
            'Coefficient': coefficients,
            'p_value': p_values.reshape(-1).tolist()
        })

    if condition is not None:
        condition_effect = regression_df[regression_df['Variable'] == condition].iloc[0]
        if print_output:
            print(f"Effect of the condition on the target variable:")
            print(f"Variable: {condition}, Coefficient: {condition_effect['Coefficient']:.4f}")
        gene_regression_df = regression_df[regression_df['Variable'] != condition]
    else:
        gene_regression_df = regression_df

    if p_values is None:
        significant_genes_df = gene_regression_df[gene_regression_df['Coefficient'] != 0].copy()
        significant_genes_df['Absolute Coefficient'] = significant_genes_df['Coefficient'].abs()
        significant_genes_df = significant_genes_df.sort_values('Absolute Coefficient', ascending=False)
    else:
        corrected_p_values = multipletests(gene_regression_df['p_value'], alpha=threshold, method='fdr_bh')[1]
        gene_regression_df.loc[:, 'corrected_p_value'] = corrected_p_values
        significant_genes_df = gene_regression_df[gene_regression_df['corrected_p_value'] < threshold]
        significant_genes_df = significant_genes_df.sort_values('corrected_p_value', ascending=True)

    if print_output:
        print(f"Found {len(significant_genes_df)} significant genes associated with the trait '{trait}', "
              f"conditional on the factor '{condition}'.")

    return significant_genes_df.to_dict(orient="list")

def save_result(significant_genes: Dict[str, any], performance: Dict[str, any], output_dir: str, condition: Optional[str] = None):
    """
    Saves the results of gene identification and model performance metrics to a JSON file.

    Args:
        significant_genes (dict): Dictionary containing identified significant genes and their related metrics.
        performance (dict): Dictionary containing performance metrics from cross-validation.
        output_dir (str): Directory path where the output file will be saved.
        condition (str, optional): Specifies the condition related to the gene identification. Include this parameter if
        the model considers a specific condition; otherwise, leave it as None.

    Outputs:
        A JSON file named 'significant_genes_condition_{condition}.json' in the specified directory, containing both the
        significant genes and cross-validation performance data.
    """
    output_path = os.path.join(output_dir, f'significant_genes_condition_{condition}.json')
    output_data = {'significant_genes': significant_genes, 'cv_performance': performance}

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)