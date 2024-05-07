import os
from openai import AzureOpenAI
import pandas as pd
import numpy as np
import json
import ast
import re
import subprocess
from utils import *

pairs = pd.read_csv("trait_condition_pairs.csv")

rel = pd.read_csv("trait_related_genes.csv").drop(columns=["Unnamed: 0"])
rel['Related_Genes'] = rel['Related_Genes'].apply(ast.literal_eval)
t2g = pd.Series(rel['Related_Genes'].values, index=rel['Trait']).to_dict()

# Azure OpenAI Client setup
client = AzureOpenAI(
    api_key="57983a2e88fa4d6b81205a8d55d9bd46",
    api_version="2023-10-01-preview",
    azure_endpoint="https://haoyang2.openai.azure.com/"
)


def call_openai_gpt_chat(trait, condition, trait_data_path, output_dir, utils_code, two_step=False, genes=None):
    if not two_step:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": STEP_ONE_PROMPT.format(trait=trait, condition=condition, trait_data_path=trait_data_path,
                                               output_dir=output_dir, utils_code=utils_code)}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": STEP_TWO_PROMPT.format(trait=trait, condition=condition, trait_data_path=trait_data_path,
                                               condition_data_path=condition_data_path, output_dir=output_dir,
                                               utils_code=utils_code, genes=genes)}
        ]
    response = client.chat.completions.create(
        model="gpt-4",  # Adjust the model name as needed
        messages=messages
    )
    return response.choices[0].message.content


def parse_code(rsp):
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text


CODE_TEMPLATE = """
Code:
{code}

Code result:

STDOUT:
{stdout}

STDERR:
{stderr}
"""

STEP_ONE_PROMPT = """
Role: You are a professional engineer; the main goal is to write good code for data analysis.
Background:
1. You can leverage the following functions from 'utils.py' in the same folder:
{utils_code}
2. Path to the input data about the trait '{trait}': {trait_data_path}
3. The output directory: {output_dir}

    Instruction: Write code to solve the following research question: What are the genetic factors related to the trait '{trait}' when considering the influence of the condition '{condition}'?
Based on the context and the following instructions, write code that is elegant and easy to read.
1. Load the input data about the trait into a dataframe, and convert to the float type.
2. We need only one condition from 'Age' and 'Gender'. Remove the redundant column if present.
3. Select the data in relevant columns for regression analysis. We need three numpy arrays X, Y and Z. Y is the trait data from the column '{trait}', Z is the condition data from the column '{condition}', and X is the rest of the data. We want to analyze and find the genetic factors related to the trait when considering the influence of the condition.
4. Check whether the feature X shows batch effect. Hint: you may use the 'detect_batch_effect' function from utils.
5. Select appropriate models based on whether the dataset has batch effect. 
if has_batch_effect:
    model_constructor = VariableSelection
    model_params = {{'modified': True, 'lamda': 3e-4}}  # Note that the 'lamda' is not 'lambda'
else:
    model_constructor = Lasso
    model_params = {{'alpha': 1.0, 'random_state': 42}}
    
6. Conduct cross-validation on the model from the previous step. Hint: You may use the 'cross_validation' function. Please set the parameter 'target_type' properly, either 'binary' or 'continuous'. This can be determined by whether the array 'Y' has two unique values.
7. Normalize the X and Z to have a mean of 0 and standard deviation of 1. Hint: you may use the 'normalize_data' function from utils.
8. Train the model on the whole dataset.
9. Interpret the trained model to identify the effect of the condition and significant genes. Set threshold to 0.05 for the p-value. Hint: You may use the 'interpret_result' function from utils, and use the output_dir given. For the parameter 'feature_names', we will only need the condition and gene, so please remove the trait {trait} from the column names.


NOTICE: Please import all the functions in 'utils.py' at the beginning of the code, and feel free to use '*' in the import statement.

Return ```python your_code_here ``` with NO other texts. your_code_here is a placeholder.
your code:
        """

STEP_TWO_PROMPT = """
Role: You are a professional engineer; the main goal is to write good code for data analysis.
Background:
1. You can leverage the following functions from 'utils.py' in the same folder:
{utils_code}
2. Path to the input data about the trait '{trait}': {trait_data_path}
3. Path to the input data about the condition '{condition}': {condition_data_path}
4. The output directory: {output_dir}
5. A list of gene symbols used as common regressors: regressors = {genes}
Note: a trait or a condition can be a multi-word phrase like 'Breast Cancer'. While they may appear as 'Breast-Cancer' in file paths, please stick to their
surface form as shown in the instruction.

Instruction: Write code to solve the following research question: What are the genetic factors related to the trait 
'{trait}' when considering the influence of the condition '{condition}'?
When we don't have data about the trait and condition from the same group of people, we can still solve the problem by
 properly using the two-step regression approach. This approach involves two steps to investigate the relationship among the trait, the condition, and the genes. 
Before doing the analysis, we need to prepare a gene expression dataset about the condition and the trait respectively.
In the first stage, we use the condition dataset to regress the condition on a few important genes, which are considered 
as "indicator genes". To choose the indicator genes, we query a database to get candidate genes that are found 
related to the condition according to biomedical knowledge, and preserve the candidate genes that exist in both the 
condition dataset and the trait dataset. If such genes exist, they serve as the common regressors between the two datasets, making 
the two-stage regression possible. We can regress the condition on those genes in the condition dataset, and use the
 trained regression model to predict the condition on the trait dataset. This results in an augmented dataset with both 
 trait and condition information from the same individuals.
In the second stage, we can directly conduct regression on the augmented trait dataset to explore the relationship between 
the genes, trait, and condition. Note that since the condition values were predicted from the common regressor genes, and we 
include the condition as a regressor now, we need to exclude those genes from the regressors in this stage to 
prevent co-linearity.
Below are more detailed instructions. Based on the context and the instructions, write code that is elegant and easy to read.
1. Load the input data about the trait and the condition into two dataframes, and convert them to float type respectively.
2. From the condition dataframe, select the columns corresponding to the gene regressors as 'X_condition', and the column corresponding to the condition value as 'Y_condition', and convert them to numpy arrays.
3. Determine the data type of the condition, which is either 'binary' or 'continuous', by seeing whether the array of condition values has two unique values.
## The first step regression
4. Please choose an appropriate regression model for the condition. 
   - If the condition is a binary variable, then use the LogisticRegression model, and choose to use L1 penalty if 'X_condition' has more columns than rows.
   - If the condition is a continuous variable, then choose Lasso or LinearRegression depending on whether 'X_condition' has more columns than rows.
   Normalize 'X_condition' to a mean of 0 and std of 1. With the model you chose, fit the model on 'normalized_X_condition' and 'Y_condition'
5. From the trait dataframe, select the columns corresponding to the common gene regressors to get a numpy array, and normalize it to a mean of 0 and std of 1.
6. With the model trained in Step 4, predict the condition of the samples in the trait dataframe based on the normalized gene regressors. 
  If the condition is a continuous variable, use the predict() method of the model to get the predicted values of the condition; otherwise, use the predict_proba() 
  method and select the column corresponding to the positive label, to get the predicted probability of the condition being true. 
  Add a column named {condition} to the trait dataframe, storing predicted condition values.
7. From the trait dataframe, drop the columns about the common gene regressors, and drop the columns 'Age' and 'Gender' if any of them exist.
## The second step regression
8. From the trait dataframe, select the data in relevant columns for regression analysis. We need three numpy arrays X, Y and Z. Y is the trait data from the column '{trait}', Z is the condition data from the column '{condition}', and X is the rest of the data. We want to analyze and find the genetic factors related to the trait when considering the influence of the condition.
9. Check whether the feature X shows batch effect. Hint: you may use the 'detect_batch_effect' function from utils.
10. Select appropriate models based on whether the dataset has batch effect. 
if has_batch_effect:
    model_constructor = VariableSelection
    model_params = {{'modified': True, 'lamda': 3e-4}}  # Note that the 'lamda' is not 'lambda'
else:
    model_constructor = Lasso
    model_params = {{'alpha': 1.0, 'random_state': 42}}

11. Conduct cross-validation on the model from the previous step. Hint: You may use the 'cross_validation' function. Please set the parameter 'target_type' properly, either 'binary' or 'continuous'. This can be determined by whether the array 'Y' has two unique values.
12. Normalize the X and Z to have a mean of 0 and standard deviation of 1. Hint: you may use the 'normalize_data' function from utils.
13. Train the model on the whole dataset.
14. Interpret the trained model to identify the effect of the condition and significant genes. Set threshold to 0.05 for the p-value. Hint: You may use the 'interpret_result' function from utils, and use the output_dir given. For the parameter 'feature_names', we will only need the condition and gene, so please remove the trait {trait} from the column names.

NOTICE: Please import all the functions in 'utils.py' at the beginning of the code, and feel free to use '*' in the import statement.

Return ```python your_code_here ``` with NO other texts. your_code_here is a placeholder.
your code:
        """

rel = pd.read_csv("trait_related_genes.csv").drop(columns=["Unnamed: 0"])
rel['Related_Genes'] = rel['Related_Genes'].apply(ast.literal_eval)
t2g = pd.Series(rel['Related_Genes'].values, index=rel['Trait']).to_dict()

for i, (index, row) in enumerate(pairs.iterrows()):
    try:
        trait, condition = row['Trait'], row['Condition']
        nm_trait = normalize_trait(trait)
        nm_condition = normalize_trait(condition)
        trait_dir = os.path.join('/home/techt/Desktop/a4s/gold_subset', nm_trait)
        output_dir = os.path.join('./output2', nm_trait)
        os.makedirs(output_dir, exist_ok=True)
        utils_code = "".join(open("./utils.py", 'r').readlines())
        if condition in ['Age', 'Gender']:
            trait_cohort_id, _ = filter_and_rank_cohorts(os.path.join(trait_dir, 'cohort_info.json'), condition)
            trait_data_path = os.path.join(trait_dir, trait_cohort_id + '.csv')

            # Call Azure OpenAI to generate the regression code using chat
            generated_code = call_openai_gpt_chat(nm_trait, nm_condition, trait_data_path, output_dir, utils_code)

            # Process and execute the generated code
            code_text = parse_code(generated_code)
            result = subprocess.run(["python", "-c", code_text], capture_output=True, text=True)
            stdout = result.stdout
            stderr = result.stderr

            print(CODE_TEMPLATE.format(code=code_text, stdout=stdout, stderr=stderr))
        else:
            condition_dir = os.path.join('/home/techt/Desktop/a4s/gold_subset', nm_condition)
            trait_cohort_id, _ = filter_and_rank_cohorts(os.path.join(trait_dir, 'cohort_info.json'))
            condition_cohort_id, _ = filter_and_rank_cohorts(os.path.join(condition_dir, 'cohort_info.json'))
            trait_data_path = os.path.join(trait_dir, trait_cohort_id + '.csv')
            condition_data_path = os.path.join(condition_dir, condition_cohort_id + '.csv')

            trait_data = pd.read_csv(trait_data_path).astype('float')
            condition_data = pd.read_csv(condition_data_path).astype('float')

            related_genes = t2g[condition]
            regressors = get_gene_regressors(trait, trait_data, condition_data, related_genes)
            if regressors is None:
                print(f'No gene regressors for trait \'{trait}\' and condition \'{condition}\'')
                continue

            rel = pd.read_csv("trait_related_genes.csv").drop(columns=["Unnamed: 0"])
            rel['Related_Genes'] = rel['Related_Genes'].apply(ast.literal_eval)
            t2g = pd.Series(rel['Related_Genes'].values, index=rel['Trait']).to_dict()  # the mapping from trait to genes
            related_genes = t2g[condition]
            regressors = get_gene_regressors(trait, trait_data, condition_data, related_genes)
            if regressors is None:
                print(f'No gene regressors for trait {trait} and condition {condition}')
                continue

            generated_code = call_openai_gpt_chat(nm_trait, nm_condition, trait_data_path, output_dir, utils_code, two_step=True, genes=regressors)
            # Process and execute the generated code
            code_text = parse_code(generated_code)
            result = subprocess.run(["python", "-c", code_text], capture_output=True, text=True)
            stdout = result.stdout
            stderr = result.stderr

            print(CODE_TEMPLATE.format(code=code_text, stdout=stdout, stderr=stderr))
    except Exception as e:
        print(f"Error processing row {i}, for the trait '{trait}' and the condition '{condition}'\n: {e}")
        continue
