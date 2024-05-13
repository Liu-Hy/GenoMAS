import os
from openai import AzureOpenAI
import pandas as pd
import numpy as np
import json
import ast
import re
import subprocess
from utils import *
import tqdm
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--num_review", type=int)
argparser.add_argument("--output_dir", type=str)
argparser.add_argument("--data_dir", type=str, default="data/preprocessed")
argparser.add_argument("--condition_pair_path", type=str, default="data/preprocessed/trait_condition_pairs_short.csv")
argparser.add_argument("--related_genes_path", type=str, default="data/preprocessed/trait_related_genes.csv")
args = argparser.parse_args()

pairs = pd.read_csv(args.condition_pair_path)

rel = pd.read_csv(args.related_genes_path).drop(columns=["Unnamed: 0"])
rel['Related_Genes'] = rel['Related_Genes'].apply(ast.literal_eval)
t2g = pd.Series(rel['Related_Genes'].values, index=rel['Trait']).to_dict()

# Azure OpenAI Client setup
client = AzureOpenAI(
    api_key="d4e35aeb201540549739bdbd4f62b957",
    api_version="2023-10-01-preview",
    azure_endpoint="https://yijiang2.openai.azure.com/"
)


def call_openai_gpt_chat(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
         "content": prompt}
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
3. Condition variable name: {condition}
4. The output directory: {output_dir}

NOTICE 1: NOTICE 1: Please USE argparser to recieve the following arguments:
1. the trait variable name (argument name: trait)
2. the input data path about the trait (argument name: input_data_path)
3. the condition variable name (argument name: condition)
4. the output directory and set the ones given as default values (argument name: output_dir)
and SET the ones GIVEN as DEFAULT VALUES.

NOTICE 2: We will test the code you write with different trait, condition, input data and output directory, so please make sure the code is general.

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
7. Normalize the X and Z to have a mean of 0 and standard deviation of 1.
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
5. A list of gene symbols used as common regressors: regressors = {genes} (Please MAKE SURE you explicitly DEFINE this variable in the code and ASSIGN it as {genes})

NOTICE 1: Please USE argparser to recieve the following arguments:
1. the trait variable name (argument name: trait)
2. the path to the input data about the trait (argument name: trait_path)
3. the condition variable name (argument name: condition)
4. the path to the input data about the condition (argument name: condition_path)
5. the output directory (argument name: output_dir)
6. a list of regressors (please make sure the code can accept a list of strings) (argument name: regressors)
and SET the ones GIVEN as DEFAULT VALUES.

NOTICE 2: We will test the code you write with different trait, condition, input data and output directory, so please make sure the code is general.

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
    model_params = {{'modified': True, 'lamda': 3e-4}}  # NOTICE: USE 'lamda' instead of 'lambda' here
else:
    model_constructor = Lasso
    model_params = {{'alpha': 1.0, 'random_state': 42}}

11. Conduct cross-validation on the model from the previous step. Hint: You may use the 'cross_validation' function. Please set the parameter 'target_type' properly, either 'binary' or 'continuous'. This can be determined by whether the array 'Y' has two unique values.
12. Normalize the X to have a mean of 0 and standard deviation of 1.
13. Normalize the Z to have a mean of 0 and standard deviation of 1.
14. Train the model on the whole dataset.
15. Interpret the trained model to identify the effect of the condition and significant genes. Set threshold to 0.05 for the p-value. Hint: You may use the 'interpret_result' function from utils, and use the output_dir given. For the parameter 'feature_names', we will only need the condition and gene, so please remove the trait {trait} from the column names.

NOTICE: Please import all the functions in 'utils.py' at the beginning of the code, and feel free to use '*' in the import statement.

Return ```python your_code_here ``` with NO other texts. your_code_here is a placeholder.
your code:
        """

REVIEW_CODE_PROMPT_TEMPLATE: str = \
    """
    Role: You are a collaborator in this project with an educational background in computer science and computational biology; given the code instruction, written code and the code results, the main goal is to review whether the code can be successfully executed and has accomplished the requirement by the instruction.
    
    Code:
    {code}
    
    Code result:
    STDOUT:
    {stdout}
    
    STDERR:
    {stderr}
    
    
    INSTRUCTION FOR REVIEWING CODE:
    1. Check if the code can be successfully executed by looking at the STDERR, If not, provide a feedback to the code writer for revision and improvement. Note, warning is accetable, but error is not.
    2. Check if the code follows exactly the instruction correctly. If not, provide a feedback to the code writer for revision and improvement.
    3. Return a final decision. If the code can be successfully executed, and the code follows exactly the instruction correctly, RETURN "Final Decision: approved" else RETURN "Final Decision: need revision".
    
    Please take a deep breath and think step by step. Provide an accurate and detailed review.
    Your review:
    """

for index, row in tqdm.tqdm(list(pairs.iterrows())):
    trait, condition = row['Trait'], row['Condition']
    nm_trait = normalize_trait(trait)
    nm_condition = normalize_trait(condition)
    trait_dir = os.path.join(args.data_dir, nm_trait)
    output_dir = os.path.join(args.output_dir, nm_trait)
    os.makedirs(output_dir, exist_ok=True)
    utils_code = "".join(open("utils.py", 'r').readlines())
    if condition in ['Age', 'Gender']:
        trait_cohort_id, _ = filter_and_rank_cohorts(os.path.join(trait_dir, 'cohort_info.json'), condition)
        trait_data_path = os.path.join(trait_dir, trait_cohort_id + '.csv')

        history = ""
        attempt_count = 0
        while True:
            attempt_count += 1
            # Call Azure OpenAI to generate the regression code using chat
            write_code_prompt = STEP_ONE_PROMPT.format(trait=trait, condition=condition,
                                                       trait_data_path=trait_data_path, output_dir=output_dir,
                                                       utils_code=utils_code)
            generated_code = call_openai_gpt_chat(
                history + write_code_prompt
            )

            # Process and execute the generated code
            code_text = parse_code(generated_code)
            result = subprocess.run(["python", "-c", code_text], capture_output=True, text=True)
            stdout = result.stdout
            stderr = result.stderr

            no_stderr = (len(stderr.strip()) == 0)

            review_code_prompt = write_code_prompt + "\n" + REVIEW_CODE_PROMPT_TEMPLATE.format(code=code_text,
                                                                                               stdout=stdout,
                                                                                               stderr=stderr)

            generated_review = call_openai_gpt_chat(
                review_code_prompt
            )
            approved_by_gpt = "approved" in generated_review.lower()

            print("*" * 25 + f" Code & results attempt {attempt_count}" + "*" * 25, flush=True)
            print(REVIEW_CODE_PROMPT_TEMPLATE.format(code=code_text, stdout=stdout, stderr=stderr))
            print("*" * 25 + f" Code & results attempt {attempt_count}" + "*" * 25, flush=True)
            print(generated_review, flush=True)

            if approved_by_gpt:
                print(f"Attempt {attempt_count} successful.")
                break

            if attempt_count > args.num_review:
                print(f"Exit due to attempt count {attempt_count} exceeded.")
                break

            history = \
                f"""
                History Code:
                {code_text}
                Review:
                {generated_review}

                Please carefully check the above review before implementing a new set of code for the next attempt.
                Please revise each point mentioned in the review and follow strictly the instruction to write a new set of code.

                """

    else:
        condition_dir = os.path.join(args.data_dir, nm_condition)
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

        rel = pd.read_csv(args.related_genes_path).drop(columns=["Unnamed: 0"])
        rel['Related_Genes'] = rel['Related_Genes'].apply(ast.literal_eval)
        t2g = pd.Series(rel['Related_Genes'].values, index=rel['Trait']).to_dict()  # the mapping from trait to genes
        related_genes = t2g[condition]
        regressors = get_gene_regressors(trait, trait_data, condition_data, related_genes)
        if regressors is None:
            print(f'No gene regressors for trait {trait} and condition {condition}')
            continue

        attempt_count = 0
        history = ""
        while True:
            attempt_count += 1
            write_code_prompt = STEP_TWO_PROMPT.format(trait=trait, condition=condition,
                                                       trait_data_path=trait_data_path,
                                                       condition_data_path=condition_data_path, output_dir=output_dir,
                                                       utils_code=utils_code, genes=regressors)
            generated_code = call_openai_gpt_chat(
                history + write_code_prompt
            )
            # Process and execute the generated code
            code_text = parse_code(generated_code)
            result = subprocess.run(["python", "-c", code_text], capture_output=True, text=True)
            stdout = result.stdout
            stderr = result.stderr

            no_stderr = (len(stderr.strip()) == 0)

            review_code_prompt = write_code_prompt + "\n" + REVIEW_CODE_PROMPT_TEMPLATE.format(code=code_text,
                                                                                               stdout=stdout,
                                                                                               stderr=stderr)

            generated_review = call_openai_gpt_chat(
                review_code_prompt
            )
            approved_by_gpt = "approved" in generated_review.lower()

            print("*" * 25 + f" Code & results attempt {attempt_count}" + "*" * 25, flush=True)
            print(REVIEW_CODE_PROMPT_TEMPLATE.format(code=code_text, stdout=stdout, stderr=stderr))
            print("*" * 25 + f" Code & results attempt {attempt_count}" + "*" * 25, flush=True)
            print(generated_review, flush=True)

            if approved_by_gpt:
                print(f"Attempt {attempt_count} successful.")
                break

            if attempt_count > 5:
                print(f"Exit due to attempt count {attempt_count} exceeded.")
                break

            history = \
                f"""
                History Code:
                {code_text}
                Review:
                {generated_review}

                Please carefully check the above review before implementing a new set of code for the next attempt.
                Please revise each point mentioned in the review and follow strictly the instruction to write a new set of code.

                """

    print(CODE_TEMPLATE.format(code=code_text, stdout=stdout, stderr=stderr))
    with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
        f.write(CODE_TEMPLATE.format(code=code_text, stdout=stdout, stderr=stderr))
    with open(os.path.join(output_dir, 'regression.py'), 'w') as f:
        f.write(code_text)