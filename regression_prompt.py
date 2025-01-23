import os
from openai import AzureOpenAI
import pandas as pd
import numpy as np
import json
import ast
import re
import subprocess
import tqdm

pairs_df = pd.read_csv("trait_condition_pairs.csv")
pair_rows = list(pairs_df.iterrows())
all_pairs = []
seen_traits = set()
for i, row in pair_rows:
    trait, condition = row['Trait'], row['Condition']
    if trait not in seen_traits:
        seen_traits.add(trait)
        all_pairs.append((trait, None))
    all_pairs.append((trait, condition))

gene_info_path = './trait_related_genes.csv'
data_root = '/home/techt/Desktop/a4s/gold_subset'
output_root = './output_agent'

# Azure OpenAI Client setup
client = AzureOpenAI(
    api_key="57983a2e88fa4d6b81205a8d55d9bd46",
    api_version="2023-10-01-preview",
    azure_endpoint="https://haoyang2.openai.azure.com/"
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

INSTRUCTION_HEAD_TEMPLATE: str = \
    """
    Role: You are a statistician in a biomedical research team, and your main goal is to write code to do statistical 
    analysis on biomedical datasets.
    In this project, you will explore gene expression datasets to identify the significant genes related to a trait, 
    optionally controlling for a condition.

    Tools: In "utils.statistics", there are lots of well-developed helper functions for this project. Please import and 
    use them when possible. Hereafter I will call it "the library". Below is the source code.
    {utils_code}

    Background:
    1. All input data are stored in the directory: '{data_root}'.
    2. The output should be saved to the directory '{output_root}', under a subdirectory named after the trait.
    3. External knowledge about genes related to each trait is available in a file '{gene_info_path}'.

    NOTICE1: Please import all the functions in 'utils.statistics' at the beginning of the code, and feel free to use '*' in the import statement.
    NOTICE2: The overall preprocessing requires multiple code snippets and each code snippet is based on the execution results of the last code snippet. 
    Consequently, the instruction will be divided into multiple STEPS, each STEP requires you to write a code snippet, then the execution result will be given to you for either revision of the current STEP or go to the next STEP.

    Based on the context, write code to follow the instructions.
    """

UNCONDITIONAL_ONE_STEP_PROMPT = """
Instruction: Write code to solve the following research question: What are the genetic factors related to the trait 
'{trait}'?
Based on the context and the following instructions, write code that is elegant and easy to read.
1. Select the best input data about the trait into a dataframe, and load the data.
2. Remove the columns 'Age' and 'Gender' if either is present.
3. Select the data in relevant columns for regression analysis. We need numpy arrays X and Y. Y is the trait data from the column '{trait}', and X is the rest of the data.
4. Check whether the feature X shows batch effect. Hint: you may use the 'detect_batch_effect' function from the library.
5. Select appropriate models based on whether the dataset has batch effect. If yes, use an LMM (Linear Mixed Model); 
   Otherwise, use a Lasso model.
6. Perform a hyperparameter search on integer powers of 10 from 1e-6 to 1e0 (inclusive). Record the best hyperparameter setting for the chosen model, and the cross-validation performance. Hint: please use the tune_hyperparameters() function from the library.
7. Normalize X to have a mean of 0 and standard deviation of 1.
8. Train a model with the best hyperparameter on the whole dataset.
9. Interpret the trained model to identify the effect of the condition and significant genes. Hint: You may use the 'interpret_result' function from the library, and use the output_dir given.
10. Save the model output and cross-validation performance. Hint: you may use the 'save_result' function from the library

Return ```python your_code_here ``` with NO other texts. your_code_here is a placeholder.
your code:
        """

CONDITIONAL_ONE_STEP_PROMPT = """
Instruction: Write code to solve the following research question: What are the genetic factors related to the trait 
'{trait}' when considering the influence of the condition '{condition}'?
Based on the context and the following instructions, write code that is elegant and easy to read.
1. Select the best input data about the trait into a dataframe, and load the data.
2. We need only one condition from 'Age' and 'Gender'. Remove the redundant column if present.
3. Select the data in relevant columns for regression analysis. We need three numpy arrays X, Y and Z. Y is the trait data from the column '{trait}', Z is the condition data from the column '{condition}', and X is the rest of the data.
4. Check whether the feature X shows batch effect. Hint: you may use the 'detect_batch_effect' function from the library.
5. Select appropriate models based on whether the dataset has batch effect. If yes, use an LMM (Linear Mixed Model); 
   Otherwise, use a Lasso model.
6. Perform a hyperparameter search on integer powers of 10 from 1e-6 to 1e0 (inclusive). Record the best hyperparameter setting for the chosen model, and the cross-validation performance. Hint: please use the tune_hyperparameters() function from the library.
7. Normalize the X and Z to have a mean of 0 and standard deviation of 1.
8. Train a model with the best hyperparameter on the whole dataset. The model should conduct residualization to account for the confounder Z.
9. Interpret the trained model to identify the effect of the condition and significant genes. Hint: You may use the 'interpret_result' function from the library, and use the output_dir given.
10. Save the model output and cross-validation performance. Hint: you may use the 'save_result' function from the library

Return ```python your_code_here ``` with NO other texts. your_code_here is a placeholder.
your code:
        """

TWO_STEP_PROMPT = """
Instruction: Write code to solve the following research question: What are the genetic factors related to the trait 
'{trait}' when considering the influence of the condition '{condition}'?
When we don't have data about the trait and condition from the same group of people, we can still solve the problem by
two-step regression. With two datasets for the trait and the condition respectively, we find common gene features among
them that are known related to the condition. We then use those those genes to fit a regression model on the condition 
dataset, to predict the condition of samples in the trait dataset. Then we can do regression on the trait dataset to 
solve the question.

Below are more detailed instructions. Based on the context and the instructions, write code that is elegant and easy to read.
1. Select the best input data about the trait and the condition into two separate dataframe, and load the data and common gene regressors.
2. From the trait dataset, remove the columns 'Age' and 'Gender' if either is present.
3. From the condition dataframe, select the columns corresponding to the gene regressors as 'X_condition', and the column corresponding to the condition value as 'Y_condition', and convert them to numpy arrays.
4. Determine the data type of the condition, which is either 'binary' or 'continuous', by seeing whether the array of condition values has two unique values.
## The first step regression
5. Please choose an appropriate regression model for the condition. 
   - If the condition is a binary variable, then use the LogisticRegression model. Use L1 penalty if 'X_condition' has more columns than rows.
   - If the condition is a continuous variable, then choose Lasso or LinearRegression depending on whether 'X_condition' has more columns than rows.
   Normalize 'X_condition' to a mean of 0 and std of 1. With the model you chose, fit the model on 'normalized_X_condition' and 'Y_condition'
6. From the trait dataframe, select the columns corresponding to the common gene regressors to get a numpy array, and normalize it to a mean of 0 and std of 1.
7. With the model trained in Step 5, predict the condition of the samples in the trait dataframe based on the normalized gene regressors. 
  If the condition is a continuous variable, use the predict() method of the model to get the predicted values of the condition; otherwise, use the predict_proba() 
  method and select the column corresponding to the positive label, to get the predicted probability of the condition being true. 
  Add a column named {condition} to the trait dataframe, storing predicted condition values. Drop the columns about the common gene regressors.
## The second step regression
8. From the trait dataframe, select the data in relevant columns for regression analysis. We need three numpy arrays X, Y and Z. Y is the trait data from the column '{trait}', Z is the condition data from the column '{condition}', and X is the rest of the data. We want to analyze and find the genetic factors related to the trait when considering the influence of the condition.
9. Check whether the feature X shows batch effect. Hint: you may use the 'detect_batch_effect' function from the library.
10. Select appropriate models based on whether the dataset has batch effect. If yes, use an LMM (Linear Mixed Model); 
   Otherwise, use a Lasso model.
11. Perform a hyperparameter search on integer powers of 10 from 1e-6 to 1e0 (inclusive). Record the best hyperparameter setting for the chosen model, and the cross-validation performance. Hint: please use the tune_hyperparameters() function from the library.
12. Normalize the X and Z to have a mean of 0 and standard deviation of 1. Hint: you may use the 'normalize_data' function from the library to normalize X and Z in two seperate lines.
13. Train a model with the best hyperparameter on the whole dataset. The model should conduct residualization to account for the confounder Z.
14. Interpret the trained model to identify the effect of the condition and significant genes. Hint: You may use the 'interpret_result' function from the library, and use the output_dir given.
15. Save the model output and cross-validation performance. Hint: you may use the 'save_result' function from the library

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
utils_code = "".join(open("utils/statistics.py", 'r').readlines())

for index, pair in enumerate(all_pairs):
    trait, condition = pair
    # if trait != 'Adrenocortical_Cancer' or condition != 'Anxiety_disorder': continue
    instruction_head = INSTRUCTION_HEAD_TEMPLATE.format(utils_code=utils_code, data_root=data_root, output_root=output_root,
                                                gene_info_path=gene_info_path)
    if condition is None or condition.lower == "none":
        regression_prompt = UNCONDITIONAL_ONE_STEP_PROMPT
    elif condition in ['Age', 'Gender']:
        regression_prompt = CONDITIONAL_ONE_STEP_PROMPT
    else:
        regression_prompt = TWO_STEP_PROMPT

    history = ""
    attempt_count = 0
    while True:
        attempt_count += 1
        # Call Azure OpenAI to generate the regression code using chat
        write_code_prompt = instruction_head + regression_prompt.format(trait=trait, condition=condition,
                                                     )
        generated_code = call_openai_gpt_chat(
            history + write_code_prompt
        )

        # Process and execute the generated code
        code_text = parse_code(generated_code)
        print(code_text)
        result = subprocess.run(["python", "-c", code_text], capture_output=True, text=True)
        stdout = result.stdout
        stderr = result.stderr

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

        if attempt_count > 3:
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
    with open(os.path.join(output_root, trait, 'log.txt'), 'w') as f:
        f.write(CODE_TEMPLATE.format(code=code_text, stdout=stdout, stderr=stderr))
    with open(os.path.join(output_root, trait, 'regression.py'), 'w') as f:
        f.write(code_text)
