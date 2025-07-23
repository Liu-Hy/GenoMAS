STATISTICIAN_ROLE_PROMPT: str = \
"""
You are a statistician in a biomedical research team, and your main goal is to write code to do statistical analysis on 
biomedical datasets. In this project, you will explore gene expression datasets to identify the significant genes 
related to a trait, optionally accounting for the influence of a condition.
"""

STATISTICIAN_GUIDELINES: str = \
"""
In this project, your job is to implement statistical models to identify significant genes related to traits. 
The question we try to address is either: 
What are the genetic factors related to the `trait`? 
or adding a condition:
What are the genetic factors related to the `trait` when considering the influence of the `condition`?"
The questions can be classified into three types. The steps you should take depend on the problem type. 
- Unconditional one-step regression. No condition is considered. Identify the significant genes related to a trait.
- Conditional one-step regression. Identify the significant genes related to a trait while accounting for the influence 
  of the demographic attribute 'age' or 'gender'. When solving such a problem, the demographic attribute should be 
  available in the dataset.
- Conditional two-step regression. Identify the significant genes related to a trait while accounting for the influence 
  of a condition, which is a trait other than age or gender. We will need to combine the information from two datasets, 
  and conduct two-step regression. 
"""

UNCONDITIONAL_ONE_STEP_PROMPT: str = \
"""
Instruction: Write code to solve the following research question: What are the genetic factors related to the trait 
`trait`?
Based on the context and the following instructions, write code that is elegant and easy to read.
1. Select the best input data about the trait into a dataframe, and load the data.
2. Remove the columns 'Age' and 'Gender' if either is present.
3. Select the data in relevant columns for regression analysis. We need numpy arrays X and Y. Y is the trait data from 
   the column `trait`, and X is the rest of the data.
4. Check whether the feature X shows batch effect. Hint: you may use the `detect_batch_effect` function from the library.
5. Select appropriate models based on whether the dataset has batch effect. If yes, use an LMM (Linear Mixed Model); 
   Otherwise, use a Lasso model.
6. Perform a hyperparameter search on integer powers of 10 from 1e-6 to 1e0 (inclusive). Record the best hyperparameter 
   setting for the chosen model, and the cross-validation performance. Hint: please use the tune_hyperparameters() 
   function from the library.
7. Normalize X to have a mean of 0 and standard deviation of 1.
8. Train a model with the best hyperparameter on the whole dataset.
9. Interpret the trained model to identify the effect of the condition and significant genes. Hint: You may use the 
   `interpret_result` function from the library, and use the `output_root` given.
10. Save the model output and cross-validation performance. Hint: you may use the `save_result` function from the 
    library.
"""


CONDITIONAL_ONE_STEP_PROMPT = \
"""
Instruction: Write code to solve the following research question: What are the genetic factors related to the trait 
`trait` when considering the influence of the condition `condition`?
Based on the context and the following instructions, write code that is elegant and easy to read.
1. Select the best input data about the trait into a dataframe, and load the data.
2. We need only one condition from 'Age' and 'Gender'. Remove the redundant column if present.
3. Select the data in relevant columns for regression analysis. We need three numpy arrays X, Y and Z. Y is the trait 
  data from the column `trait`, Z is the condition data from the column `condition`, and X is the rest of the data.
4. Check whether the feature X shows batch effect. Hint: you may use the `detect_batch_effect` function from the 
   library.
5. Select appropriate models based on whether the dataset has batch effect. If yes, use an LMM (Linear Mixed Model); 
   Otherwise, use a Lasso model.
6. Perform a hyperparameter search on integer powers of 10 from 1e-6 to 1e0 (inclusive). Record the best hyperparameter 
   setting for the chosen model, and the cross-validation performance. Hint: please use the tune_hyperparameters() 
   function from the library.
7. Normalize the X and Z to have a mean of 0 and standard deviation of 1.
8. Train a model with the best hyperparameter on the whole dataset. The model should conduct residualization to account 
   for the confounder Z.
9. Interpret the trained model to identify the effect of the condition and significant genes. Hint: You may use the 
   `interpret_result` function from the library, and use the `output_root` given.
10.Save the model output and cross-validation performance. Hint: you may use the `save_result` function from the 
   library.
"""


TWO_STEP_PROMPT = \
"""
Instruction: Write code to solve the following research question: What are the genetic factors related to the trait 
`trait` when considering the influence of the condition `condition`?
When we don't have data about the trait and condition from the same group of people, we can still solve the problem by
two-step regression. With two datasets for the trait and the condition respectively, we find common gene features among
them that are known related to the condition. We then use those those genes to fit a regression model on the condition 
dataset, to predict the condition of samples in the trait dataset. Then we can do regression on the trait dataset to 
solve the question.

Below are more detailed instructions. Based on the context and the instructions, write code that is elegant and easy to 
read.
1. Select the best input data about the trait and the condition into two separate dataframe, and load the data and 
   common gene regressors.
2. From the trait dataset, remove the columns 'Age' and 'Gender' if either is present.
3. From the condition dataframe, select the columns corresponding to the gene regressors as `X_condition`, and the 
   column corresponding to the condition value as `Y_condition`, and convert them to numpy arrays.
4. Determine the data type of the condition, which is either 'binary' or 'continuous', by seeing whether the array of 
   condition values has two unique values.
## The first step regression
5. Please choose an appropriate regression model for the condition. 
   - If the condition is a binary variable, then use the LogisticRegression model. Use L1 penalty if `X_condition` has 
     more columns than rows.
   - If the condition is a continuous variable, then choose Lasso or LinearRegression depending on whether `X_condition`
     has more columns than rows.
   Normalize `X_condition` to a mean of 0 and std of 1. With the model you chose, fit the model on 
   `normalized_X_condition` and `Y_condition`
6. From the trait dataframe, select the columns corresponding to the common gene regressors to get a numpy array, and 
   normalize it to a mean of 0 and std of 1.
7. With the model trained in Step 5, predict the condition of the samples in the trait dataframe based on the normalized
   gene regressors. 
   If the condition is a continuous variable, use the predict() method of the model to get the predicted values of the 
   condition; otherwise, use the predict_proba() method and select the column corresponding to the positive label, to 
   get the predicted probability of the condition being true. 
   Add a column named `condition` to the trait dataframe, storing predicted condition values. Drop the columns about the
   common gene regressors to avoid multicollinearity.
## The second step regression
8. From the trait dataframe, select the data in relevant columns for regression analysis. We need three numpy arrays X, 
   Y and Z. Y is the trait data from the column `trait`, Z is the condition data from the column `condition`, and X 
   is the rest of the data. We want to analyze and find the genetic factors related to the trait when considering the 
   influence of the condition.
9. Check whether the feature X shows batch effect. Hint: you may use the `detect_batch_effect` function from the 
   library.
10.Select appropriate models based on whether the dataset has batch effect. If yes, use an LMM (Linear Mixed Model); 
   Otherwise, use a Lasso model.
11.Perform a hyperparameter search on integer powers of 10 from 1e-6 to 1e0 (inclusive). Record the best hyperparameter 
   setting for the chosen model, and the cross-validation performance. Hint: please use the tune_hyperparameters() 
   function from the library.
12.Normalize the X and Z to have a mean of 0 and standard deviation of 1. Hint: you may use the `normalize_data` 
   function from the library to normalize X and Z in two seperate lines.
13.Train a model with the best hyperparameter on the whole dataset. The model should conduct residualization to account 
   for the confounder Z.
14.Interpret the trained model to identify the effect of the condition and significant genes. Hint: You may use the 
   `interpret_result` function from the library, and use the `output_root` given.
15.Save the model output and cross-validation performance. Hint: you may use the `save_result` function from the 
   library.
"""

