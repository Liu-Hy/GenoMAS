import pandas as pd
import os
import json
import ast
from utils.statistics import *

pairs = pd.read_csv("trait_condition_pairs.csv")

all_traits = pd.read_csv("all_traits.csv")["Trait"].tolist()
all_traits = [normalize_trait(at) for at in all_traits]

rel = pd.read_csv("trait_related_genes.csv")
rel['Related_Genes'] = rel['Related_Genes'].apply(ast.literal_eval)
t2g = pd.Series(rel['Related_Genes'].values, index=rel['Trait']).to_dict()

gene_info_path = './trait_related_genes.csv'
data_root = os.path.join('./output/preprocess', args.version)  # '/home/techt/Desktop/a4s/gold_subset'
output_root = os.path.join('./output/regress', args.version)

condition = None

for trait in all_traits:
    print(f"Trait {trait} only")
    output_dir = os.path.join(output_root, trait)
    os.makedirs(output_dir, exist_ok=True)
    try:
        trait_data, _, _ = select_and_load_cohort(data_root, trait, is_two_step=False)
        trait_data = trait_data.drop(columns=['Age', 'Gender'], errors="ignore")

        Y = trait_data[trait].values
        X = trait_data.drop(columns=[trait]).values

        has_batch_effect = detect_batch_effect(X)
        if has_batch_effect:
            model_constructor = LMM
        else:
            model_constructor = Lasso

        param_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        best_config, best_performance = tune_hyperparameters(model_constructor, param_values, X, Y, trait_data.columns, trait, gene_info_path, condition)
        model = ResidualizationRegressor(model_constructor, best_config)
        normalized_X, _ = normalize_data(X)
        model.fit(normalized_X, Y)

        var_names = trait_data.columns.tolist()
        significant_genes = interpret_result(model, var_names, trait, condition)
        save_result(significant_genes, best_performance, output_dir)

    except:
        print(f"Error processing trait {trait}")
        continue


for i, (index, row) in enumerate(pairs.iterrows()):
    try:
        trait, condition = row['Trait'], row['Condition']
        if trait != 'Adrenocortical_Cancer' or condition != 'Anxiety_disorder': continue
        output_dir = os.path.join(output_root, trait)
        os.makedirs(output_dir, exist_ok=True)

        if condition in ['Age', 'Gender']:
            trait_data, _, _ = select_and_load_cohort(data_root, trait, condition, is_two_step=False)
            redundant_col = 'Age' if condition == 'Gender' else 'Gender'
            if redundant_col in trait_data.columns:
                trait_data = trait_data.drop(columns=[redundant_col])
        else:
            trait_data, condition_data, regressors = select_and_load_cohort(data_root, trait, condition, is_two_step=True, gene_info_path=gene_info_path)
            trait_data = trait_data.drop(columns=['Age', 'Gender'], errors='ignore')
            if regressors is None:
                print(f'No gene regressors for trait {trait} and condition {condition}')
                continue

            print("Common gene regressors for condition and trait", regressors)
            X_condition = condition_data[regressors].values
            Y_condition = condition_data[condition].values

            condition_type = 'binary' if len(np.unique(Y_condition)) == 2 else 'continuous'

            if condition_type == 'binary':
                if X_condition.shape[1] > X_condition.shape[0]:
                    model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
                else:
                    model = LogisticRegression()
            else:
                if X_condition.shape[1] > X_condition.shape[0]:
                    model = Lasso()
                else:
                    model = LinearRegression()

            normalized_X_condition, _ = normalize_data(X_condition)
            model.fit(normalized_X_condition, Y_condition)

            regressors_in_trait = trait_data[regressors].values
            normalized_regressors_in_trait, _ = normalize_data(regressors_in_trait)
            if condition_type == 'binary':
                predicted_condition = model.predict_proba(normalized_regressors_in_trait)[:, 1]
            else:
                predicted_condition = model.predict(normalized_regressors_in_trait)

            trait_data[condition] = predicted_condition
            trait_data = trait_data.drop(columns=regressors)

        Y = trait_data[trait].values
        Z = trait_data[condition].values
        X = trait_data.drop(columns=[trait, condition]).values

        has_batch_effect = detect_batch_effect(X)
        if has_batch_effect:
            model_constructor = LMM
        else:
            model_constructor = Lasso

        param_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        best_config, best_performance = tune_hyperparameters(model_constructor, param_values, X, Y, trait_data.columns, trait, gene_info_path, condition, Z)

        model = ResidualizationRegressor(model_constructor, best_config)
        normalized_X, _ = normalize_data(X)
        normalized_Z, _ = normalize_data(Z)
        model.fit(normalized_X, Y, normalized_Z)

        var_names = trait_data.columns.tolist()
        significant_genes = interpret_result(model, var_names, trait, condition)
        save_result(significant_genes, best_performance, output_dir, condition)
    except Exception as e:
        print(f"Error processing row {i}, for the trait '{trait}' and the condition '{condition}'\n: {e}")
        continue