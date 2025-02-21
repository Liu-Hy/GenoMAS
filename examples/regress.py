import argparse
import traceback

from sklearn.linear_model import LogisticRegression, LinearRegression

from tools.statistics import *
from utils.utils import get_question_pairs

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--version', type=str, help='Specify the version.')

args = parser.parse_args()

task_info_file = './metadata/task_info.json'
all_pairs = get_question_pairs(task_info_file)

in_data_root = os.path.join('./output/preprocess', args.version)  # '/home/techt/Desktop/a4s/gold_subset'
output_root = os.path.join('./output/regress', 'test7')

for i, (trait, condition) in enumerate(all_pairs):
    print(f"Analyzing question {i}: trait {trait} and condition {condition}")
    try:
        if condition is None:
            print(f"Trait {trait} only")
            trait_data, _, _ = select_and_load_cohort(in_data_root, trait, is_two_step=False)
            trait_data = trait_data.drop(columns=['Age', 'Gender'], errors="ignore")

            Y = trait_data[trait].values
            X = trait_data.drop(columns=[trait]).values

            has_batch_effect = detect_batch_effect(X)
            if has_batch_effect:
                model_constructor = LMM
            else:
                model_constructor = Lasso

            param_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            best_config, best_performance = tune_hyperparameters(model_constructor, param_values, X, Y,
                                                                 trait_data.columns, trait, task_info_file,
                                                                 condition)
            model = ResidualizationRegressor(model_constructor, best_config)
            normalized_X, _ = normalize_data(X)
            model.fit(normalized_X, Y)

            var_names = trait_data.columns.tolist()
            significant_genes = interpret_result(model, var_names, trait, condition)
            save_result(significant_genes, best_performance, output_root, trait)

        else:
            if condition in ['Age', 'Gender']:
                trait_data, _, _ = select_and_load_cohort(in_data_root, trait, condition, is_two_step=False)
                redundant_col = 'Age' if condition == 'Gender' else 'Gender'
                if redundant_col in trait_data.columns:
                    trait_data = trait_data.drop(columns=[redundant_col])
            else:
                trait_data, condition_data, regressors = select_and_load_cohort(in_data_root, trait, condition, is_two_step=True, gene_info_path=task_info_file)
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

            Y = trait_data[trait].values
            Z = trait_data[condition].values
            X = trait_data.drop(columns=[trait, condition]).values

            has_batch_effect = detect_batch_effect(X)
            if has_batch_effect:
                model_constructor = LMM
            else:
                model_constructor = Lasso

            param_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            best_config, best_performance = tune_hyperparameters(model_constructor, param_values, X, Y, trait_data.columns, trait, task_info_file, condition, Z)

            model = ResidualizationRegressor(model_constructor, best_config)
            normalized_X, _ = normalize_data(X)
            normalized_Z, _ = normalize_data(Z)
            model.fit(normalized_X, Y, normalized_Z)

            var_names = trait_data.columns.tolist()
            significant_genes = interpret_result(model, var_names, trait, condition)
            save_result(significant_genes, best_performance, output_root, trait, condition)

    except Exception as e:
        print(f"Error processing pair {i}, for the trait '{trait}' and the condition '{condition}':\n{traceback.format_exc()}")
        continue