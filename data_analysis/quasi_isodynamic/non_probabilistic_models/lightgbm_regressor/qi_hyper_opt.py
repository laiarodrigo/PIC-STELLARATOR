import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import sqlite3

# Load the true positive predictions from the CSV file
conn = sqlite3.connect('../../../../data/nfp2/nfp2_combined.db')  # Adjust the path to your database file

# Step 2 & 3: Query the database and load the data into a pandas DataFrame
query = "SELECT * FROM stellarators_combined"  # Adjust your query as needed
data_df = pd.read_sql_query(query, conn)

data_df_clean = data_df[data_df['convergence'] == 1]
data_df_clean = data_df_clean.dropna(subset=['quasiisodynamic'])

X = data_df_clean[['rbc_1_0', 'rbc_m1_1', 'rbc_0_1', 'rbc_1_1','zbs_1_0', 'zbs_m1_1', 'zbs_0_1', 'zbs_1_1']] 
Y = np.log(data_df_clean['quasiisodynamic'])

# Split the data into training and testing sets
features_no_outliers, test_features_no_outliers, target_no_outliers, test_target_no_outliers = train_test_split(X, Y, test_size=0.2, random_state=42)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import optuna
import math
import sys
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from optuna.integration import LightGBMPruningCallback
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.samplers import TPESampler, CmaEsSampler

print('starting objective function')

def objective(trial):
    param = {
        'objective': 'regression',
        'metric': ['mse'], 
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
        'max_depth': trial.suggest_int('max_depth', 1, 150),
        'num_leaves': trial.suggest_int('num_leaves', 2, 5000, log=False),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 7000),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=False),
        'num_iterations': trial.suggest_int('num_iterations', 50, 3000, log = False),
        'data_sample_strategy': trial.suggest_categorical('data_sample_strategy', ['bagging', 'goss']),
        'max_bins': trial.suggest_int('max_bins', 5, 10000),
        'linear_tree': True,  # Enable linear tree
        #'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),  # Add min_child_weight parameter
        'tree_learner': trial.suggest_categorical('tree_learner', ['voting', 'data', 'feature', 'serial']),
        'force_row_wise': True,  # Ensure row-wise growth to support monotonic constraints
        #'device': 'gpu'  # Use GPU
    }

    # Train the model on the entire training set
    gbm = lgb.LGBMRegressor(**param,
                            #monotone_constraints=monotone_constraints
                            )
    gbm.fit(features_no_outliers, target_no_outliers)

    # Predict on the test set
    preds = gbm.predict(test_features_no_outliers)

    # Calculate metrics
    mse = mean_squared_error(test_target_no_outliers, preds)
    mae = mean_absolute_error(test_target_no_outliers, preds)
    r2 = r2_score(test_target_no_outliers, preds)

    # Return the mean squared error
    return mse

# Set TPESampler as the sampler algorithm
sampler = TPESampler()

# Create a study object and specify the optimization direction (minimize)
study = optuna.create_study(direction='minimize', sampler=sampler, pruner=optuna.pruners.MedianPruner())

# Run the optimization with TPESampler as the sampler
study.optimize(objective, n_trials=300, gc_after_trial=True)

# Access the best parameters and best score
best_params = study.best_params
best_score = study.best_value

print("Best Parameters:", best_params)
print("Best Score:", best_score)

print('Best trial:', study.best_trial)
print('Best value:', study.best_value)
print('Best parameters:', study.best_params)

import lightgbm as lgb

# Assuming study.best_params already includes the best hyperparameters from your Optuna study for a regression problem
model = lgb.LGBMRegressor(**study.best_params)

# Assuming features_no_outliers and target_no_outliers are your feature matrix and target vector, respectively
model.fit(features_no_outliers, target_no_outliers)

print('fitted')

# After fitting, you can use the model to predict or evaluate it further
# For example, to predict new values
predictions = model.predict(test_features_no_outliers)

mse = mean_squared_error(test_target_no_outliers, predictions)
mae = mean_absolute_error(test_target_no_outliers, predictions)
r2 = r2_score(test_target_no_outliers, predictions)

print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")
print(f"Test R^2: {r2}")

lgb.plot_importance(model, max_num_features=10)
plt.title('Feature Importance')
plt.show()

import pandas as pd
import os
import tempfile

# Define the directory path
#directory = 'data_analysis/quasi_isodynamic/non_probabilistic_models/lightgbm_regressor'

# Create the directory if it doesn't exist
# if not os.path.exists(directory):
#     os.makedirs(directory)

# Load or initialize the DataFrame to hold study results
results_file = os.path.join('study_results.csv')
if os.path.exists(results_file):
    df_results = pd.read_csv(results_file)
else:
    df_results = pd.DataFrame(columns=['Study Name', 'Best Score', 'Parameters'])

# Create a new DataFrame for the new row
study_name = 'stellarator_study_test'
new_row = pd.DataFrame({
    'Study Name': [study_name],
    'Best Score': [study.best_value],
    'Parameters': [str(study.best_params)]
})

# Ensure consistent data types for the columns
dtypes = {'Study Name': str, 'Best Score': float, 'Parameters': str}
new_row = new_row.astype(dtypes)

# Exclude empty or all-NA columns before concatenation
df_results = df_results.dropna(axis=1, how='all')

# Concatenate the new row to the existing DataFrame
df_results = pd.concat([df_results, new_row], ignore_index=True, sort=False)

# Save the updated results
df_results.to_csv(results_file, index=False)