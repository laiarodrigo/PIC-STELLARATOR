import sqlite3
import pandas as pd

conn = sqlite3.connect('../data/nfp2/nfp2.db')  # Adjust the path to your database file

# Step 2 & 3: Query the database and load the data into a pandas DataFrame
query = "SELECT * FROM stellarators"  # Adjust your query as needed
data_df = pd.read_sql_query(query, conn)

from sklearn.model_selection import train_test_split, GridSearchCV

data_df_clean = data_df.dropna(subset=['quasisymmetry'])

X = data_df_clean[['rbc_1_0', 'rbc_m1_1', 'rbc_0_1', 'rbc_1_1','zbs_1_0', 'zbs_m1_1', 'zbs_0_1', 'zbs_1_1']] 
Y = data_df_clean[['quasisymmetry', 'quasiisodynamic', 'rotational_transform', 'inverse_aspect_ratio', 'mean_local_magnetic_shear', 'vacuum_magnetic_well', 'maximum_elongation', 'mirror_ratio']]

target = Y['quasisymmetry']
features = X

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print(features.shape)
print(target.shape)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming Y_train, X_train, Y_test, and X_test are pandas Series/DataFrames

# Calculate the IQR and bounds for outliers
q1 = Y_train.quantile(0.05)
q3 = Y_train.quantile(0.95) 
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Filter out the outliers from Y_train
target_no_outliers = Y_train[(Y_train >= lower_bound) & (Y_train <= upper_bound)]

# Check and filter X_train based on the indices of the filtered Y_train
features_no_outliers = X_train.loc[target_no_outliers.index.intersection(X_train.index)]

# For X_test and Y_test, you need to apply a similar filter or ensure the indices match
# Assuming Y_test should be filtered using the same bounds defined by Y_train
test_target_no_outliers = Y_test[(Y_test >= lower_bound) & (Y_test <= upper_bound)]
test_features_no_outliers = X_test.loc[test_target_no_outliers.index.intersection(X_test.index)]

import lightgbm as lgb
import numpy as np
import optuna
import logging
import sys
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from optuna.integration import LightGBMPruningCallback
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.samplers import TPESampler, CmaEsSampler


def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 1, 25),
        'num_leaves': trial.suggest_int('num_leaves', 2, 50),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 300),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_boost_round': trial.suggest_int('num_boost_round', 100, 1000)
    }

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    
    for train_index, valid_index in kf.split(X_train):
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        Y_train_fold, Y_valid_fold = Y_train.iloc[train_index], Y_train.iloc[valid_index]
        
        gbm = lgb.LGBMRegressor(**param)
        gbm.fit(X_train_fold, Y_train_fold, eval_set=[(X_valid_fold, Y_valid_fold)], eval_metric='mse',
                callbacks=[lgb.early_stopping(stopping_rounds=50)])
        preds = gbm.predict(X_valid_fold)
        mse_scores.append(mean_squared_error(Y_valid_fold, preds))
    
    return np.mean(mse_scores)

# Set TPESampler as the sampler algorithm
sampler = TPESampler()

# Create a study object and specify the optimization direction (minimize)
study = optuna.create_study(direction='minimize', sampler=sampler, pruner=optuna.pruners.MedianPruner())

# Add stream handler of stdout to show the messages
#optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# Run the optimization with TPESampler as the sampler
study.optimize(objective, n_trials=20, gc_after_trial=True)

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

# After fitting, you can use the model to predict or evaluate it further
# For example, to predict new values
predictions = model.predict(test_features_no_outliers)

mse = mean_squared_error(test_target_no_outliers, predictions)
mae = mean_absolute_error(test_target_no_outliers, predictions)
r2 = r2_score(test_target_no_outliers, predictions)

print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")
print(f"Test R^2: {r2}")

# predictions = predictions.flatten()  # ensuring predictions are flat
# actual_values = test_target_no_outliers.to_numpy()  # ensuring actual values are in a numpy array for consistent handling

# plt.figure(figsize=(10, 6))
# sns.kdeplot(predictions, fill=true, color='blue', label='predicted')
# sns.kdeplot(actual_values, fill=true, color='orange', label='actual')
# plt.title('density plot of predicted outputs vs actual values')
# plt.xlabel('values')
# plt.ylabel('density')
# plt.legend()
# plt.show()

import pandas as pd
import os
import tempfile

# Create a temporary directory
temp_dir = tempfile.mkdtemp(prefix='study_results_')

# Load or initialize the DataFrame to hold study results
results_file = os.path.join(temp_dir, 'study_results.csv')
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
