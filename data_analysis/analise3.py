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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from optuna.integration import LightGBMPruningCallback
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.samplers import TPESampler, CmaEsSampler


def objective(trial):
    
    param_dict = {
        "max_depth": trial.suggest_int('max_depth', 1, 25),
        "num_leaves": trial.suggest_int('num_leaves', 2, 100),
        "min_data_in_leaf": trial.suggest_int('min_data_in_leaf', 20, 500),
        "min_gain_to_split": trial.suggest_float('min_gain_to_split', 0.01, 40),
        "min_sum_hessian_in_leaf": trial.suggest_float('min_sum_hessian_in_leaf', 0.01, 100),
        "feature_fraction": trial.suggest_uniform('feature_fraction', 0.3, 1.0),
        "boosting_type": "dart",
        "learning_rate": trial.suggest_loguniform('learning_rate', 0.1, 0.2),
        "num_boost_round": trial.suggest_int('num_boost_round', 5, 1000),
        "feature_pre_filter": False
    }

    # Create and fit the LightGBM regressor
    gbm = lgb.LGBMRegressor(**param_dict)
    pruning_callback = LightGBMPruningCallback(trial, 'l2')  # 'l2' is equivalent to 'mse' in LightGBM
    gbm.fit(
        features_no_outliers, (target_no_outliers), 
        eval_set=[(test_features_no_outliers, test_target_no_outliers)],
        eval_metric='mse',
        callbacks=[pruning_callback, lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # Predict the results on the test set using the best iteration
    preds = gbm.predict(test_features_no_outliers, num_iteration=gbm.best_iteration_)

    # Calculate the Mean Squared Error on the test set
    mse = mean_squared_error(test_target_no_outliers, preds)
    return mse

# Set TPESampler as the sampler algorithm
sampler = TPESampler()

# Create a study object and specify the optimization direction (minimize)
study = optuna.create_study(direction='minimize', sampler=sampler, pruner=optuna.pruners.MedianPruner())

# Add stream handler of stdout to show the messages
#optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# Run the optimization with TPESampler as the sampler
study.optimize(objective, n_trials=7000, gc_after_trial=True)

# Write results to a file
with open('/home/rofarate/PIC-STELLARATOR/data_analysis/optuna_trials.txt', 'w') as f:
    f.write(f"Best Parameters: {study.best_params}\n")
    f.write(f"Best Score: {study.best_value}\n")

    # Optionally, write all trial results
    for trial in study.trials:
        f.write(f"Trial {trial.number}, Value: {trial.value}, Params: {trial.params}\n")


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
