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

df_predictions = pd.DataFrame({
    "Predicted": predictions.flatten(),  # Flatten in case the predictions are in a 2D array
    "Type": "Predicted"
})
df_actual = pd.DataFrame({
    "Predicted": np.tile(Y_test, (len(predictions) // len(Y_test))),
    "Type": "Actual"
})

import matplotlib.pyplot as plt
import seaborn as sns

# Combine and plot
df_combined = pd.concat([df_predictions, df_actual])
plt.figure(figsize=(10, 6))
ax = sns.kdeplot(data=df_combined, x="Predicted", hue="Type", fill=True)
plt.title('Density Plot of Predicted Outputs vs Actual Values')
plt.xlabel('Values')
plt.ylabel('Density')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=["Predicted", "Actual"], title="Type")
plt.show()

import pandas as pd
import os

# Dataframe to hold study results
if not os.path.exists('/home/rofarate/PIC-STELLARATOR/data_analysis/study_results.csv'):
    df_results = pd.DataFrame(columns=['Study Name', 'Best Score', 'Parameters'])
else:
    df_results = pd.read_csv('/home/rofarate/PIC-STELLARATOR/data_analysis/study_results.csv')

# Append new study results
study_name = 'stellarator_study_test'
new_row = {'Study Name': study_name, 'Best Score': study.best_value, 'Parameters': str(study.best_params)}
df_results = df_results.append(new_row, ignore_index=True)

# Save the updated results
df_results.to_csv('/home/rofarate/PIC-STELLARATOR/data_analysis/study_results.csv', index=False)
