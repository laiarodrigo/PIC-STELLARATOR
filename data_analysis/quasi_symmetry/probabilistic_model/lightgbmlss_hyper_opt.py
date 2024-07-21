import sqlite3
import pandas as pd
from lightgbmlss.model import LightGBMLSS
from lightgbmlss.distributions.Gaussian import Gaussian
from lightgbmlss.distributions.Mixture import Mixture
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import torch
import os

# Connect to the SQLite database
conn = sqlite3.connect('../../../data/nfp2/nfp2_combined.db')  # Adjust the path to your database file

# Query the database and load the data into a pandas DataFrame
query = "SELECT * FROM stellarators_combined"  # Adjust your query as needed
data_df = pd.read_sql_query(query, conn)

data_df_clean = data_df[data_df['convergence'] == 1]
data_df_clean = data_df_clean.dropna(subset=['quasisymmetry'])

X = data_df_clean[['rbc_1_0', 'rbc_m1_1', 'rbc_0_1', 'rbc_1_1', 'zbs_1_0', 'zbs_m1_1', 'zbs_0_1', 'zbs_1_1']]
Y = np.log(data_df_clean['quasisymmetry'])

features_no_outliers, test_features_no_outliers, target_no_outliers, test_target_no_outliers = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the Dataset with max_bin parameter specified
dtrain = lgb.Dataset(features_no_outliers, label=target_no_outliers.values)

lgblss = LightGBMLSS(
    Mixture(
        Gaussian(response_fn="exp"), 
        M = 9,
        tau=1.0,
        hessian_mode="individual", #grouped
    )
)

# Define the parameter dictionary without max_bin
param_dict = {
    "max_depth": ["int", {"low": 1, "high": 100, "log":False}],
    "num_leaves": ["int", {"low": 2, "high": 384, "log": True}],
    "min_data_in_leaf": ["int", {"low": 20, "high": 2000, "log": True}],
    "min_gain_to_split": ["float", {"low": 0.01, "high": 100, "log": True}],
    "min_sum_hessian_in_leaf": ["float", {"low": 0.01, "high": 100, "log": True}],
    #"subsample": ["float", {"low": 0.3, "high": 1.0, "log": False}],
    #"subsample_freq": ["int", {"low": 1, "high": 30, "log": True}],
    "feature_fraction": ["float", {"low": 0.3, "high": 1.0, "log": False}],
    "boosting_type": ["categorical", ["dart", "goss", "gbdt"]],
    "learning_rate": ["float", {"low": 0.01, "high": 0.5, "log": True}],
    "max_delta_step": ["float", {"low": 0, "high": 1, "log": False}],
    "feature_pre_filter": ["categorical", [False]],
    "boosting": ["categorical", ["dart", "gbdt"]]
}

# Set a seed for reproducibility
np.random.seed(123)

# Perform hyperparameter optimization
opt_param = lgblss.hyper_opt(
    param_dict,
    dtrain,
    num_boost_round=200,
    nfold=5,
    early_stopping_rounds=50,
    max_minutes=6000,
    n_trials=2,
    silence=False,
    seed=13,
    hp_seed=123
)

# Print the best parameters after optimization
print("Best Parameters:", opt_param)

# Assuming opt_param is defined somewhere in your code
opt_params = opt_param.copy()
n_rounds = opt_params["opt_rounds"]
del opt_params["opt_rounds"]

# Train the model with the optimized hyperparameters
lgblss.train(opt_params, dtrain, num_boost_round=n_rounds)

# # Evaluate the model on the test set
# test_predictions = lgblss.predict(test_features_no_outliers, pred_type="parameters")
# test_predictions_mean = test_predictions[:, 0]

# # Calculate the mean squared error (MSE) on the test set
# mse = np.mean((test_target_no_outliers - test_predictions_mean) ** 2)
# print(f"Test MSE: {mse}")

# Save the results
directory = '../quasi_symmetry/probabilistic_model'
if not os.path.exists(directory):
    os.makedirs(directory)

results_file = os.path.join(directory, 'study_results_lss.csv')
if os.path.exists(results_file):
    df_results = pd.read_csv(results_file)
else:
    df_results = pd.DataFrame(columns=['Study Name', 'Best Score', 'Parameters'])

study_name = 'stellarator_study_test'
new_row = pd.DataFrame({
    'Study Name': [study_name],
   # 'Best Score': [mse],
    'Parameters': [str(opt_params)]
})

dtypes = {'Study Name': str, 'Best Score': float, 'Parameters': str}
new_row = new_row.astype(dtypes)

df_results = pd.concat([df_results, new_row], ignore_index=True, sort=False)
df_results.to_csv(results_file, index=False)
