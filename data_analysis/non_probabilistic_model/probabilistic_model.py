import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbmlss.distributions import *
from lightgbmlss.distributions.distribution_utils import DistributionClass
import numpy as np


# Load the true positive predictions from the CSV file
true_positive_predictions_file = 'true_positive_predictions_with_quasi.csv'
true_positive_data = pd.read_csv(true_positive_predictions_file)

# Separate features and target variable
X = true_positive_data[['rbc_1_0', 'rbc_m1_1', 'rbc_0_1', 'rbc_1_1','zbs_1_0', 'zbs_m1_1', 'zbs_0_1', 'zbs_1_1']] 
Y = true_positive_data['quasisymmetry']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

target_no_outliers_np = np.array(Y_train)

# Initialize DistributionClass
lgblss_dist_class = DistributionClass()

# Define candidate distributions
candidate_distributions = [Gaussian, StudentT, Gamma, Cauchy, LogNormal, Weibull, Gumbel, Laplace]

# Selecting the best distribution based on negative log-likelihood
dist_nll = lgblss_dist_class.dist_select(target=target_no_outliers_np, candidate_distributions=candidate_distributions, max_iter=50, plot=True, figure_size=(8, 4))
print(dist_nll)

# Plot the actual data density
plt.figure(figsize=(8, 4))
sns.kdeplot(target_no_outliers_np, bw_adjust=0.5, label='Actual Data Density')
plt.title('Density Function of Target Data')
plt.xlabel('Data')
plt.ylabel('Density')
plt.legend()
plt.show()

# Plot the distribution of Y and Y_test
plt.figure(figsize=(10, 6))
sns.kdeplot(Y, label='Y (quasisymmetry)')
sns.kdeplot(Y_test, label='Y_test (quasisymmetry)')
plt.title('Density Plot of Y and Y_test')
plt.xlabel('Quasisymmetry')
plt.ylabel('Density')
plt.legend()
plt.show()


from lightgbmlss.model import LightGBMLSS
from lightgbmlss.distributions.Weibull import *
import lightgbm as lgb
import numpy as np

# Select the first 1000 samples from X_train and their corresponding labels
X_train_subset = X_train
Y_train_subset = Y_train

# Create the Dataset with max_bin parameter specified
dtrain = lgb.Dataset(X_train_subset, label=Y_train_subset.values, params={'max_bin': 500})

# Initialize the LightGBMLSS model with the Weibull distribution
lgblss = LightGBMLSS(
    Weibull(stabilization="L2", response_fn="exp", loss_fn="nll")
)

# Define the parameter dictionary without max_bin
param_dict = {
    "max_depth": ["int", {"low": 1, "high": 25, "log": False}],
    "num_leaves": ["int", {"low": 2, "high": 100, "log": True}],
    "min_data_in_leaf": ["int", {"low": 20, "high": 500, "log": False}],
    "min_gain_to_split": ["float", {"low": 0.01, "high": 40, "log": True}],
    "min_sum_hessian_in_leaf": ["float", {"low": 0.01, "high": 100, "log": True}],
    #"subsample": ["float", {"low": 0.5, "high": 1.0, "log": False}],
    #"subsample_freq": ["int", {"low": 1, "high": 20, "log": False}],
    "feature_fraction": ["float", {"low": 0.3, "high": 1.0, "log": False}],
    "boosting_type": ["categorical", ["dart", "goss", "gbdt"]],
    "learning_rate": ["float", {"low": 0.1, "high": 0.2, "log": True}],
    # "lambda_l1" and "lambda_l2" are commented out as before
    "max_delta_step": ["float", {"low": 0, "high": 1, "log": False}],
    "num_boost_round": ["int", {"low": 5, "high": 1000, "log": True}],
    "feature_pre_filter": ["categorical", [False]],
}

# Set a seed for reproducibility
np.random.seed(123)

# Perform hyperparameter optimization
opt_param = lgblss.hyper_opt(
    param_dict,
    dtrain,
    #num_boost_round=30,
    nfold=5,
    early_stopping_rounds=50,
    max_minutes=403,
    n_trials=1,
    silence=True,
    seed=13,
    hp_seed=123
)

import numpy as np
import torch
from lightgbmlss.model import LightGBMLSS  # Ensure this import matches your actual usage

# Seed for reproducibility in numpy operations
np.random.seed(123)

# Assuming opt_param is defined somewhere in your code
opt_params = opt_param.copy()
n_rounds = opt_params['num_boost_round']
del opt_params['num_boost_round']

print('ewfre', opt_params)
print('nu, boost', n_rounds)

# Assuming dtrain is defined and is an appropriate dataset for training
# Train Model with optimized hyperparameters
lgblss.train(opt_params, dtrain, num_boost_round=n_rounds)

# Seed for reproducibility in torch operations
torch.manual_seed(123)

# Number of samples to draw from predicted distribution
n_samples = len(X_test)  # Use the number of rows in X_test as the number of samples

# Sample from predicted distribution
pred_samples = lgblss.predict(
    X_test,
    pred_type="samples",
    n_samples=n_samples,
    seed=123
)

# Return predicted distributional parameters
pred_params = lgblss.predict(
    X_test,
    pred_type="parameters"
)

# Define the path for saving the CSV file
csv_file_path = '/home/rofarate/PIC-STELLARATOR/data_analysis/non_probabilistic_model/best_parameters_prob_model.csv'

# Create a DataFrame from the opt_param dictionary
df_best_params = pd.DataFrame.from_dict(opt_param, orient='index', columns=['Value'])
df_best_params.index.name = 'Parameter'

# Save the DataFrame to a CSV file
df_best_params.to_csv(csv_file_path)

# Print confirmation message
print("Best parameters saved to:", csv_file_path)
