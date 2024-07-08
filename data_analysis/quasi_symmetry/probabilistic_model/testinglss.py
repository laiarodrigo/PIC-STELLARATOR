import sqlite3
import pandas as pd
from lightgbmlss.model import *
from lightgbmlss.distributions.Weibull import *
import lightgbm as lgb
import numpy as np
from lightgbmlss.distributions.Gaussian import *
from lightgbmlss.distributions.Mixture import *
from lightgbmlss.distributions.mixture_distribution_utils import MixtureDistributionClass

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
figure_size = (10,5)
import plotnine
from plotnine import *
plotnine.options.figure_size = figure_size

conn = sqlite3.connect('../../data/nfp2/nfp2.db')  # Adjust the path to your database file

# Step 2 & 3: Query the database and load the data into a pandas DataFrame
query = "SELECT * FROM stellarators"  # Adjust your query as needed
data_df = pd.read_sql_query(query, conn)

data_df_clean = data_df[data_df['convergence'] == 1]
data_df_clean = data_df_clean.dropna(subset=['quasisymmetry'])


X = data_df_clean[['rbc_1_0', 'rbc_m1_1', 'rbc_0_1', 'rbc_1_1','zbs_1_0', 'zbs_m1_1', 'zbs_0_1', 'zbs_1_1']] 
Y = np.log(data_df_clean['quasisymmetry'])

features_no_outliers, test_features_no_outliers, target_no_outliers, test_target_no_outliers = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the Dataset with max_bin parameter specified
dtrain = lgb.Dataset(features_no_outliers, label=target_no_outliers.values)

mix_dist_class = MixtureDistributionClass()

# # Define the candidate distributions
candidate_distributions = [
    Mixture(Gaussian(response_fn="softplus"), M=2),
    Mixture(Gaussian(response_fn="softplus"), M=3),
    Mixture(Gaussian(response_fn="softplus"), M=4),
    Mixture(Gaussian(response_fn="softplus"), M=5),
    Mixture(Gaussian(response_fn="softplus"), M=6),
    Mixture(Gaussian(response_fn="softplus"), M=7),
    Mixture(Gaussian(response_fn="softplus"), M=8),
    Mixture(Gaussian(response_fn="softplus"), M=9),
    Mixture(Gaussian(response_fn="softplus"), M=10),
]

# Convert target to numpy array if not already
target_no_outliers_np = np.array(target_no_outliers)

# Selecting the best distribution based on negative log-likelihood
dist_nll = mix_dist_class.dist_select(target=target_no_outliers_np, candidate_distributions=candidate_distributions, max_iter=50, plot=True, figure_size=(8, 5))

# Output the negative log-likelihoods
print(dist_nll)

# Plot the actual data density
plt.figure(figsize=(8, 4))
sns.kdeplot(target_no_outliers_np, bw_adjust=0.5, label='Actual Data Density')
plt.title('Density Function of Target Data')
plt.xlabel('Data')
plt.ylabel('Density')
plt.legend()
plt.show()

lgblss = LightGBMLSS(
    Mixture(
        Gaussian(response_fn="softplus", stabilization="L2"), 
        M = 2,
        tau=1.0,
        hessian_mode="individual",
    )
)

# Define the parameter dictionary without max_bin
param_dict = {
    "max_depth": ["int", {"low": 1, "high": 25, "log": False}],
    "num_leaves": ["int", {"low": 2, "high": 100, "log": True}],
    "min_data_in_leaf": ["int", {"low": 20, "high": 500, "log": False}],
    "min_gain_to_split": ["float", {"low": 0.01, "high": 40, "log": True}],
    "min_sum_hessian_in_leaf": ["float", {"low": 0.01, "high": 100, "log": True}],
    "subsample": ["float", {"low": 0.5, "high": 1.0, "log": False}],
    "subsample_freq": ["int", {"low": 1, "high": 20, "log": False}],
    "feature_fraction": ["float", {"low": 0.3, "high": 1.0, "log": False}],
    "boosting_type": ["categorical", ["dart", "goss", "gbdt"]],
    "learning_rate": ["float", {"low": 0.1, "high": 0.2, "log": True}],
    # "lambda_l1" and "lambda_l2" are commented out as before
    "max_delta_step": ["float", {"low": 0, "high": 1, "log": False}],
    "feature_pre_filter": ["categorical", [False]],
    "boosting": ["categorical", ["dart"]]
}

# Set a seed for reproducibility
np.random.seed(123)

# Perform hyperparameter optimization
opt_param = lgblss.hyper_opt(
    param_dict,
    dtrain,
    num_boost_round=30,
    nfold=5,
    early_stopping_rounds=100,
    max_minutes=60,
    n_trials=300,
    silence=False,
    seed=13,
    hp_seed=123
)

from lightgbmlss.model import LightGBMLSS  # Ensure this import matches your actual usage

# Seed for reproducibility in numpy operations
np.random.seed(123)

# Assuming opt_param is defined somewhere in your code
opt_params = opt_param.copy()
n_rounds = opt_params["opt_rounds"]
del opt_params["opt_rounds"]

# Assuming dtrain is defined and is an appropriate dataset for training
# Train Model with optimized hyperparameters
lgblss.train(opt_params, dtrain, num_boost_round=n_rounds)

# Seed for reproducibility in torch operations
torch.manual_seed(123)

# Number of samples to draw from predicted distribution
n_samples = len(test_target_no_outliers)  # Use the number of rows in X_test as the number of samples

# Quantiles to calculate from predicted distribution
quant_sel = [0.25, 0.75]

# Sample from predicted distribution
pred_samples = lgblss.predict(
    test_features_no_outliers,
    pred_type="samples",
    n_samples=n_samples,
    seed=123
)

# Calculate quantiles from predicted distribution
pred_quantiles = lgblss.predict(
    test_features_no_outliers,
    pred_type="quantiles",
    n_samples=n_samples,
    quantiles=quant_sel
)

# Return predicted distributional parameters
pred_params = lgblss.predict(
    test_features_no_outliers,
    pred_type="parameters"
)

lgblss.plot(test_features_no_outliers,
            parameter="scale",
            plot_type="Feature_Importance")

import os

# Define the directory path
directory = '/home/rofarate/PIC-STELLARATOR/data_analysis/probabilistic_model'

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Load or initialize the DataFrame to hold study results
results_file = os.path.join(directory, 'study_results_lss.csv')
if os.path.exists(results_file):
    df_results = pd.read_csv(results_file)
else:
    df_results = pd.DataFrame(columns=['Study Name', 'Best Score', 'Parameters'])

# Create a new DataFrame for the new row
study_name = 'stellarator_study_test'
new_row = pd.DataFrame({
    'Study Name': [study_name],
    'Best Score': [np.min(lgblss.evals_result_['valid_0']['l2'])],  # Assuming 'l2' is the evaluation metric used
    'Parameters': [str(opt_params)]
})

# Ensure consistent data types for the columns
dtypes = {'Study Name': str, 'Best Score': float, 'Parameters': str}
new_row = new_row.astype(dtypes)

# Concatenate the new row to the existing DataFrame
df_results = pd.concat([df_results, new_row], ignore_index=True, sort=False)

# Save the updated results
df_results.to_csv(results_file, index=False)