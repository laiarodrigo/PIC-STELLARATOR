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
print("Data loaded successfully")

from lightgbmlss.model import LightGBMLSS  # Ensure this import matches your actual usage

lgblss = LightGBMLSS(
    Mixture(
        Gaussian(response_fn="softplus", stabilization="L2"), 
        M = 9,
        tau=1.0,
        hessian_mode="individual",
    )
)

opt_params = {
    "max_depth": 12,
    "num_leaves": 28,
    "min_data_in_leaf": 1301,
    "min_gain_to_split": 0.4319605082782415,
    "min_sum_hessian_in_leaf": 0.30164458790087806,
    "subsample": 0.8580461591753848,
    "subsample_freq": 17,
    "feature_fraction": 0.9625238100384692,
    "boosting_type": "goss",
    "learning_rate": 0.32940258973561587,
    "max_delta_step": 0.3796994086587884,
    "feature_pre_filter": False,
    "boosting": "dart",  # Assuming this value remains the same
    "opt_rounds": 300  # Assuming this value remains the same
}

n_rounds = opt_params["opt_rounds"]
del opt_params["opt_rounds"]

lgblss.train(opt_params, dtrain, num_boost_round=n_rounds)

print("Model trained successfully")

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