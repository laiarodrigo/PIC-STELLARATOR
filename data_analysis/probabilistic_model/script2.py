import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from lightgbmlss.model import LightGBMLSS
from lightgbmlss.distributions.Mixture import Mixture
from lightgbmlss.distributions.Gaussian import Gaussian
import lightgbm as lgb

# Load data
conn = sqlite3.connect('../../data/nfp2/nfp2_combined.db')
query = "SELECT * FROM stellarators"
data_df = pd.read_sql_query(query, conn)
conn.close()

data_df_clean = data_df[data_df['convergence'] == 1]
data_df_clean = data_df_clean.dropna(subset=['quasisymmetry'])

X = data_df_clean[['rbc_1_0', 'rbc_m1_1', 'rbc_0_1', 'rbc_1_1', 'zbs_1_0', 'zbs_m1_1', 'zbs_0_1', 'zbs_1_1']]
Y = np.log(data_df_clean['quasisymmetry'])

features_no_outliers, test_features_no_outliers, target_no_outliers, test_target_no_outliers = train_test_split(X, Y, test_size=0.2, random_state=42)

print('Data loaded')

# LightGBMLSS Model
lgblss = LightGBMLSS(
    Mixture(
        Gaussian(response_fn="softplus", stabilization="L2"), 
        M=9,
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
    "boosting": "dart",
    "opt_rounds": 200
}

n_rounds = opt_params.pop("opt_rounds")
lgblss.train(opt_params, lgb.Dataset(features_no_outliers, label=target_no_outliers.values), num_boost_round=n_rounds)

print('LSS trained')

torch.manual_seed(123)

pred_samples = lgblss.predict(test_features_no_outliers, pred_type="samples", n_samples=10000, seed=123)
predictions_lgblss = pred_samples.mean(axis=1)

# Calculate errors
errors_lgblss = np.abs(predictions_lgblss - test_target_no_outliers)

# Create a DataFrame
df_lgblss = pd.DataFrame({
    'quasisymmetry': test_target_no_outliers,
    'errors_lgblss': errors_lgblss
})

# Save the DataFrame to a CSV file
df_lgblss.to_csv('errors_lgblss.csv', index=False)
print('Errors saved to errors_lgblss.csv')
