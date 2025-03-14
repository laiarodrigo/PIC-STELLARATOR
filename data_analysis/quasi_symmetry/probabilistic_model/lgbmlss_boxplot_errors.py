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
conn = sqlite3.connect('../../../data/nfp2/nfp2_combined.db')
query = "SELECT * FROM stellarators_combined"
data_df = pd.read_sql_query(query, conn)
conn.close()

data_df_clean = data_df[data_df['convergence'] == 1]
data_df_clean = data_df_clean.dropna(subset=['quasiisodynamic'])

X = data_df_clean[['rbc_1_0', 'rbc_m1_1', 'rbc_0_1', 'rbc_1_1', 'zbs_1_0', 'zbs_m1_1', 'zbs_0_1', 'zbs_1_1']]
Y = np.log(data_df_clean['quasiisodynamic'])

features_no_outliers, test_features_no_outliers, target_no_outliers, test_target_no_outliers = train_test_split(X, Y, test_size=0.2, random_state=42)

print('Data loaded')

# LightGBMLSS Model
lgblss = LightGBMLSS(
    Mixture(
        Gaussian(response_fn="softplus"), 
        M=2,
        tau=1.0,
        hessian_mode="individual",
    )
)

opt_params = {
    "max_depth": 113,
    "num_leaves": 114,
    "min_data_in_leaf": 1537,
    "min_gain_to_split": 0.011698795932704837,
    "min_sum_hessian_in_leaf": 0.01321079065291914,
    "subsample": 0.8580461591753848,  # Original value retained
    "subsample_freq": 17,              # Original value retained
    "feature_fraction": 0.9786304627265173,
    "boosting_type": "gbdt",
    "learning_rate": 0.34583344744652667,
    "max_delta_step": 0.46888176823970534,
    "feature_pre_filter": False,
    "boosting": "gbdt",  # Updated from the provided params
    "opt_rounds": 200,   # Original value retained
    "device_type": "cpu",
    "num_class": 6,
    "metric": "None",
    "objective": "<bound method MixtureDistributionClass.objective_fn of <lightgbmlss.distributions.Mixture.Mixture object>>",  # Reference to the method, assuming the object stays consistent.
    "random_seed": 123,
    "verbose": -1
}

n_rounds = opt_params.pop("opt_rounds")
lgblss.train(opt_params, lgb.Dataset(features_no_outliers, label=target_no_outliers.values), num_boost_round=n_rounds)

print('LSS trained')

torch.manual_seed(123)

pred_samples = lgblss.predict(test_features_no_outliers, pred_type="samples", n_samples=10000, seed=123)
predictions_lgblss = pred_samples.mean(axis=1)
print(pred_samples.shape)
print(predictions_lgblss.shape)
print(test_features_no_outliers.shape)
print(test_target_no_outliers.shape)
# Calculate errors
errors_lgblss = np.zeros(shape=len(predictions_lgblss))
predictions_lgblss = predictions_lgblss.to_numpy()
test_target_no_outliers = np.array(test_target_no_outliers)
for i in range(predictions_lgblss.shape[0]):
    errors_lgblss[i] = predictions_lgblss[i] - test_target_no_outliers[i]

#errors_lgblss = predictions_lgblss - test_target_no_outliers
print(errors_lgblss.shape)

print('pred')
print(predictions_lgblss)



print('errors')
print(errors_lgblss)
# Create a DataFrame
df_lgblss = pd.DataFrame({
    'quasiisodynamic': test_target_no_outliers,
    'errors_lgblss': errors_lgblss
})

print(df_lgblss.shape)

# Check for NaN values
print("NaN values in DataFrame before saving:")
print(df_lgblss.isna().sum())

# Optionally, remove any rows with NaN values
df_lgblss_clean = df_lgblss.dropna()

# Save the cleaned DataFrame to a CSV file
df_lgblss_clean.to_csv('errors_lgblss.csv', index=False)
print('Cleaned errors saved to errors_lgblss_clean.csv')
