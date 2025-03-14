import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# LightGBM Model
best_params_manual = {
    'boosting_type': 'dart',
    'max_depth': 139,
    'num_leaves': 1684,
    'min_data_in_leaf': 51,
    'feature_fraction': 0.9631161865696998,
    'learning_rate': 0.12654909960762645,
    'num_iterations': 2707,
    'tree_learner': 'feature'
}

print('Initializing LightGBM model')
model = lgb.LGBMRegressor(**best_params_manual)
print('Fitting model')
model.fit(features_no_outliers, target_no_outliers)
print('Model fitted')

predictions_lgbm = model.predict(test_features_no_outliers)
print('Predicted with LightGBM')

# Calculate errors
errors_lgbm = np.abs(predictions_lgbm - test_target_no_outliers)

# Create a DataFrame
df_lgbm = pd.DataFrame({
    'quasiisodynamic': test_target_no_outliers,
    'errors_lgbm': errors_lgbm
})

# Save the DataFrame to a CSV file
df_lgbm.to_csv('errors_lgbm.csv', index=False)
print('Errors saved to errors_lgbm.csv')

print(test_target_no_outliers)
