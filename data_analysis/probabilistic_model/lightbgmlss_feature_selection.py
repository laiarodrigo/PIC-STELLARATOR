import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbmlss.model import LightGBMLSS
from lightgbmlss.distributions.Weibull import *
import lightgbm as lgb
import numpy as np
import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load data from SQLite database
conn = sqlite3.connect('../data/nfp2/nfp2.db')
query = "SELECT * FROM stellarators"
data_df = pd.read_sql_query(query, conn)

# Step 2: Clean the data by dropping rows with missing values in the 'quasisymmetry' column
data_df_clean = data_df.dropna(subset=['quasisymmetry'])

# Step 3: Prepare features and target
X = data_df_clean[['rbc_1_0', 'rbc_m1_1', 'rbc_0_1', 'rbc_1_1','zbs_1_0', 'zbs_m1_1', 'zbs_0_1', 'zbs_1_1']] 
Y = data_df_clean['quasisymmetry']

# Step 4: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 5: Feature Importance of scale parameter
lgblss = LightGBMLSS(
    Weibull(stabilization="None", response_fn="exp", loss_fn="nll")
)
lgblss.plot(X_test,
            parameter="scale",
            plot_type="Feature_Importance")

# Step 6: Discard the 3 least important features
least_important_features = ['rbc_1_1', 'zbs_0_1', 'zbs_1_1']
X_train_subset = X_train.drop(columns=least_important_features)
X_test_subset = X_test.drop(columns=least_important_features)

# Step 7: Hyperparameter tuning
dtrain = lgb.Dataset(X_train_subset, label=Y_train.values, params={'max_bin': 500})
param_dict = {
    "max_depth": ["int", {"low": 1, "high": 25, "log": False}],
    "num_leaves": ["int", {"low": 2, "high": 100, "log": True}],
    "min_data_in_leaf": ["int", {"low": 20, "high": 500, "log": False}],
    "min_gain_to_split": ["float", {"low": 0.01, "high": 40, "log": True}],
    "min_sum_hessian_in_leaf": ["float", {"low": 0.01, "high": 100, "log": True}],
    "feature_fraction": ["float", {"low": 0.3, "high": 1.0, "log": False}],
    "boosting_type": ["categorical", ["dart", "goss", "gbdt"]],
    "learning_rate": ["float", {"low": 0.1, "high": 0.2, "log": True}],
    "num_boost_round": ["int", {"low": 5, "high": 1000, "log": True}],
    "feature_pre_filter": ["categorical", [False]],
    "boosting": ["categorical", ["dart"]]
}

np.random.seed(123)
opt_param = lgblss.hyper_opt(
    param_dict,
    dtrain,
    nfold=5,
    early_stopping_rounds=50,
    max_minutes=403,
    n_trials=7000,
    silence=False,
    seed=13,
    hp_seed=123
)

# Save best parameters to a file
with open('best_params.pkl', 'wb') as f:
    pickle.dump(opt_param, f)

# Later, when you want to use these parameters for training
with open('best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

# Step 8: Train model with optimized hyperparameters
opt_params = opt_param.copy()
n_rounds = opt_params["opt_rounds"]
del opt_params["opt_rounds"]
lgblss.train(opt_params, dtrain, num_boost_round=n_rounds)

# Step 9: Model evaluation
np.random.seed(123)
n_samples = len(X_test_subset)
pred_samples = lgblss.predict(
    X_test_subset,
    pred_type="samples",
    n_samples=n_samples,
    seed=123
)
pred_params = lgblss.predict(
    X_test_subset,
    pred_type="parameters"
)

df_predictions = pd.DataFrame({
    "Predicted": pred_samples.flatten(),
    "Type": "Predicted"
})
df_actual = pd.DataFrame({
    "Predicted": np.tile(Y_test, (len(pred_samples) // len(Y_test))),
    "Type": "Actual"
})

df_combined = pd.concat([df_predictions, df_actual])
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_combined, x="Predicted", hue="Type", fill=True)
plt.title('Density Plot of Predicted Outputs vs Actual Values')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend(title='Type')
plt.show()
