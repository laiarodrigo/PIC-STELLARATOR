import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import optuna
import numpy as np

# Train the binary classification model
conn = sqlite3.connect('../../../../data/nfp2/nfp2_combined.db')  # Adjust the path to your database file

# Step 2 & 3: Query the database and load the data into a pandas DataFrame
query = "SELECT * FROM stellarators_combined"  # Adjust your query as needed
data_df = pd.read_sql_query(query, conn)

# Extract features and target variable
X = data_df[['rbc_1_0', 'rbc_m1_1', 'rbc_0_1', 'rbc_1_1','zbs_1_0', 'zbs_m1_1', 'zbs_0_1', 'zbs_1_1']] 
Y = data_df['convergence']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

total_stel = len(data_df)
converged_stel = data_df['convergence'].sum()
not_converged_stel = total_stel - converged_stel

print("Total stel:", total_stel)
print("Converged stel:", converged_stel)
print("Not converged stel:", not_converged_stel)

print(X_train.shape)
print(Y_train.shape)

# Define the objective function for Optuna
def objective(trial):
    # Define the parameters to be optimized
    params = {
        'objective': 'binary',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
        'max_depth': trial.suggest_int('max_depth', 1, 70),
        'num_leaves': trial.suggest_int('num_leaves', 2, 1000, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 5000),
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 100, 300000, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 3000)  # Adjust the range as needed

    }

    # Additional parameters for Random Forest boosting type
    if params['boosting_type'] == 'rf':
        # Ensure bagging frequency is set to a non-zero value
        params['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 30)
        # Ensure bagging fraction is set to a valid range
        params['bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0.1, 1.0)
        # Ensure feature fraction is set to a valid range
        params['feature_fraction'] = trial.suggest_uniform('feature_fraction', 0.1, 1.0)

    # Train the model
    gbm = lgb.LGBMClassifier(**params)
    gbm.fit(X_train, Y_train)

    # Predict on the test set
    preds = gbm.predict(X_test)

    # Calculate accuracy
    accuracy = (preds == Y_test).mean()

    # Return the negated accuracy since Optuna aims to minimize the objective function
    return 1 - accuracy

# Set TPESampler as the sampler algorithm
sampler = optuna.samplers.TPESampler(seed=42)

# Create a study object and specify the optimization direction (minimize)
study = optuna.create_study(direction='minimize', sampler=sampler)

# Run the optimization
study.optimize(objective, n_trials=100)

# Access the best parameters and best score
best_params = study.best_params
best_score = 1 - study.best_value

print("Best Parameters:", best_params)
print("Best Score (Accuracy):", best_score)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Train the model with the best parameters found by Optuna
best_gbm = lgb.LGBMClassifier(**best_params)
best_gbm.fit(X_train, Y_train)

# Predict on the test set
preds = best_gbm.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(Y_test, preds)
precision = precision_score(Y_test, preds)
recall = recall_score(Y_test, preds)
f1 = f1_score(Y_test, preds)
roc_auc = roc_auc_score(Y_test, preds)
conf_matrix = confusion_matrix(Y_test, preds)

# Print evaluation metrics
print("Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:")
print(conf_matrix)

feature_importance = best_gbm.feature_importances_

feature_importance_dict = dict(zip(X_train.columns, feature_importance))

sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("Feature Importance:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")
    
import pandas as pd
import os
import tempfile

# Define the directory path
directory = '../quasi_symmetry/non_probabilistic_model/lightgbm_regressor'

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Load or initialize the DataFrame to hold study results
results_file = os.path.join(directory, 'study_results_class_model.csv')
if os.path.exists(results_file):
    df_results = pd.read_csv(results_file)
else:
    df_results = pd.DataFrame(columns=['Study Name', 'Best Score', 'Parameters'])

# Create a new DataFrame for the new row
study_name = 'stellarator_study_test'
new_row = pd.DataFrame({
    'Study Name': [study_name],
    'Best Score': [study.best_value],
    'Parameters': [str(study.best_params)]
})

# Ensure consistent data types for the columns
dtypes = {'Study Name': str, 'Best Score': float, 'Parameters': str}
new_row = new_row.astype(dtypes)

# Exclude empty or all-NA columns before concatenation
df_results = df_results.dropna(axis=1, how='all')

# Concatenate the new row to the existing DataFrame
df_results = pd.concat([df_results, new_row], ignore_index=True, sort=False)

# Save the updated results
df_results.to_csv(results_file, index=False)


