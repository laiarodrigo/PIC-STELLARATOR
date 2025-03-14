import numpy as np
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the data from the SQLite database
conn = sqlite3.connect('../../../../data/nfp2/nfp2_combined.db')  # Adjust the path to your database file
query = "SELECT * FROM stellarators_combined"  # Adjust your query as needed
data_df = pd.read_sql_query(query, conn)
conn.close()

# Clean the data
data_df_clean = data_df[data_df['convergence'] == 1]
data_df_clean = data_df_clean.dropna(subset=['quasisymmetry'])

# Define features (X) and target (Y)
X = data_df_clean[['rbc_1_0', 'rbc_m1_1', 'rbc_0_1', 'rbc_1_1', 'zbs_1_0', 'zbs_m1_1', 'zbs_0_1', 'zbs_1_1']]
Y = np.log(data_df_clean['quasisymmetry'])

# Normalize the features and target
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.values.reshape(-1, 1)).flatten()

# Split the data into training and testing sets
features_no_outliers, test_features_no_outliers, target_no_outliers, test_target_no_outliers = train_test_split(
    X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Define and compile a model with 4 hidden layers
def build_model_with_hidden_layers():
    # model = Sequential()
    # model.add(Dense(512, activation='relu', input_shape=(X.shape[1],)))  
    # model.add(Dense(256, activation='relu'))  
    # model.add(Dense(128, activation='relu')) 
    # model.add(Dense(64, activation='relu'))  
    # model.add(Dense(16, activation='relu'))  
    # model.add(Dense(1))  
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))  
    model.add(Dense(128, activation='relu'))  
    model.add(Dense(64, activation='relu')) 
    model.add(Dense(16, activation='relu'))  
    model.add(Dense(1)) 
    
    # Define Adam optimizer with a custom learning rate
    optimizer = Adam(learning_rate=0.001)  # You can adjust the learning rate as needed
    
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    return model

# Train the model with 4 hidden layers
model_with_hidden_layers = build_model_with_hidden_layers()
model_with_hidden_layers.fit(features_no_outliers, target_no_outliers,
                             epochs=300,  # Specify the number of epochs
                             batch_size=500,  # Specify the batch size
                             validation_split=0.2)

# Make predictions on the test set
predictions = model_with_hidden_layers.predict(test_features_no_outliers)

# Evaluate the model
mse = mean_squared_error(test_target_no_outliers, predictions)
r2 = r2_score(test_target_no_outliers, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Save the trained model
model_with_hidden_layers.save('520_normalizes.keras')
