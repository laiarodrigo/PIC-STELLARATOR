import numpy as np
import sqlite3
from simsopt.mhd import Vmec
#from simsopt.mhd import QuasisymmetryRatioResidual
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_base_support.vmecPlot2 import main as vmecPlot2
# Assume other necessary imports are included

def run_vmec_simulation_with_plots(record_id):
    # Connect to the SQLite database
    conn = sqlite3.connect('data/nfp2/nfp2.db') 
    cursor = conn.cursor()
    
    # Retrieve the specific record from the database
    cursor.execute("SELECT * FROM stellarators WHERE id=?", (record_id,))
    record = cursor.fetchone()
    conn.close()

    if record is None:
        print("Record not found.")
        return
    
    this_path = Path(__file__).resolve().parent

    # Load the original VMEC input file
    input_vmec_file_original = str(this_path / 'data/nfp2/input.nfp2_QA')
    stel = Vmec(input_vmec_file_original, verbose=False)
    surf = stel.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
    surf.fix("rc(0,0)") # Fix major radius to be the same
    surf.x=np.concatenate((record[2:6], record[7:11]),axis=0)
    stel.run()
    ## Run initial stellarator and plot
    vmecPlot2(stel.output_file)
    return surf.x

# Step 1: Connect to the SQLite database
conn = sqlite3.connect('data/nfp2/nfp2.db')  # Adjust the path to your database file

# Step 2 & 3: Query the database and load the data into a pandas DataFrame
query = "SELECT * FROM stellarators"  # Adjust your query as needed
data_df = pd.read_sql_query(query, conn)

X = data_df[['rbc_0_0', 'rbc_1_0', 'rbc_m1_1', 'rbc_0_1', 'rbc_1_1', 'zbs_0_0', 'zbs_1_0', 'zbs_m1_1', 'zbs_0_1', 'zbs_1_1']] 
y = data_df[['quasisymmetry', 'quasiisodynamic', 'rotational_transform', 'inverse_aspect_ratio', 'mean_local_magnetic_shear', 'vacuum_magnetic_well', 'maximum_elongation', 'mirror_ratio']] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the MinMaxScaler to scale data between -1 and 1
xscaler = Normalizer(norm='l2')
yscaler = Normalizer(norm='l2') # Oque usar aqui? (MinMaxScaler ou StandardScaler?)

y_train_scaled = yscaler.fit_transform(y_train)
y_test_scaled = yscaler.transform(y_test)
X_train_scaled = xscaler.fit_transform(X_train)
X_test_scaled = xscaler.transform(X_test)
print(y_train_scaled.shape)
print(y_test_scaled.shape)
print(X_train_scaled.shape)
print(X_test_scaled.shape)

####################SUMMARY STATISTICS####################
'''
summary_stats = data_df.describe()
print(summary_stats)

##########################################################

####################DATA DISTRIBUTION#####################

plt.figure(figsize=(10, 4))
for column in y.columns:
    sns.histplot(y[column], bins=20, kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()
    
##########################################################


####################CORRELATION MATRIX####################

# Combine the features and targets for correlation analysis
combined_df = y
combined_df_scaled = pd.DataFrame(y_test_scaled)
# Calculate the correlation matrix
corr_matrix = combined_df.corr()
corr_matrix_scaled = combined_df_scaled.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(14, 12))  # You can adjust the figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=.5)
plt.title('Correlation Map of Targets (unscaled)')
plt.show()

sns.heatmap(corr_matrix_scaled, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=.5)
plt.title('Correlation Map of Targets (scaled)')
plt.show()

sns.pairplot(y, diag_kind='kde') # Existem clusters?
plt.show()
'''
##########################################################
'''
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Assuming 'X' is your features DataFrame and 'y' is a Series or DataFrame containing the target variable 'quasisymmetry'
quasisymmetry = y['quasisymmetry'] if isinstance(y, pd.DataFrame) else y  # Correctly extract 'quasisymmetry' if 'y' is a DataFrame

# Perform K-Means Clustering on Features
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_clusters = kmeans.fit_predict(X)  # Store cluster labels from K-Means

# Perform Agglomerative Hierarchical Clustering
agglo = AgglomerativeClustering(n_clusters=3)  # Adjust n_clusters as needed
agglo_clusters = agglo.fit_predict(X)  # Store cluster labels from Agglomerative Clustering

# Create a DataFrame for plotting without modifying 'X'
plot_df = X.copy()
plot_df['Quasisymmetry'] = quasisymmetry  # Add 'quasisymmetry' to the DataFrame
plot_df['KMeans_Cluster'] = kmeans_clusters  # Add K-Means cluster labels
plot_df['Agglo_Cluster'] = agglo_clusters  # Add Agglomerative Clustering labels

# Plotting setup for both K-Means and Agglomerative Clustering results
n_features = X.shape[1]
n_cols = 2  # Desired number of columns in the subplot grid
n_rows = n_features // n_cols + (n_features % n_cols > 0)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

best_score = -1
best_kmeans = None
best_agglo = None

for feature in X.columns:
    # Perform K-Means Clustering on Features
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_clusters = kmeans.fit_predict(X[[feature]])  # Store cluster labels from K-Means
    
    # Perform Agglomerative Hierarchical Clustering
    agglo = AgglomerativeClustering(n_clusters=3)  # Adjust n_clusters as needed
    agglo_clusters = agglo.fit_predict(X[[feature]])  # Store cluster labels from Agglomerative Clustering
    
    # Create a DataFrame for plotting without modifying 'X'
    plot_df = X.copy()
    plot_df['Quasisymmetry'] = quasisymmetry  # Add 'quasisymmetry' to the DataFrame
    plot_df['KMeans_Cluster'] = kmeans_clusters  # Add K-Means cluster labels
    plot_df['Agglo_Cluster'] = agglo_clusters  # Add Agglomerative Clustering labels
    
    plt.figure(figsize=(12, 6))
    
    # Plotting K-Means Clustering
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=plot_df, x=feature, y='Quasisymmetry', hue='KMeans_Cluster', palette='viridis', alpha=0.6)
    plt.title(f'KMeans Clustering - {feature}')
    plt.ylim(0, 3)  # Set y-axis limits
    
    # Plotting Agglomerative Clustering
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=plot_df, x=feature, y='Quasisymmetry', hue='Agglo_Cluster', palette='viridis', alpha=0.6)
    plt.title(f'Agglomerative Clustering - {feature}')
    plt.ylim(0, 3)  # Set y-axis limits
    
    plt.tight_layout()
    plt.show()
'''
#################### TREE ####################

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

# Initialize Gradient Boosted Trees regressor
gbt_regressor = GradientBoostingRegressor(random_state=42)

# Fit the Gradient Boosted Trees regressor
gbt_regressor.fit(X_train_scaled, y_train_scaled[:, 0])

# Predict quasisymmetry values on the test set
y_pred = gbt_regressor.predict(X_test_scaled)

# Calculate Mean Squared Error (MSE) as a measure of model performance
mse = mean_squared_error(y_test_scaled[:,0], y_pred)
print("Mean Squared Error (MSE):", mse)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=X.columns, y=gbt_regressor.feature_importances_, palette='viridis')
plt.title('Feature Importance (Gradient Boosted Trees)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

tree_id = 0 # You can change this to extract a different tree
individual_tree = gbt_regressor.estimators_[tree_id][0]

# Plot the individual tree
plt.figure(figsize=(20,10))
plot_tree(individual_tree, filled=True, feature_names=X.columns)
plt.show()

# Initialize Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)

# Fit the Decision Tree Regressor
dt_regressor.fit(X_train_scaled, y_train_scaled[:,0])

# Predict quasisymmetry values on the test set
y_pred = dt_regressor.predict(X_test_scaled)

# Calculate Mean Squared Error (MSE) as a measure of model performance
mse = mean_squared_error(y_test_scaled[:,0], y_pred)
print("Mean Squared Error (MSE):", mse)

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(dt_regressor, filled=True, feature_names=X.columns)
plt.show()


