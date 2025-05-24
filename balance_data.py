import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df):
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # For each column, replace NaN with median of that column
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    return df

# Read the recurrence dataframe
print("Loading data...")
df = pd.read_csv('joined_recurrence_dataframe.tsv', sep='\t')  # Don't set index_col here

# Store original Patient IDs
original_indices = df['Patient_ID'].values

# Get features and target
y = df['recurrence_target']
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = numeric_cols.drop('recurrence_target')
X = df[numeric_cols]

# Preprocess the data
print("Preprocessing data...")
X = preprocess_data(X)

# Print initial class distribution
print("\nInitial class distribution:")
print(y.value_counts())

# Apply SMOTE to balance the dataset
print("\nApplying SMOTE oversampling...")
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Print balanced class distribution
print("\nBalanced class distribution:")
print(pd.Series(y_balanced).value_counts())

# Create synthetic Patient IDs for new samples
n_original = len(original_indices)
n_synthetic = len(y_balanced) - n_original
synthetic_ids = [f'SYNTH_{i+1:03d}' for i in range(n_synthetic)]
balanced_patient_ids = np.concatenate([original_indices, synthetic_ids])

# Standardize the features
print("\nStandardizing features...")
scaler = StandardScaler()
X_balanced_scaled = scaler.fit_transform(X_balanced)

# Perform PCA on balanced data
print("Performing PCA...")
pca = PCA()
X_pca = pca.fit_transform(X_balanced_scaled)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Create DataFrame with balanced data
balanced_df = pd.DataFrame(X_balanced, columns=X.columns)
balanced_df.insert(0, 'Patient_ID', balanced_patient_ids)  # Add Patient_ID as first column
balanced_df['recurrence_target'] = y_balanced

# Save balanced dataset
print("\nSaving balanced dataset...")
balanced_df.to_csv('balanced_recurrence_dataframe.tsv', sep='\t', index=False)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio - Balanced Data')
plt.grid(True)
plt.savefig('balanced_explained_variance.png')
plt.close()

# Plot first two principal components with balanced data
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_balanced, cmap='viridis')
plt.colorbar(scatter, label='Recurrence Target')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('First Two Principal Components - Balanced Data')
plt.savefig('balanced_pca_scatter.png')
plt.close()

# Create DataFrame with PCA results
pca_df = pd.DataFrame(
    X_pca[:, :50],  # Keep first 50 components
    columns=[f'PC{i+1}' for i in range(50)]
)
pca_df.insert(0, 'Patient_ID', balanced_patient_ids)  # Add Patient_ID as first column
pca_df['recurrence_target'] = y_balanced

# Save PCA results
pca_df.to_csv('balanced_pca_results.tsv', sep='\t', index=False)

# Print PCA results
print("\nPCA Results for balanced data:")
print(f"Number of original features: {X.shape[1]}")
print("\nExplained variance ratio by first 10 components:")
for i in range(10):
    print(f"PC{i+1}: {explained_variance_ratio[i]:.4f} ({cumulative_variance_ratio[i]:.4f} cumulative)")

# Find number of components needed for 90% variance
n_components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
print(f"\nNumber of components needed for 90% variance: {n_components_90}")

# Print information about synthetic samples
print(f"\nNumber of original samples: {n_original}")
print(f"Number of synthetic samples: {n_synthetic}")
print("Synthetic Patient IDs format: SYNTH_001, SYNTH_002, etc.")

# Additional visualization: Plot class distribution before and after balancing
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x=y)
plt.title('Original Class Distribution')
plt.xlabel('Recurrence Target')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.countplot(x=y_balanced)
plt.title('Balanced Class Distribution')
plt.xlabel('Recurrence Target')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('class_distribution_comparison.png')
plt.close()

print("\nAll visualizations have been saved.") 