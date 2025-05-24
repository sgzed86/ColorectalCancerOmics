import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

def preprocess_data(df):
    """
    Preprocess the data by handling infinite values and missing data
    """
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # For each column, replace NaN with median of that column
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    return df

def calculate_fisher_ratio(X, y):
    """
    Calculate Fisher's discriminant ratio for each feature.
    FDR = (μ1 - μ2)² / (σ1² + σ2²)
    where μ1, μ2 are means and σ1², σ2² are variances of the two classes
    """
    # Split data by class
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    
    # Calculate means and variances for each feature
    mean_0 = np.mean(X_0, axis=0)
    mean_1 = np.mean(X_1, axis=0)
    var_0 = np.var(X_0, axis=0)
    var_1 = np.var(X_1, axis=0)
    
    # Calculate Fisher's ratio for each feature
    numerator = (mean_0 - mean_1) ** 2
    denominator = var_0 + var_1
    
    # Handle division by zero and extremely large values
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)  # Avoid division by zero
    fisher_ratios = numerator / denominator
    
    # Cap extremely large values
    fisher_ratios = np.minimum(fisher_ratios, 1e3)  # Cap at 1000
    
    return fisher_ratios

def evaluate_dataset(X, y, dataset_name, feature_names):
    """
    Evaluate dataset using Fisher's ratio and F1 score
    """
    # Calculate Fisher's ratio
    fisher_ratios = calculate_fisher_ratio(X, y)
    max_fisher_ratio = np.max(fisher_ratios)
    avg_fisher_ratio = np.mean(fisher_ratios)
    
    # Get top discriminative features
    top_features_idx = np.argsort(fisher_ratios)[-10:][::-1]
    
    # Calculate F1 score using cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nResults for {dataset_name}:")
    print(f"Maximum Fisher's Discriminant Ratio: {max_fisher_ratio:.4f}")
    print(f"Average Fisher's Discriminant Ratio: {avg_fisher_ratio:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nTop 10 discriminative features:")
    for idx in top_features_idx:
        print(f"{feature_names[idx]}: {fisher_ratios[idx]:.4f}")
    
    return fisher_ratios, f1, top_features_idx

# Load original dataset
print("Loading original dataset...")
original_df = pd.read_csv('joined_recurrence_dataframe.tsv', sep='\t')
original_y = original_df['recurrence_target']
original_X = original_df.select_dtypes(include=[np.number])
original_X = original_X.drop('recurrence_target', axis=1)

# Load balanced dataset
print("Loading balanced dataset...")
balanced_df = pd.read_csv('balanced_recurrence_dataframe.tsv', sep='\t')
balanced_y = balanced_df['recurrence_target']
balanced_X = balanced_df.select_dtypes(include=[np.number])
balanced_X = balanced_X.drop('recurrence_target', axis=1)

# Preprocess both datasets
print("Preprocessing datasets...")
original_X = preprocess_data(original_X)
balanced_X = preprocess_data(balanced_X)

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
original_X_scaled = scaler.fit_transform(original_X)
balanced_X_scaled = scaler.fit_transform(balanced_X)

# Store feature names
feature_names = original_X.columns

# Evaluate original dataset
print("\nEvaluating original dataset...")
original_fisher_ratios, original_f1, original_top_idx = evaluate_dataset(
    original_X_scaled, original_y, "Original Dataset", feature_names
)

# Evaluate balanced dataset
print("\nEvaluating balanced dataset...")
balanced_fisher_ratios, balanced_f1, balanced_top_idx = evaluate_dataset(
    balanced_X_scaled, balanced_y, "Balanced Dataset", feature_names
)

# Save detailed results
results_df = pd.DataFrame({
    'Feature': feature_names,
    'Original_Fisher_Ratio': original_fisher_ratios,
    'Balanced_Fisher_Ratio': balanced_fisher_ratios
})
results_df = results_df.sort_values('Balanced_Fisher_Ratio', ascending=False)
results_df.to_csv('fisher_ratio_analysis.tsv', sep='\t', index=False)

print("\nDetailed Fisher's ratio analysis saved to 'fisher_ratio_analysis.tsv'")

# Calculate improvement metrics
print("\nImprovement Metrics:")
if original_f1 == 0:
    print("F1 Score Improvement: Cannot calculate (original F1 score was 0)")
else:
    f1_improvement = (balanced_f1 - original_f1) / original_f1 * 100
    print(f"F1 Score Improvement: {f1_improvement:.2f}%")

fisher_ratio_improvement = (np.mean(balanced_fisher_ratios) - np.mean(original_fisher_ratios)) / np.mean(original_fisher_ratios) * 100
print(f"Average Fisher's Ratio Improvement: {fisher_ratio_improvement:.2f}%") 