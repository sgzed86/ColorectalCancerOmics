import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import f_classif
import os
from typing import List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

def load_common_identifiers(file_path: str = 'common_identifiers.txt') -> Set[str]:
    """
    Load the list of common patient identifiers.
    """
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a TSV file and return a pandas DataFrame
    """
    # For large files, read only the first row to get column names
    if os.path.getsize(file_path) > 1e6:  # If file is larger than 1MB
        cols = pd.read_csv(file_path, sep='\t', nrows=0).columns
        # Check if we need to specify the index column
        if 'Unnamed: 0' in cols:
            df = pd.read_csv(file_path, sep='\t', index_col=0)
        else:
            df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path, sep='\t')
    
    # Handle 'Name' column if present
    if 'Name' in df.columns and 'Patient_ID' not in df.columns:
        df = df.rename(columns={'Name': 'Patient_ID'})
    
    return df

def identify_data_types(df: pd.DataFrame, patient_id_col: str, exclude_cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Identify continuous and discrete columns in the DataFrame
    """
    continuous_cols = []
    discrete_cols = []
    
    for col in df.columns:
        if col == patient_id_col or col in exclude_cols:
            continue
        
        # Remove columns with too many missing values
        missing_ratio = df[col].isna().sum() / len(df)
        if missing_ratio > 0.3:  # Skip columns with more than 30% missing values
            continue
            
        # Check if column has numeric values
        try:
            # Convert to numeric, coerce errors to NaN
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            if numeric_data.isna().all():  # If all values are NaN after conversion
                discrete_cols.append(col)
                continue
                
            # If column has less than 10 unique values or all values are integers, consider it discrete
            if numeric_data.nunique() < 10 or numeric_data.dropna().apply(float.is_integer).all():
                discrete_cols.append(col)
            else:
                continuous_cols.append(col)
        except:
            discrete_cols.append(col)
    
    return continuous_cols, discrete_cols

def analyze_discrete_features(df: pd.DataFrame, discrete_cols: List[str], target_col: str, 
                            alpha: float = 0.05) -> List[str]:
    """
    Perform Fisher's exact test and Chi-square test for discrete features
    """
    significant_features = []
    
    for col in discrete_cols:
        try:
            # Remove missing values
            data = df[[col, target_col]].dropna()
            if len(data) < 10:  # Skip if too few samples
                continue
                
            # Create contingency table
            contingency = pd.crosstab(data[col], data[target_col])
            
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue
                
            # Fisher's exact test for small samples
            if contingency.shape[0] * contingency.shape[1] <= 20:  # Small contingency table
                _, fisher_p = stats.fisher_exact(contingency)
                if fisher_p < alpha:
                    significant_features.append(col)
                    continue
            
            # Chi-square test for larger samples
            if contingency.size > 1:  # Check if contingency table is not empty
                _, chi2_p, _, _ = stats.chi2_contingency(contingency)
                if chi2_p < alpha:
                    significant_features.append(col)
        except Exception as e:
            continue
    
    return significant_features

def analyze_continuous_features(df: pd.DataFrame, continuous_cols: List[str], target_col: str,
                              alpha: float = 0.05) -> List[str]:
    """
    Perform t-test, Mann-Whitney U test, and ANOVA for continuous features
    """
    significant_features = []
    
    # Process features in batches to handle large datasets
    batch_size = 1000
    for i in range(0, len(continuous_cols), batch_size):
        batch_cols = continuous_cols[i:i+batch_size]
        
        for col in batch_cols:
            try:
                # Convert to numeric and remove missing values
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                data = pd.DataFrame({col: numeric_data, target_col: df[target_col]}).dropna()
                
                if len(data) < 10:  # Skip if too few samples
                    continue
                    
                # Split data by target classes
                groups = [group[col].values for name, group in data.groupby(target_col)]
                
                if len(groups) < 2:  # Skip if not enough groups
                    continue
                
                # For binary classification
                if len(groups) == 2:
                    # Student's t-test
                    _, ttest_p = stats.ttest_ind(groups[0], groups[1], nan_policy='omit')
                    
                    # Mann-Whitney U test
                    try:
                        _, mw_p = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                    except ValueError:
                        mw_p = 1.0
                    
                    if ttest_p < alpha or mw_p < alpha:
                        significant_features.append(col)
                        continue
                
                # ANOVA for all cases
                try:
                    _, anova_p = stats.f_oneway(*groups)
                    if anova_p < alpha:
                        significant_features.append(col)
                except ValueError:
                    continue
                    
            except Exception as e:
                continue
        
        # Print progress for large datasets
        if len(continuous_cols) > batch_size:
            print(f"Processed {min(i+batch_size, len(continuous_cols))}/{len(continuous_cols)} features")
    
    return significant_features

def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target variables for survival and recurrence
    """
    # Create survival target (1 = deceased, 0 = alive)
    survival_col = [col for col in df.columns if 'survival status' in col.lower()]
    if survival_col:
        df['survival_target'] = df[survival_col[0]].fillna(0)
    
    # Create recurrence target (1 = recurrence, 0 = no recurrence)
    recurrence_col = [col for col in df.columns if 'recurrence status' in col.lower()]
    if recurrence_col:
        df['recurrence_target'] = df[recurrence_col[0]].fillna(0)
    
    return df

def process_clinical_data(file_path: str, patient_id_col: str = 'Patient_ID', alpha: float = 0.05) -> pd.DataFrame:
    """
    Process clinical data file
    """
    print(f"\nProcessing clinical file: {file_path}")
    
    # Load data and common identifiers
    df = load_data(file_path)
    common_ids = load_common_identifiers()
    print(f"Original shape: {df.shape}")
    
    # Filter for common identifiers
    df = df[df[patient_id_col].isin(common_ids)]
    print(f"Shape after filtering for common identifiers: {df.shape}")
    
    # Create target variables
    df = create_target_variables(df)
    target_cols = ['survival_target', 'recurrence_target']
    
    # Columns to exclude from feature selection
    exclude_cols = target_cols + [col for col in df.columns if any(x in col.lower() for x in 
                                ['survival', 'recurrence', 'days', 'date'])]
    
    # Identify data types
    continuous_cols, discrete_cols = identify_data_types(df, patient_id_col, exclude_cols)
    print(f"Found {len(continuous_cols)} continuous and {len(discrete_cols)} discrete features")
    
    significant_features = [patient_id_col]  # Always keep patient ID
    
    # Perform feature selection for each target
    for target_col in target_cols:
        print(f"\nAnalyzing features for target: {target_col}")
        
        # Perform feature selection
        sig_discrete = analyze_discrete_features(df, discrete_cols, target_col, alpha)
        sig_continuous = analyze_continuous_features(df, continuous_cols, target_col, alpha)
        
        print(f"Significant discrete features: {len(sig_discrete)}")
        print(f"Significant continuous features: {len(sig_continuous)}")
        
        significant_features.extend([f for f in sig_discrete + sig_continuous if f not in significant_features])
    
    # Create cleaned DataFrame with all common IDs
    all_patients_df = pd.DataFrame({patient_id_col: list(common_ids)})
    cleaned_df = df[significant_features + target_cols]
    cleaned_df = all_patients_df.merge(cleaned_df, on=patient_id_col, how='left')
    print(f"\nFinal shape after feature selection (including all common IDs): {cleaned_df.shape}")
    
    return cleaned_df

def process_omics_data(file_path: str, clinical_df: pd.DataFrame, patient_id_col: str = 'Patient_ID', 
                      alpha: float = 0.05) -> pd.DataFrame:
    """
    Process omics data file using target variables from clinical data
    """
    print(f"\nProcessing omics file: {file_path}")
    
    # Load data and common identifiers
    df = load_data(file_path)
    common_ids = load_common_identifiers()
    print(f"Original shape: {df.shape}")
    
    # Ensure patient ID column exists
    if patient_id_col not in df.columns:
        print(f"Warning: {patient_id_col} not found in columns. Available columns: {df.columns[:5]}...")
        return None
    
    # Filter for common identifiers
    df = df[df[patient_id_col].isin(common_ids)]
    print(f"Shape after filtering for common identifiers: {df.shape}")
    
    # Merge with clinical data to get target variables
    df = df.merge(clinical_df[[patient_id_col, 'survival_target', 'recurrence_target']], 
                 on=patient_id_col, how='inner')
    print(f"Shape after merging with clinical data: {df.shape}")
    
    target_cols = ['survival_target', 'recurrence_target']
    
    # Identify data types (assume all features are continuous in omics data)
    feature_cols = [col for col in df.columns if col not in [patient_id_col] + target_cols]
    
    significant_features = [patient_id_col]  # Always keep patient ID
    
    # Perform feature selection for each target
    for target_col in target_cols:
        print(f"\nAnalyzing features for target: {target_col}")
        sig_features = analyze_continuous_features(df, feature_cols, target_col, alpha)
        print(f"Significant features: {len(sig_features)}")
        significant_features.extend([f for f in sig_features if f not in significant_features])
    
    # Create cleaned DataFrame with all common IDs
    all_patients_df = pd.DataFrame({patient_id_col: list(common_ids)})
    cleaned_df = df[significant_features + target_cols]
    cleaned_df = all_patients_df.merge(cleaned_df, on=patient_id_col, how='left')
    print(f"\nFinal shape after feature selection (including all common IDs): {cleaned_df.shape}")
    
    return cleaned_df

def main():
    """
    Main function to process all data files
    """
    # Parameters
    alpha = 0.05
    patient_id_col = 'Patient_ID'
    
    # Process clinical data first
    clinical_file = 'clinical_dataframe_filtered.tsv'
    if os.path.exists(clinical_file):
        clinical_df = process_clinical_data(clinical_file, patient_id_col, alpha)
        output_file = f"cleaned_{clinical_file}"
        clinical_df.to_csv(output_file, sep='\t', index=False)
        print(f"Saved cleaned clinical data to: {output_file}")
    else:
        print(f"Clinical file not found: {clinical_file}")
        return
    
    # Process omics data files
    omics_files = [
        'miRNA_dataframe_filtered.tsv',
        'proteomics_dataframe_filtered.tsv',
        'phosphoproteomics_dataframe_filtered.tsv',
        'transcriptomics_dataframe_filtered.tsv',
        'somatic_mutation_dataframe_filtered.tsv'
    ]
    
    for file in omics_files:
        if os.path.exists(file):
            cleaned_df = process_omics_data(file, clinical_df, patient_id_col, alpha)
            if cleaned_df is not None:
                output_file = f"cleaned_{file}"
                cleaned_df.to_csv(output_file, sep='\t', index=False)
                print(f"Saved cleaned data to: {output_file}")
        else:
            print(f"File not found: {file}")

if __name__ == "__main__":
    main() 