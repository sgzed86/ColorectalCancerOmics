import pandas as pd
import os

# List of filtered TSV files (separating somatic mutation data)
filtered_files = [
    'filtered_clinical_dataframe.tsv',
    'filtered_miRNA_dataframe.tsv',
    'filtered_proteomics_dataframe.tsv',
    'filtered_phosphoproteomics_dataframe.tsv',
    'filtered_transcriptomics_dataframe.tsv'
]

def load_and_prepare_dataset(file_path):
    """
    Load a dataset and prepare it for merging by setting the patient ID as index
    """
    try:
        # Read the TSV file
        df = pd.read_csv(file_path, sep='\t')
        
        # Get the original column names
        original_columns = df.columns.tolist()
        
        # Add file prefix to column names (except patient ID column) to avoid duplicates
        prefix = file_path.replace('filtered_', '').replace('_dataframe.tsv', '')
        new_columns = [original_columns[0]] + [f"{prefix}_{col}" for col in original_columns[1:]]
        df.columns = new_columns
        
        # Set patient ID as index for merging
        df.set_index(df.columns[0], inplace=True)
        
        print(f"Loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def process_somatic_mutations(somatic_file, patient_ids):
    """
    Process somatic mutation data in chunks and create mutation count features
    """
    try:
        print("\nProcessing somatic mutation data...")
        # Initialize an empty DataFrame for mutation counts
        mutation_counts = pd.DataFrame(index=patient_ids)
        mutation_counts['somatic_total_mutations'] = 0
        
        # Read somatic mutation data in chunks
        chunk_size = 10000
        chunks = pd.read_csv(somatic_file, sep='\t', low_memory=False)
        
        # Get patient column name (first column)
        patient_col = chunks.columns[0]
        
        # Count mutations per patient
        patient_mutation_counts = chunks[patient_col].value_counts()
        
        # Update total mutation counts
        for patient in patient_ids:
            if patient in patient_mutation_counts.index:
                mutation_counts.loc[patient, 'somatic_total_mutations'] = patient_mutation_counts[patient]
        
        # Process mutation types if available
        if 'Variant_Classification' in chunks.columns:
            # Get counts by mutation type for each patient
            mutation_type_counts = chunks.groupby([patient_col, 'Variant_Classification']).size().unstack(fill_value=0)
            
            # Add mutation type columns to our counts dataframe
            for col in mutation_type_counts.columns:
                col_name = f'somatic_count_{col}'
                mutation_counts[col_name] = 0
                for patient in patient_ids:
                    if patient in mutation_type_counts.index:
                        mutation_counts.loc[patient, col_name] = mutation_type_counts.loc[patient, col]
        
        print("Completed processing somatic mutation data")
        return mutation_counts
    
    except Exception as e:
        print(f"Error processing somatic mutations: {str(e)}")
        return None

def main():
    print("Loading and preparing datasets...")
    
    # Initialize merged_df as None
    merged_df = None
    
    # Load and merge regular datasets
    for file in filtered_files:
        if os.path.exists(file):
            current_df = load_and_prepare_dataset(file)
            
            if current_df is not None:
                if merged_df is None:
                    merged_df = current_df
                else:
                    # Merge with existing dataframe
                    merged_df = merged_df.join(current_df, how='inner')
                print(f"Merged dataset shape: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        else:
            print(f"Warning: {file} not found")
    
    if merged_df is not None:
        # Process somatic mutation data if available
        somatic_file = 'filtered_somatic_mutation_dataframe.tsv'
        if os.path.exists(somatic_file):
            mutation_counts = process_somatic_mutations(somatic_file, merged_df.index)
            if mutation_counts is not None:
                # Merge mutation counts with the main dataset
                merged_df = merged_df.join(mutation_counts, how='inner')
                print(f"Added somatic mutation features. New shape: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        
        # Save the merged dataset
        output_file = 'merged_complete_dataset.tsv'
        merged_df.to_csv(output_file, sep='\t')
        print(f"\nFinal merged dataset saved to {output_file}")
        print(f"Final dataset shape: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        
        # Save column names for reference
        with open('merged_dataset_columns.txt', 'w') as f:
            f.write('\n'.join(merged_df.columns.tolist()))
        print("Column names saved to merged_dataset_columns.txt")
    else:
        print("No datasets were successfully merged")

if __name__ == "__main__":
    main() 